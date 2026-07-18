"""Submit, inspect, and download standalone Claude labels for BioASQ runs.

This is intentionally a local post-processing tool.  It neither changes the
generation pipeline nor writes into ``bioasq_eval`` produced on the server.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl, write_jsonl
from archehr_sebaseline.evaluation.claude_judge import (
    CLAUDE_JUDGE_SYSTEM_PROMPT,
    build_semantic_quality_prompt,
    extract_label,
)


# Preserve legacy three-level outputs. The current protocol uses independent
# binary labels for low-temperature main answers.
OUTPUT_DIRNAME = "claude_binary_main_answer_judge"
MANIFEST_NAME = "request_manifest.jsonl"
METADATA_NAME = "batch_metadata.json"
LABELS_NAME = "claude_generation_labels.csv"
RETRY_MANIFEST_NAME = "retry_request_manifest.jsonl"
RETRY_METADATA_NAME = "retry_batch_metadata.json"
LABEL_FIELDS = [
    "custom_id",
    "example_id",
    "sample_id",
    "bioasq_type",
    "reference_key",
    "prompt_sha256",
    "result_type",
    "label",
    "label_valid",
    "stop_reason",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_client(env_file: Path):
    load_dotenv(env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(f"ANTHROPIC_API_KEY was not found in {env_file}.")
    from anthropic import Anthropic

    return Anthropic()


def judge_directory(run_dir: Path) -> Path:
    return run_dir / OUTPUT_DIRNAME


def metadata_path(run_dir: Path) -> Path:
    return judge_directory(run_dir) / METADATA_NAME


def retry_metadata_path(run_dir: Path) -> Path:
    return judge_directory(run_dir) / RETRY_METADATA_NAME


def load_generation_requests(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create one semantic-quality request per stored generated answer."""

    examples = {str(row["id"]): row for row in read_jsonl(run_dir / "examples.jsonl")}
    generations = sorted(
        read_jsonl(run_dir / "best_generations.jsonl"),
        key=lambda row: (str(row["example_id"]), int(row["sample_id"])),
    )
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for generation in generations:
        example_id = str(generation["example_id"])
        sample_id = int(generation["sample_id"])
        example = examples.get(example_id)
        if example is None:
            raise KeyError(f"Generation {example_id}/{sample_id} has no matching example.")
        bioasq_type = str(example.get("bioasq_type") or "").lower()
        reference_key = "exact_answers" if bioasq_type in {"factoid", "list"} else "ideal_answers"
        references = [str(value).strip() for value in example.get(reference_key, []) if str(value).strip()]
        if not references and reference_key != "ideal_answers":
            references = [str(value).strip() for value in example.get("ideal_answers", []) if str(value).strip()]
        if not references and str(example.get("gold_answer") or "").strip():
            references = [str(example["gold_answer"]).strip()]
        candidate = str(generation.get("clean_answer") or generation.get("raw_answer") or "").strip()
        if not candidate:
            raise ValueError(f"Generation {example_id}/{sample_id} has an empty answer.")
        prompt = build_semantic_quality_prompt(
            question=str(example.get("question") or "").strip(),
            references=references,
            candidate=candidate,
            bioasq_type=bioasq_type,
        )
        custom_id = f"e{example_id}-s{sample_id}"
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": "__MODEL__",
                    "max_tokens": "__MAX_TOKENS__",
                    "system": CLAUDE_JUDGE_SYSTEM_PROMPT,
                    "output_config": {"effort": "__EFFORT__"},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "example_id": example_id,
                "sample_id": sample_id,
                "bioasq_type": bioasq_type,
                "reference_key": reference_key,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        )
    if not requests:
        raise ValueError(f"No cleaned generations found in {run_dir}.")
    if len({request["custom_id"] for request in requests}) != len(requests):
        raise ValueError(f"Duplicate Claude custom IDs found for {run_dir}.")
    return requests, manifest


def submit(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        output_dir = judge_directory(run_dir)
        metadata_file = metadata_path(run_dir)
        if metadata_file.exists() and not args.overwrite:
            raise FileExistsError(f"A Claude batch is already recorded for {run_dir}: {metadata_file}")
        requests, manifest = load_generation_requests(run_dir)
        for request in requests:
            params = request["params"]
            params["model"] = args.model
            params["max_tokens"] = args.max_tokens
            params["output_config"]["effort"] = args.effort
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / MANIFEST_NAME
        if manifest_path.exists() and not args.overwrite:
            existing_manifest = read_jsonl(manifest_path)
            if existing_manifest != manifest:
                raise ValueError(
                    f"Prepared manifest differs from the current requests for {run_dir}; "
                    "review it or pass --overwrite deliberately."
                )
        else:
            write_jsonl(manifest, manifest_path, overwrite=args.overwrite)
        batch = client.messages.batches.create(requests=requests)
        metadata = {
            "batch_id": batch.id,
            "run_dir": str(run_dir),
            "submitted_at_utc": utc_now(),
            "model": args.model,
            "effort": args.effort,
            "max_tokens": args.max_tokens,
            "num_requests": len(requests),
        }
        metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"submitted {run_dir.name}: {batch.id} ({len(requests)} requests)")
    return 0


def prepare(args: argparse.Namespace) -> int:
    """Write reviewable request manifests without contacting the Claude API."""

    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        output_dir = judge_directory(run_dir)
        manifest_path = output_dir / MANIFEST_NAME
        if manifest_path.exists() and not args.overwrite:
            raise FileExistsError(f"Prepared manifest already exists: {manifest_path}")
        requests, manifest = load_generation_requests(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(manifest, manifest_path, overwrite=args.overwrite)
        print(f"prepared {run_dir.name}: {len(requests)} binary Claude requests at {manifest_path}")
    return 0


def read_metadata(run_dir: Path) -> dict[str, Any]:
    path = metadata_path(run_dir)
    if not path.exists():
        raise FileNotFoundError(f"No Claude batch metadata at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))


def configure_requests(requests: list[dict[str, Any]], args: argparse.Namespace) -> None:
    for request in requests:
        params = request["params"]
        params["model"] = args.model
        params["max_tokens"] = args.max_tokens
        params["output_config"]["effort"] = args.effort


def result_row(item: Any, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = manifest.get(item.custom_id)
    if base is None:
        raise KeyError(f"Result has no manifest entry: {item.custom_id}")
    result_type = item.result.type
    row: dict[str, Any] = {
        **base,
        "result_type": result_type,
        "label": "",
        "label_valid": False,
        "stop_reason": "",
        "error": "",
    }
    if result_type == "succeeded":
        row["stop_reason"] = item.result.message.stop_reason
        try:
            row["label"] = extract_label(item.result.message)
            row["label_valid"] = True
        except ValueError as exc:
            row["error"] = str(exc)
    else:
        row["error"] = str(getattr(item.result, "error", result_type))
    return row


def write_label_rows(rows: list[dict[str, Any]], path: Path) -> None:
    rows.sort(key=lambda row: (row["example_id"], int(row["sample_id"])))
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def status(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        metadata = read_metadata(run_dir)
        batch = client.messages.batches.retrieve(metadata["batch_id"])
        counts = batch.request_counts
        print(
            f"{run_dir.name}: {batch.processing_status}; "
            f"processing={counts.processing}, succeeded={counts.succeeded}, "
            f"errored={counts.errored}, expired={counts.expired}, canceled={counts.canceled}"
        )
    return 0


def recover(args: argparse.Namespace) -> int:
    """Restore local tracking after a successful API submission interrupted locally."""

    client = load_client(args.env_file)
    if not args.run_dir or len(args.run_dir) != 1:
        raise ValueError("recover requires exactly one --run_dir.")
    run_dir = args.run_dir[0].resolve()
    output_dir = judge_directory(run_dir)
    metadata_file = metadata_path(run_dir)
    if metadata_file.exists() and not args.overwrite:
        raise FileExistsError(f"A Claude batch is already recorded for {run_dir}: {metadata_file}")
    _, manifest = load_generation_requests(run_dir)
    batch = client.messages.batches.retrieve(args.batch_id)
    counts = batch.request_counts
    received = counts.processing + counts.succeeded + counts.errored + counts.expired + counts.canceled
    if received != len(manifest):
        raise RuntimeError(f"Batch {batch.id} has {received} requests, expected {len(manifest)} for {run_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifest, output_dir / MANIFEST_NAME, overwrite=args.overwrite)
    metadata = {
        "batch_id": batch.id,
        "run_dir": str(run_dir),
        "submitted_at_utc": str(batch.created_at),
        "model": args.model,
        "effort": args.effort,
        "max_tokens": args.max_tokens,
        "num_requests": len(manifest),
    }
    metadata_file.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recovered {run_dir.name}: {batch.id} ({len(manifest)} requests)")
    return 0


def download(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        output_dir = judge_directory(run_dir)
        labels_path = output_dir / LABELS_NAME
        if labels_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {labels_path}; pass --overwrite.")
        metadata = read_metadata(run_dir)
        batch = client.messages.batches.retrieve(metadata["batch_id"])
        if batch.processing_status != "ended":
            raise RuntimeError(f"Batch {batch.id} is {batch.processing_status}; wait until it ends before downloading.")
        manifest = {row["custom_id"]: row for row in read_jsonl(output_dir / MANIFEST_NAME)}
        rows: list[dict[str, Any]] = []
        for item in client.messages.batches.results(batch.id):
            rows.append(result_row(item, manifest))
        if len(rows) != metadata["num_requests"]:
            raise RuntimeError(f"Expected {metadata['num_requests']} results, received {len(rows)}.")
        write_label_rows(rows, labels_path)
        print(f"downloaded {run_dir.name}: {len(rows)} labels to {labels_path}")
    return 0


def retry(args: argparse.Namespace) -> int:
    """Submit only labels that were blank, invalid, or otherwise unsuccessful."""

    client = load_client(args.env_file)
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        output_dir = judge_directory(run_dir)
        labels_path = output_dir / LABELS_NAME
        metadata_file = retry_metadata_path(run_dir)
        if metadata_file.exists() and not args.overwrite:
            raise FileExistsError(f"A retry batch is already recorded for {run_dir}: {metadata_file}")
        with labels_path.open(encoding="utf-8", newline="") as infile:
            invalid_ids = {
                row["custom_id"]
                for row in csv.DictReader(infile)
                if str(row.get("label_valid")).lower() != "true"
            }
        if not invalid_ids:
            print(f"retry skipped {run_dir.name}: all labels are valid")
            continue
        requests, manifest = load_generation_requests(run_dir)
        requests = [row for row in requests if row["custom_id"] in invalid_ids]
        manifest = [row for row in manifest if row["custom_id"] in invalid_ids]
        if len(requests) != len(invalid_ids):
            raise RuntimeError(f"Retry manifest mismatch for {run_dir}: expected {len(invalid_ids)}, got {len(requests)}.")
        configure_requests(requests, args)
        write_jsonl(manifest, output_dir / RETRY_MANIFEST_NAME, overwrite=True)
        batch = client.messages.batches.create(requests=requests)
        metadata_file.write_text(
            json.dumps(
                {
                    "batch_id": batch.id,
                    "run_dir": str(run_dir),
                    "submitted_at_utc": utc_now(),
                    "model": args.model,
                    "effort": args.effort,
                    "max_tokens": args.max_tokens,
                    "num_requests": len(requests),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"retry submitted {run_dir.name}: {batch.id} ({len(requests)} requests)")
    return 0


def download_retry(args: argparse.Namespace) -> int:
    """Merge a completed retry batch into the original label file."""

    client = load_client(args.env_file)
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        output_dir = judge_directory(run_dir)
        labels_path = output_dir / LABELS_NAME
        metadata = json.loads(retry_metadata_path(run_dir).read_text(encoding="utf-8"))
        batch = client.messages.batches.retrieve(metadata["batch_id"])
        if batch.processing_status != "ended":
            raise RuntimeError(f"Retry batch {batch.id} is {batch.processing_status}; wait until it ends before downloading.")
        manifest = {row["custom_id"]: row for row in read_jsonl(output_dir / RETRY_MANIFEST_NAME)}
        retry_rows = [result_row(item, manifest) for item in client.messages.batches.results(batch.id)]
        if len(retry_rows) != metadata["num_requests"]:
            raise RuntimeError(f"Expected {metadata['num_requests']} retry results, received {len(retry_rows)}.")
        with labels_path.open(encoding="utf-8", newline="") as infile:
            original_rows = {row["custom_id"]: row for row in csv.DictReader(infile)}
        original_rows.update({row["custom_id"]: row for row in retry_rows})
        write_label_rows(list(original_rows.values()), labels_path)
        valid = sum(str(row.get("label_valid")).lower() == "true" for row in original_rows.values())
        print(f"retry downloaded {run_dir.name}: replaced {len(retry_rows)} rows; valid labels={valid}/{len(original_rows)}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env_file", type=Path, default=PROJECT_ROOT / ".env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "submit", "status", "recover", "download", "retry", "download-retry"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--run_dir", type=Path, action="append", required=command != "recover")
        subparser.add_argument("--overwrite", action="store_true")
        if command in {"submit", "recover", "retry"}:
            subparser.add_argument("--model", default="claude-sonnet-5")
            subparser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
            subparser.add_argument("--max_tokens", type=int, default=32)
        if command == "recover":
            subparser.add_argument("--batch_id", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "submit":
        if args.max_tokens < 32:
            raise ValueError("max_tokens must be at least 32.")
        return submit(args)
    if args.command == "retry":
        if args.max_tokens < 32:
            raise ValueError("max_tokens must be at least 32.")
        return retry(args)
    if args.command == "status":
        return status(args)
    if args.command == "recover":
        return recover(args)
    if args.command == "download-retry":
        return download_retry(args)
    return download(args)


if __name__ == "__main__":
    raise SystemExit(main())
