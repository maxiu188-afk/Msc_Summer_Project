#!/usr/bin/env python3
"""Extend the bounded long-summary clustering diagnostic from 48 to 96 questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.claude_semantic_clustering import (
    CLAUDE_STRUCTURED_EQUIVALENCE_SYSTEM_PROMPT,
    build_structured_equivalence_prompt,
    extract_structured_pair_bitstring,
)
from archehr_sebaseline.data_io import read_jsonl, write_jsonl

from run_summary_claude_clustering import (
    CLUSTERS_NAME,
    MERGED_RESULTS_NAME,
    PAIRWISE_NAME,
    deterministic_balanced_cohort,
    load_binary_labels,
    load_client,
    sha256_file,
    source_records,
    structured_output_format,
    utc_now,
    write_result_rows_and_derived,
)


COHORT_NAME = "cohort_manifest.jsonl"
ADDED_COHORT_NAME = "added_cohort_manifest.jsonl"
REQUEST_NAME = "request_manifest.jsonl"
PROTOCOL_NAME = "protocol.json"
METADATA_NAME = "batch_metadata.json"
EXTENSION_RESULTS_NAME = "extension_results.jsonl"
MAX_TOKENS = 2048


def build_requests(
    records: list[dict[str, Any]],
    *,
    added_ids: set[str],
    model: str,
    effort: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in records:
        example_id = str(row["example_id"])
        if example_id not in added_ids:
            continue
        prompt = build_structured_equivalence_prompt(
            question=str(row["question"]), answers=list(row["answers"])
        )
        custom_id = f"summary-cluster-extension-{example_id}"
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "system": CLAUDE_STRUCTURED_EQUIVALENCE_SYSTEM_PROMPT,
                    "output_config": {
                        "effort": effort,
                        "format": structured_output_format(),
                    },
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "example_id": example_id,
                "incorrect": int(row["incorrect"]),
                "sample_ids": list(row["sample_ids"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "expected_pairs": 45,
            }
        )
    return requests, manifest


def prepare(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to use non-empty output directory: {args.output_dir}")
    collection_ids = {
        str(row["id"])
        for row in read_jsonl(args.collection_dir / "current" / "examples.jsonl")
    }
    labels = {
        example_id: label
        for example_id, label in load_binary_labels(args.labels).items()
        if example_id in collection_ids
    }
    if set(labels) != collection_ids:
        raise ValueError("Every collection question must have one valid binary label.")
    cohort = deterministic_balanced_cohort(
        labels, per_class=args.per_class, seed=args.cohort_seed
    )
    base_cohort = read_jsonl(args.base_diagnostic_dir / COHORT_NAME)
    base_ids = {str(row["example_id"]) for row in base_cohort}
    target_ids = {str(row["example_id"]) for row in cohort}
    if not base_ids.issubset(target_ids):
        raise ValueError("Expanded cohort is not a strict superset of the base cohort.")
    added_ids = target_ids.difference(base_ids)
    if len(cohort) != 96 or len(base_ids) != 48 or len(added_ids) != 48:
        raise ValueError(
            f"Expected 96 target / 48 base / 48 added, found "
            f"{len(cohort)} / {len(base_ids)} / {len(added_ids)}."
        )
    records = source_records(args.collection_dir, cohort)
    requests, manifest = build_requests(
        records, added_ids=added_ids, model=args.model, effort=args.effort
    )
    if len(requests) != 48:
        raise ValueError(f"Expected 48 added requests, found {len(requests)}.")
    added_cohort = [
        row for row in cohort if str(row["example_id"]) in added_ids
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cohort, args.output_dir / COHORT_NAME, overwrite=args.overwrite)
    write_jsonl(
        added_cohort, args.output_dir / ADDED_COHORT_NAME, overwrite=args.overwrite
    )
    write_jsonl(manifest, args.output_dir / REQUEST_NAME, overwrite=args.overwrite)
    base_results = read_jsonl(args.base_diagnostic_dir / MERGED_RESULTS_NAME)
    base_valid = sum(bool(row["valid"]) for row in base_results)
    base_invalid_ids = [
        str(row["example_id"]) for row in base_results if not bool(row["valid"])
    ]
    protocol = {
        "schema_version": "bioasq_summary_claude_clustering_current96_v1",
        "status": "prepared",
        "purpose": (
            "Test whether semantic-equivalence clustering difficulty contributes "
            "to weak long-answer Semantic Entropy; not a deployment proposal."
        ),
        "scope": "96 label-balanced current-condition long-summary questions",
        "sampling": {
            "method": "nested SHA-256-ranked label-stratified extension",
            "cohort_seed": args.cohort_seed,
            "per_class": args.per_class,
            "base_questions": len(base_ids),
            "added_questions": len(added_ids),
            "target_questions": len(cohort),
        },
        "claude": {
            "model": args.model,
            "effort": args.effort,
            "max_tokens": MAX_TOKENS,
            "structured_output": True,
            "new_requests_only": len(requests),
        },
        "base_results": {
            "valid": base_valid,
            "invalid": len(base_results) - base_valid,
            "invalid_example_ids": base_invalid_ids,
            "retry_invalid_base_rows": False,
        },
        "no_retry_boundary": (
            "Validate once and retain complete cases; do not retry invalid extension rows."
        ),
        "interpretation_boundary": (
            "Use Claude only as an offline mechanism diagnostic. Do not present "
            "Claude clustering as a practical SE improvement because its cost "
            "precludes the intended deployment setting."
        ),
        "input_sha256": {
            "examples": sha256_file(
                args.collection_dir / "current" / "examples.jsonl"
            ),
            "cleaned_generations": sha256_file(
                args.collection_dir / "current" / "cleaned_generations.jsonl"
            ),
            "labels": sha256_file(args.labels),
            "base_cohort": sha256_file(args.base_diagnostic_dir / COHORT_NAME),
            "base_merged_results": sha256_file(
                args.base_diagnostic_dir / MERGED_RESULTS_NAME
            ),
            "cohort_manifest": sha256_file(args.output_dir / COHORT_NAME),
            "added_cohort_manifest": sha256_file(
                args.output_dir / ADDED_COHORT_NAME
            ),
            "request_manifest": sha256_file(args.output_dir / REQUEST_NAME),
        },
    }
    (args.output_dir / PROTOCOL_NAME).write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"prepared nested extension: 48 base + 48 added = 96 target; "
        f"base valid={base_valid}"
    )
    return 0


def recreate(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = read_jsonl(args.output_dir / COHORT_NAME)
    added = read_jsonl(args.output_dir / ADDED_COHORT_NAME)
    protocol = json.loads((args.output_dir / PROTOCOL_NAME).read_text(encoding="utf-8"))
    records = source_records(args.collection_dir, cohort)
    requests, manifest = build_requests(
        records,
        added_ids={str(row["example_id"]) for row in added},
        model=str(protocol["claude"]["model"]),
        effort=str(protocol["claude"]["effort"]),
    )
    if manifest != read_jsonl(args.output_dir / REQUEST_NAME):
        raise ValueError("Recreated extension requests differ from the frozen manifest.")
    return records, requests, manifest


def submit(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    _, requests, _ = recreate(args)
    metadata_path = args.output_dir / METADATA_NAME
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Batch metadata already exists: {metadata_path}")
    batch = client.messages.batches.create(requests=requests)
    metadata = {
        "batch_id": batch.id,
        "submitted_at_utc": utc_now(),
        "num_requests": len(requests),
        "max_tokens": requests[0]["params"]["max_tokens"],
        "model": requests[0]["params"]["model"],
        "effort": requests[0]["params"]["output_config"]["effort"],
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"submitted extension {batch.id}: {len(requests)} new requests")
    return 0


def status(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    metadata = json.loads((args.output_dir / METADATA_NAME).read_text(encoding="utf-8"))
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    counts = batch.request_counts
    print(
        f"{batch.id}: {batch.processing_status}; processing={counts.processing}, "
        f"succeeded={counts.succeeded}, errored={counts.errored}, "
        f"expired={counts.expired}, canceled={counts.canceled}"
    )
    return 0


def download(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    records, _, manifest_rows = recreate(args)
    metadata = json.loads((args.output_dir / METADATA_NAME).read_text(encoding="utf-8"))
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    if batch.processing_status != "ended":
        raise RuntimeError(f"Extension batch {batch.id} is {batch.processing_status}.")
    items = list(client.messages.batches.results(batch.id))
    if len(items) != int(metadata["num_requests"]):
        raise RuntimeError(
            f"Expected {metadata['num_requests']} results, received {len(items)}."
        )
    manifest = {str(row["custom_id"]): row for row in manifest_rows}
    extension_rows: list[dict[str, Any]] = []
    for item in items:
        base = manifest.get(str(item.custom_id))
        if base is None:
            raise KeyError(f"Unrecognized extension custom_id: {item.custom_id}")
        result_type = str(item.result.type)
        row: dict[str, Any] = {
            **base,
            "result_type": result_type,
            "bitstring": "",
            "valid": False,
            "stop_reason": "",
            "error": "",
            "result_source": "current96_extension_batch",
        }
        if result_type == "succeeded":
            row["stop_reason"] = str(item.result.message.stop_reason)
            try:
                row["bitstring"] = extract_structured_pair_bitstring(
                    item.result.message
                )
                row["valid"] = True
            except ValueError as exc:
                row["error"] = str(exc)
        else:
            row["error"] = str(getattr(item.result, "error", result_type))
        extension_rows.append(row)
    extension_rows.sort(key=lambda row: str(row["example_id"]))
    extension_path = args.output_dir / EXTENSION_RESULTS_NAME
    if extension_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {extension_path}.")
    write_jsonl(extension_rows, extension_path, overwrite=args.overwrite)

    base_results = read_jsonl(args.base_diagnostic_dir / MERGED_RESULTS_NAME)
    merged_rows = base_results + extension_rows
    if len({str(row["example_id"]) for row in merged_rows}) != 96:
        raise ValueError("Merged extension results do not cover 96 unique target IDs.")
    valid = write_result_rows_and_derived(
        records=records,
        result_rows=merged_rows,
        output_dir=args.output_dir,
        results_name=MERGED_RESULTS_NAME,
        overwrite=args.overwrite,
    )
    invalid_ids = [
        str(row["example_id"]) for row in merged_rows if not bool(row["valid"])
    ]
    print(
        f"extension complete cases: {valid}/96; no retries; "
        f"invalid={','.join(invalid_ids) if invalid_ids else 'none'}"
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--base-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--per-class", type=int, default=48)
    prepare_parser.add_argument("--cohort-seed", type=int, default=20260726)
    prepare_parser.add_argument("--model", default="claude-sonnet-5")
    prepare_parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    prepare_parser.add_argument("--overwrite", action="store_true")
    for command in ("submit", "status", "download"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return prepare(args)
    if args.command == "submit":
        return submit(args)
    if args.command == "status":
        return status(args)
    return download(args)


if __name__ == "__main__":
    raise SystemExit(main())
