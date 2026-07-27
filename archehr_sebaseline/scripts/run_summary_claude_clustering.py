#!/usr/bin/env python3
"""Run the bounded Claude clustering diagnostic over fixed long-summary samples."""

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

from archehr_sebaseline.claude_semantic_clustering import (
    CLAUDE_EQUIVALENCE_SYSTEM_PROMPT,
    CLAUDE_STRUCTURED_EQUIVALENCE_SYSTEM_PROMPT,
    build_equivalence_prompt,
    build_structured_equivalence_prompt,
    cluster_from_pair_decisions,
    decisions_from_bitstring,
    extract_pair_bitstring,
    extract_structured_pair_bitstring,
)
from archehr_sebaseline.data_io import read_jsonl, write_jsonl


COHORT_NAME = "cohort_manifest.jsonl"
REQUEST_NAME = "request_manifest.jsonl"
PROTOCOL_NAME = "protocol.json"
METADATA_NAME = "batch_metadata.json"
RESULTS_NAME = "claude_pair_results.jsonl"
CLUSTERS_NAME = "claude_clusters.jsonl"
PAIRWISE_NAME = "claude_pairwise_decisions.jsonl"
STRUCTURED_RETRY_MANIFEST_NAME = "structured_retry_manifest.jsonl"
STRUCTURED_RETRY_SMOKE_NAME = "structured_retry_smoke.json"
STRUCTURED_RETRY_METADATA_NAME = "structured_retry_batch_metadata.json"
STRUCTURED_RETRY_RESULTS_NAME = "structured_retry_results.jsonl"
STRUCTURED_RETRY_ROUND2_MANIFEST_NAME = "structured_retry_round2_manifest.jsonl"
STRUCTURED_RETRY_ROUND2_METADATA_NAME = "structured_retry_round2_batch_metadata.json"
STRUCTURED_RETRY_ROUND2_RESULTS_NAME = "structured_retry_round2_results.jsonl"
MERGED_RESULTS_NAME = "claude_pair_results_merged.jsonl"
STRUCTURED_RETRY_MAX_TOKENS = 512
STRUCTURED_RETRY_ROUND2_MAX_TOKENS = 2048


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_client(env_file: Path):
    load_dotenv(env_file)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise EnvironmentError(f"ANTHROPIC_API_KEY was not found in {env_file}.")
    from anthropic import Anthropic

    return Anthropic()


def load_binary_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as infile:
        for row in csv.DictReader(infile):
            if (
                int(row.get("sample_id") or 0) != 0
                or str(row.get("label_valid") or "").lower() != "true"
            ):
                continue
            label = str(row.get("label") or "").lower()
            if label not in {"correct", "incorrect"}:
                continue
            example_id = str(row["example_id"])
            if example_id in labels:
                raise ValueError(f"Duplicate label for {example_id}.")
            labels[example_id] = int(label == "incorrect")
    return labels


def deterministic_balanced_cohort(
    labels: dict[str, int], *, per_class: int, seed: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for incorrect in (0, 1):
        candidates = [example_id for example_id, value in labels.items() if value == incorrect]
        ranked = sorted(
            candidates,
            key=lambda example_id: hashlib.sha256(
                f"{seed}:{incorrect}:{example_id}".encode("utf-8")
            ).hexdigest(),
        )
        if len(ranked) < per_class:
            raise ValueError(
                f"Need {per_class} labels in class {incorrect}, found {len(ranked)}."
            )
        rows.extend(
            {"example_id": example_id, "incorrect": incorrect}
            for example_id in ranked[:per_class]
        )
    return sorted(rows, key=lambda row: str(row["example_id"]))


def source_records(
    collection_dir: Path, cohort: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    current_dir = collection_dir / "current"
    examples = {
        str(row["id"]): row for row in read_jsonl(current_dir / "examples.jsonl")
    }
    generations_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(current_dir / "cleaned_generations.jsonl"):
        generations_by_id.setdefault(str(row["example_id"]), []).append(row)
    records: list[dict[str, Any]] = []
    for cohort_row in cohort:
        example_id = str(cohort_row["example_id"])
        example = examples.get(example_id)
        generations = sorted(
            generations_by_id.get(example_id, []), key=lambda row: int(row["sample_id"])
        )
        if example is None or len(generations) != 10:
            raise ValueError(
                f"{example_id} requires one example and exactly ten generations."
            )
        sample_ids = [int(row["sample_id"]) for row in generations]
        if sample_ids != list(range(10)):
            raise ValueError(f"Unexpected sample IDs for {example_id}: {sample_ids}.")
        answers = [str(row["clean_answer"]).strip() for row in generations]
        prompt = build_equivalence_prompt(
            question=str(example["question"]).strip(), answers=answers
        )
        records.append(
            {
                **cohort_row,
                "question": str(example["question"]).strip(),
                "sample_ids": sample_ids,
                "answers": answers,
                "prompt": prompt,
            }
        )
    return records


def request_and_manifest(
    records: list[dict[str, Any]], *, model: str, effort: str, max_tokens: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in records:
        example_id = str(row["example_id"])
        prompt = str(row["prompt"])
        custom_id = f"summary-cluster-{example_id}"
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": CLAUDE_EQUIVALENCE_SYSTEM_PROMPT,
                    "output_config": {"effort": effort},
                    "messages": [{"role": "user", "content": prompt}],
                },
            }
        )
        manifest.append(
            {
                "custom_id": custom_id,
                "example_id": example_id,
                "incorrect": int(row["incorrect"]),
                "sample_ids": row["sample_ids"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "answer_sha256": [
                    hashlib.sha256(answer.encode("utf-8")).hexdigest()
                    for answer in row["answers"]
                ],
                "expected_pairs": 45,
            }
        )
    return requests, manifest


def structured_output_format() -> dict[str, Any]:
    """Return the JSON schema used only to serialize frozen pair decisions."""

    return {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "decisions": {
                    "type": "array",
                    "description": (
                        "Exactly 45 integer decisions in the pair order listed in "
                        "the prompt; each value is 0 or 1."
                    ),
                    "items": {"type": "integer", "enum": [0, 1]},
                }
            },
            "required": ["decisions"],
            "additionalProperties": False,
        },
    }


def structured_retry_requests(
    *,
    records: list[dict[str, Any]],
    initial_results: list[dict[str, Any]],
    model: str,
    effort: str,
    max_tokens: int = STRUCTURED_RETRY_MAX_TOKENS,
    custom_id_prefix: str = "summary-cluster-structured",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recreate requests only for invalid first-pass results."""

    invalid_by_id = {
        str(row["example_id"]): row
        for row in initial_results
        if not bool(row["valid"])
    }
    if len(invalid_by_id) != sum(not bool(row["valid"]) for row in initial_results):
        raise ValueError("Initial results contain duplicate invalid example IDs.")
    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for row in records:
        example_id = str(row["example_id"])
        initial = invalid_by_id.get(example_id)
        if initial is None:
            continue
        prompt = build_structured_equivalence_prompt(
            question=str(row["question"]), answers=list(row["answers"])
        )
        custom_id = f"{custom_id_prefix}-{example_id}"
        requests.append(
            {
                "custom_id": custom_id,
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
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
                "source_custom_id": str(initial["custom_id"]),
                "example_id": example_id,
                "incorrect": int(row["incorrect"]),
                "sample_ids": list(row["sample_ids"]),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "expected_pairs": 45,
                "retry_reason": str(initial["error"]),
                "initial_stop_reason": str(initial["stop_reason"]),
            }
        )
    if len(requests) != len(invalid_by_id):
        raise ValueError(
            f"Expected {len(invalid_by_id)} invalid records, recreated {len(requests)}."
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
        missing = sorted(collection_ids.difference(labels))
        raise ValueError(
            f"Current collection requires a valid label for every ID; missing {missing}."
        )
    cohort = deterministic_balanced_cohort(
        labels, per_class=args.per_class, seed=args.cohort_seed
    )
    records = source_records(args.collection_dir, cohort)
    _, manifest = request_and_manifest(
        records, model=args.model, effort=args.effort, max_tokens=args.max_tokens
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(cohort, args.output_dir / COHORT_NAME, overwrite=args.overwrite)
    write_jsonl(manifest, args.output_dir / REQUEST_NAME, overwrite=args.overwrite)
    protocol = {
        "schema_version": "bioasq_summary_claude_clustering_current48_v1",
        "status": "prepared",
        "scope": "current long-summary condition only; fixed saved generations",
        "questions": len(cohort),
        "answers": len(cohort) * 10,
        "pair_decisions": len(cohort) * 45,
        "sampling": {
            "method": "SHA-256-ranked label-stratified sample",
            "per_class": args.per_class,
            "cohort_seed": args.cohort_seed,
            "positive_class": "Claude correctness judge incorrect",
        },
        "claude": {
            "model": args.model,
            "effort": args.effort,
            "max_tokens": args.max_tokens,
            "one_request_per_question": True,
            "independent_pair_decisions_per_request": 45,
        },
        "analysis_boundary": (
            "Detect a large NLI-bottleneck signal only; do not claim subtle effects "
            "or expand beyond 48 questions automatically."
        ),
        "manual_review": (
            "A blinded 24-pair worksheet is created after clustering; final mechanism "
            "interpretation remains conditional on human review."
        ),
        "input_sha256": {
            "examples": sha256_file(
                args.collection_dir / "current" / "examples.jsonl"
            ),
            "cleaned_generations": sha256_file(
                args.collection_dir / "current" / "cleaned_generations.jsonl"
            ),
            "nli_clusters": sha256_file(
                args.collection_dir / "current" / "clusters.jsonl"
            ),
            "condition_scores": sha256_file(
                args.collection_dir / "current" / "condition_scores.csv"
            ),
            "labels": sha256_file(args.labels),
            "cohort_manifest": sha256_file(args.output_dir / COHORT_NAME),
            "request_manifest": sha256_file(args.output_dir / REQUEST_NAME),
        },
    }
    (args.output_dir / PROTOCOL_NAME).write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"prepared {len(cohort)} questions / {len(cohort) * 10} fixed answers "
        f"at {args.output_dir}"
    )
    return 0


def recreate_requests(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cohort = read_jsonl(args.output_dir / COHORT_NAME)
    protocol = json.loads((args.output_dir / PROTOCOL_NAME).read_text(encoding="utf-8"))
    records = source_records(args.collection_dir, cohort)
    requests, manifest = request_and_manifest(
        records,
        model=str(protocol["claude"]["model"]),
        effort=str(protocol["claude"]["effort"]),
        max_tokens=int(protocol["claude"]["max_tokens"]),
    )
    if manifest != read_jsonl(args.output_dir / REQUEST_NAME):
        raise ValueError("Recreated requests differ from the frozen request manifest.")
    return records, requests, manifest


def recreate_structured_retry(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records, _, _ = recreate_requests(args)
    protocol = json.loads((args.output_dir / PROTOCOL_NAME).read_text(encoding="utf-8"))
    initial_results = read_jsonl(args.output_dir / RESULTS_NAME)
    requests, manifest = structured_retry_requests(
        records=records,
        initial_results=initial_results,
        model=str(protocol["claude"]["model"]),
        effort=str(protocol["claude"]["effort"]),
    )
    manifest_path = args.output_dir / STRUCTURED_RETRY_MANIFEST_NAME
    if manifest_path.exists() and manifest != read_jsonl(manifest_path):
        raise ValueError("Recreated structured retry differs from its frozen manifest.")
    return records, requests, manifest


def retry_smoke(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    _, requests, manifest = recreate_structured_retry(args)
    if not requests:
        raise ValueError("No invalid first-pass results require a retry.")
    smoke_path = args.output_dir / STRUCTURED_RETRY_SMOKE_NAME
    if smoke_path.exists() and not args.overwrite:
        raise FileExistsError(f"Structured retry smoke already exists: {smoke_path}")
    request = requests[0]
    message = client.messages.create(**request["params"])
    row: dict[str, Any] = {
        "custom_id": request["custom_id"],
        "example_id": manifest[0]["example_id"],
        "tested_at_utc": utc_now(),
        "model": str(message.model),
        "stop_reason": str(message.stop_reason),
        "max_tokens": int(request["params"]["max_tokens"]),
        "valid": False,
        "bitstring": "",
        "error": "",
    }
    try:
        row["bitstring"] = extract_structured_pair_bitstring(message)
        row["valid"] = True
    except ValueError as exc:
        row["error"] = str(exc)
    smoke_path.write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"structured retry smoke: valid={row['valid']} "
        f"stop_reason={row['stop_reason']} example_id={row['example_id']}"
    )
    return 0 if row["valid"] else 2


def submit_structured_retry(args: argparse.Namespace) -> int:
    smoke_path = args.output_dir / STRUCTURED_RETRY_SMOKE_NAME
    if not smoke_path.exists():
        raise FileNotFoundError("Run retry-smoke before submitting the retry batch.")
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if not bool(smoke["valid"]):
        raise RuntimeError("The structured retry smoke is not valid.")
    client = load_client(args.env_file)
    _, requests, manifest = recreate_structured_retry(args)
    metadata_path = args.output_dir / STRUCTURED_RETRY_METADATA_NAME
    manifest_path = args.output_dir / STRUCTURED_RETRY_MANIFEST_NAME
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Retry batch metadata already exists: {metadata_path}")
    if manifest_path.exists() and not args.overwrite:
        frozen = read_jsonl(manifest_path)
        if frozen != manifest:
            raise ValueError("Retry manifest differs from the existing frozen manifest.")
    else:
        write_jsonl(manifest, manifest_path, overwrite=args.overwrite)
    batch = client.messages.batches.create(requests=requests)
    metadata = {
        "batch_id": batch.id,
        "submitted_at_utc": utc_now(),
        "model": requests[0]["params"]["model"],
        "effort": requests[0]["params"]["output_config"]["effort"],
        "max_tokens": requests[0]["params"]["max_tokens"],
        "num_requests": len(requests),
        "source_results": RESULTS_NAME,
        "source_results_sha256": sha256_file(args.output_dir / RESULTS_NAME),
        "serialization_only_retry": True,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"submitted structured retry {batch.id}: {len(requests)} requests")
    return 0


def status_structured_retry(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    metadata = json.loads(
        (args.output_dir / STRUCTURED_RETRY_METADATA_NAME).read_text(encoding="utf-8")
    )
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    counts = batch.request_counts
    print(
        f"{batch.id}: {batch.processing_status}; processing={counts.processing}, "
        f"succeeded={counts.succeeded}, errored={counts.errored}, "
        f"expired={counts.expired}, canceled={counts.canceled}"
    )
    return 0


def recreate_structured_retry_round2(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    records, _, _ = recreate_requests(args)
    protocol = json.loads((args.output_dir / PROTOCOL_NAME).read_text(encoding="utf-8"))
    first_retry_results = read_jsonl(args.output_dir / STRUCTURED_RETRY_RESULTS_NAME)
    requests, manifest = structured_retry_requests(
        records=records,
        initial_results=first_retry_results,
        model=str(protocol["claude"]["model"]),
        effort=str(protocol["claude"]["effort"]),
        max_tokens=STRUCTURED_RETRY_ROUND2_MAX_TOKENS,
        custom_id_prefix="summary-cluster-structured-r2",
    )
    manifest_path = args.output_dir / STRUCTURED_RETRY_ROUND2_MANIFEST_NAME
    if manifest_path.exists() and manifest != read_jsonl(manifest_path):
        raise ValueError("Recreated round-2 retry differs from its frozen manifest.")
    return records, requests, manifest


def submit_structured_retry_round2(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    _, requests, manifest = recreate_structured_retry_round2(args)
    if not requests:
        raise ValueError("No invalid structured retry results require round 2.")
    metadata_path = args.output_dir / STRUCTURED_RETRY_ROUND2_METADATA_NAME
    manifest_path = args.output_dir / STRUCTURED_RETRY_ROUND2_MANIFEST_NAME
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Round-2 metadata already exists: {metadata_path}")
    if manifest_path.exists() and not args.overwrite:
        frozen = read_jsonl(manifest_path)
        if frozen != manifest:
            raise ValueError("Round-2 manifest differs from the existing manifest.")
    else:
        write_jsonl(manifest, manifest_path, overwrite=args.overwrite)
    batch = client.messages.batches.create(requests=requests)
    metadata = {
        "batch_id": batch.id,
        "submitted_at_utc": utc_now(),
        "model": requests[0]["params"]["model"],
        "effort": requests[0]["params"]["output_config"]["effort"],
        "max_tokens": requests[0]["params"]["max_tokens"],
        "num_requests": len(requests),
        "source_results": STRUCTURED_RETRY_RESULTS_NAME,
        "source_results_sha256": sha256_file(
            args.output_dir / STRUCTURED_RETRY_RESULTS_NAME
        ),
        "serialization_only_retry": True,
        "retry_round": 2,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"submitted structured retry round 2 {batch.id}: {len(requests)} requests")
    return 0


def status_structured_retry_round2(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    metadata = json.loads(
        (args.output_dir / STRUCTURED_RETRY_ROUND2_METADATA_NAME).read_text(
            encoding="utf-8"
        )
    )
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    counts = batch.request_counts
    print(
        f"{batch.id}: {batch.processing_status}; processing={counts.processing}, "
        f"succeeded={counts.succeeded}, errored={counts.errored}, "
        f"expired={counts.expired}, canceled={counts.canceled}"
    )
    return 0


def submit(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    _, requests, _ = recreate_requests(args)
    metadata_path = args.output_dir / METADATA_NAME
    if metadata_path.exists() and not args.overwrite:
        raise FileExistsError(f"Batch metadata already exists: {metadata_path}")
    batch = client.messages.batches.create(requests=requests)
    protocol = json.loads((args.output_dir / PROTOCOL_NAME).read_text(encoding="utf-8"))
    metadata = {
        "batch_id": batch.id,
        "submitted_at_utc": utc_now(),
        "model": protocol["claude"]["model"],
        "effort": protocol["claude"]["effort"],
        "max_tokens": protocol["claude"]["max_tokens"],
        "num_requests": len(requests),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"submitted {batch.id}: {len(requests)} requests")
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


def write_result_rows_and_derived(
    *,
    records: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    output_dir: Path,
    results_name: str,
    overwrite: bool,
) -> int:
    record_by_id = {str(row["example_id"]): row for row in records}
    clusters: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for row in result_rows:
        if not row["valid"]:
            continue
        source = record_by_id[str(row["example_id"])]
        cluster = cluster_from_pair_decisions(
            example_id=str(row["example_id"]),
            answers=list(source["answers"]),
            sample_ids=[int(value) for value in source["sample_ids"]],
            bitstring=str(row["bitstring"]),
        )
        clusters.append(cluster)
        pairwise_rows.append(
            {
                "example_id": str(row["example_id"]),
                "incorrect": int(source["incorrect"]),
                "bitstring": str(row["bitstring"]),
                "decisions": decisions_from_bitstring(
                    str(row["bitstring"]),
                    sample_ids=[int(value) for value in source["sample_ids"]],
                ),
            }
        )
    result_rows.sort(key=lambda row: str(row["example_id"]))
    clusters.sort(key=lambda row: str(row["example_id"]))
    pairwise_rows.sort(key=lambda row: str(row["example_id"]))
    for path in (
        output_dir / results_name,
        output_dir / CLUSTERS_NAME,
        output_dir / PAIRWISE_NAME,
    ):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}.")
    write_jsonl(result_rows, output_dir / results_name, overwrite=overwrite)
    write_jsonl(clusters, output_dir / CLUSTERS_NAME, overwrite=overwrite)
    write_jsonl(pairwise_rows, output_dir / PAIRWISE_NAME, overwrite=overwrite)
    valid = sum(bool(row["valid"]) for row in result_rows)
    print(f"materialized valid Claude clustering results: {valid}/{len(result_rows)}")
    return valid


def materialize(
    *,
    records: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    items: list[Any],
    output_dir: Path,
    overwrite: bool,
) -> int:
    manifest = {str(row["custom_id"]): row for row in manifest_rows}
    result_rows: list[dict[str, Any]] = []
    for item in items:
        base = manifest.get(str(item.custom_id))
        if base is None:
            raise KeyError(f"Unrecognized result custom_id: {item.custom_id}")
        result_type = str(item.result.type)
        row: dict[str, Any] = {
            **base,
            "result_type": result_type,
            "bitstring": "",
            "valid": False,
            "stop_reason": "",
            "error": "",
        }
        if result_type == "succeeded":
            row["stop_reason"] = str(item.result.message.stop_reason)
            try:
                row["bitstring"] = extract_pair_bitstring(item.result.message)
                row["valid"] = True
            except ValueError as exc:
                row["error"] = str(exc)
        else:
            row["error"] = str(getattr(item.result, "error", result_type))
        result_rows.append(row)
    return write_result_rows_and_derived(
        records=records,
        result_rows=result_rows,
        output_dir=output_dir,
        results_name=RESULTS_NAME,
        overwrite=overwrite,
    )


def download(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    records, _, manifest = recreate_requests(args)
    metadata = json.loads((args.output_dir / METADATA_NAME).read_text(encoding="utf-8"))
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    if batch.processing_status != "ended":
        raise RuntimeError(f"Batch {batch.id} is {batch.processing_status}.")
    items = list(client.messages.batches.results(batch.id))
    if len(items) != int(metadata["num_requests"]):
        raise RuntimeError(
            f"Expected {metadata['num_requests']} results, received {len(items)}."
        )
    valid = materialize(
        records=records,
        manifest_rows=manifest,
        items=items,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    return 0 if valid == len(items) else 2


def download_structured_retry(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    records, _, retry_manifest_rows = recreate_structured_retry(args)
    metadata = json.loads(
        (args.output_dir / STRUCTURED_RETRY_METADATA_NAME).read_text(encoding="utf-8")
    )
    if sha256_file(args.output_dir / RESULTS_NAME) != str(
        metadata["source_results_sha256"]
    ):
        raise ValueError("Initial result file changed after the retry batch was submitted.")
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    if batch.processing_status != "ended":
        raise RuntimeError(f"Retry batch {batch.id} is {batch.processing_status}.")
    items = list(client.messages.batches.results(batch.id))
    if len(items) != int(metadata["num_requests"]):
        raise RuntimeError(
            f"Expected {metadata['num_requests']} retry results, received {len(items)}."
        )
    manifest = {str(row["custom_id"]): row for row in retry_manifest_rows}
    retry_rows: list[dict[str, Any]] = []
    for item in items:
        base = manifest.get(str(item.custom_id))
        if base is None:
            raise KeyError(f"Unrecognized retry custom_id: {item.custom_id}")
        result_type = str(item.result.type)
        row: dict[str, Any] = {
            **base,
            "result_type": result_type,
            "bitstring": "",
            "valid": False,
            "stop_reason": "",
            "error": "",
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
        retry_rows.append(row)
    retry_rows.sort(key=lambda row: str(row["example_id"]))
    retry_results_path = args.output_dir / STRUCTURED_RETRY_RESULTS_NAME
    if retry_results_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {retry_results_path}.")
    write_jsonl(retry_rows, retry_results_path, overwrite=args.overwrite)

    retry_by_id = {str(row["example_id"]): row for row in retry_rows}
    merged_rows: list[dict[str, Any]] = []
    for initial in read_jsonl(args.output_dir / RESULTS_NAME):
        example_id = str(initial["example_id"])
        merged = dict(initial)
        if bool(initial["valid"]):
            merged["result_source"] = "initial_bitstring_batch"
        else:
            retry = retry_by_id.get(example_id)
            if retry is None:
                raise KeyError(f"Missing structured retry result for {example_id}.")
            merged.update(
                {
                    "bitstring": str(retry["bitstring"]),
                    "valid": bool(retry["valid"]),
                    "stop_reason": str(retry["stop_reason"]),
                    "error": str(retry["error"]),
                    "result_source": "structured_serialization_retry",
                    "retry_custom_id": str(retry["custom_id"]),
                    "retry_result_type": str(retry["result_type"]),
                }
            )
        merged_rows.append(merged)
    valid = sum(bool(row["valid"]) for row in merged_rows)
    if valid != len(merged_rows):
        invalid_ids = [
            str(row["example_id"]) for row in merged_rows if not bool(row["valid"])
        ]
        print(
            f"structured retry still invalid: {len(invalid_ids)}/{len(merged_rows)} "
            f"({', '.join(invalid_ids)})"
        )
        return 2
    materialized = write_result_rows_and_derived(
        records=records,
        result_rows=merged_rows,
        output_dir=args.output_dir,
        results_name=MERGED_RESULTS_NAME,
        overwrite=args.overwrite,
    )
    return 0 if materialized == len(merged_rows) else 2


def download_structured_retry_round2(args: argparse.Namespace) -> int:
    client = load_client(args.env_file)
    records, _, retry_manifest_rows = recreate_structured_retry_round2(args)
    metadata = json.loads(
        (args.output_dir / STRUCTURED_RETRY_ROUND2_METADATA_NAME).read_text(
            encoding="utf-8"
        )
    )
    if sha256_file(args.output_dir / STRUCTURED_RETRY_RESULTS_NAME) != str(
        metadata["source_results_sha256"]
    ):
        raise ValueError("First retry results changed after round 2 was submitted.")
    batch = client.messages.batches.retrieve(metadata["batch_id"])
    if batch.processing_status != "ended":
        raise RuntimeError(f"Round-2 retry batch {batch.id} is {batch.processing_status}.")
    items = list(client.messages.batches.results(batch.id))
    if len(items) != int(metadata["num_requests"]):
        raise RuntimeError(
            f"Expected {metadata['num_requests']} round-2 results, received {len(items)}."
        )
    manifest = {str(row["custom_id"]): row for row in retry_manifest_rows}
    round2_rows: list[dict[str, Any]] = []
    for item in items:
        base = manifest.get(str(item.custom_id))
        if base is None:
            raise KeyError(f"Unrecognized round-2 custom_id: {item.custom_id}")
        result_type = str(item.result.type)
        row: dict[str, Any] = {
            **base,
            "result_type": result_type,
            "bitstring": "",
            "valid": False,
            "stop_reason": "",
            "error": "",
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
        round2_rows.append(row)
    round2_rows.sort(key=lambda row: str(row["example_id"]))
    round2_path = args.output_dir / STRUCTURED_RETRY_ROUND2_RESULTS_NAME
    if round2_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {round2_path}.")
    write_jsonl(round2_rows, round2_path, overwrite=args.overwrite)

    first_retry_by_id = {
        str(row["example_id"]): row
        for row in read_jsonl(args.output_dir / STRUCTURED_RETRY_RESULTS_NAME)
    }
    round2_by_id = {str(row["example_id"]): row for row in round2_rows}
    merged_rows: list[dict[str, Any]] = []
    for initial in read_jsonl(args.output_dir / RESULTS_NAME):
        example_id = str(initial["example_id"])
        merged = dict(initial)
        if bool(initial["valid"]):
            merged["result_source"] = "initial_bitstring_batch"
        else:
            first_retry = first_retry_by_id.get(example_id)
            if first_retry is None:
                raise KeyError(f"Missing first structured retry result for {example_id}.")
            chosen = first_retry
            source = "structured_serialization_retry"
            if not bool(first_retry["valid"]):
                chosen = round2_by_id.get(example_id)
                source = "structured_serialization_retry_round2"
                if chosen is None:
                    raise KeyError(f"Missing round-2 retry result for {example_id}.")
            merged.update(
                {
                    "bitstring": str(chosen["bitstring"]),
                    "valid": bool(chosen["valid"]),
                    "stop_reason": str(chosen["stop_reason"]),
                    "error": str(chosen["error"]),
                    "result_source": source,
                    "retry_custom_id": str(chosen["custom_id"]),
                    "retry_result_type": str(chosen["result_type"]),
                }
            )
        merged_rows.append(merged)
    valid = sum(bool(row["valid"]) for row in merged_rows)
    if valid != len(merged_rows):
        invalid_ids = [
            str(row["example_id"]) for row in merged_rows if not bool(row["valid"])
        ]
        write_result_rows_and_derived(
            records=records,
            result_rows=merged_rows,
            output_dir=args.output_dir,
            results_name=MERGED_RESULTS_NAME,
            overwrite=args.overwrite,
        )
        print(
            f"round-2 retry still invalid: {len(invalid_ids)}/{len(merged_rows)} "
            f"({', '.join(invalid_ids)})"
        )
        return 2
    materialized = write_result_rows_and_derived(
        records=records,
        result_rows=merged_rows,
        output_dir=args.output_dir,
        results_name=MERGED_RESULTS_NAME,
        overwrite=args.overwrite,
    )
    return 0 if materialized == len(merged_rows) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--per-class", type=int, default=24)
    prepare_parser.add_argument("--cohort-seed", type=int, default=20260726)
    prepare_parser.add_argument("--model", default="claude-sonnet-5")
    prepare_parser.add_argument("--effort", choices=("low", "medium", "high"), default="low")
    prepare_parser.add_argument("--max-tokens", type=int, default=128)
    prepare_parser.add_argument("--overwrite", action="store_true")
    for command in (
        "submit",
        "status",
        "download",
        "retry-smoke",
        "retry-submit",
        "retry-status",
        "retry-download",
        "retry2-submit",
        "retry2-status",
        "retry2-download",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        if args.per_class <= 0:
            raise ValueError("--per-class must be positive.")
        if args.max_tokens < 64:
            raise ValueError("--max-tokens must be at least 64.")
        return prepare(args)
    if args.command == "submit":
        return submit(args)
    if args.command == "status":
        return status(args)
    if args.command == "download":
        return download(args)
    if args.command == "retry-smoke":
        return retry_smoke(args)
    if args.command == "retry-submit":
        return submit_structured_retry(args)
    if args.command == "retry-status":
        return status_structured_retry(args)
    if args.command == "retry-download":
        return download_structured_retry(args)
    if args.command == "retry2-submit":
        return submit_structured_retry_round2(args)
    if args.command == "retry2-status":
        return status_structured_retry_round2(args)
    return download_structured_retry_round2(args)


if __name__ == "__main__":
    raise SystemExit(main())
