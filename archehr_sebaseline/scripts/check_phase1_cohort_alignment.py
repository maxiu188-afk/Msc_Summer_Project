#!/usr/bin/env python3
"""Verify that a model-scale run uses the accepted Phase-1 BioASQ cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.dataset_adapters import load_common_examples


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_type_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid type limit {value!r}; expected TYPE=COUNT.")
        question_type, raw_count = value.split("=", 1)
        question_type = question_type.strip().lower()
        if not question_type or question_type in limits:
            raise ValueError(f"Duplicate or empty type in {value!r}.")
        count = int(raw_count)
        if count < 0:
            raise ValueError(f"Type limit must be non-negative: {value!r}.")
        limits[question_type] = count
    if set(limits) != {"factoid", "list", "summary"}:
        raise ValueError("Type limits must specify factoid, list, and summary exactly once.")
    return limits


def record_by_id(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in rows:
        example_id = str(row.get("id") or "")
        if not example_id:
            raise ValueError(f"{source} contains an example without id.")
        if example_id in records:
            raise ValueError(f"{source} contains duplicate id {example_id}.")
        records[example_id] = row
    return records


def compare_cohorts(
    *,
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    mode: str,
    expected_type_limits: dict[str, int],
) -> dict[str, Any]:
    reference = record_by_id(reference_rows, "reference")
    candidate = record_by_id(candidate_rows, "candidate")
    reference_ids = set(reference)
    candidate_ids = set(candidate)
    if mode == "exact" and candidate_ids != reference_ids:
        raise ValueError(
            "Exact cohort mismatch: "
            f"missing={sorted(reference_ids - candidate_ids)}, "
            f"unexpected={sorted(candidate_ids - reference_ids)}."
        )
    if mode == "subset" and not candidate_ids.issubset(reference_ids):
        raise ValueError(
            f"Candidate contains IDs outside Phase 1: {sorted(candidate_ids - reference_ids)}."
        )
    expected_count = sum(expected_type_limits.values())
    if len(candidate) != expected_count:
        raise ValueError(
            f"Expected {expected_count} candidate examples, found {len(candidate)}."
        )
    actual_type_counts = Counter(
        str(row.get("bioasq_type") or "").lower() for row in candidate.values()
    )
    if actual_type_counts != Counter(expected_type_limits):
        raise ValueError(
            f"Candidate type counts {dict(actual_type_counts)} do not match "
            f"{expected_type_limits}."
        )
    changed_records = [
        example_id
        for example_id in sorted(candidate_ids)
        if candidate[example_id] != reference[example_id]
    ]
    if changed_records:
        raise ValueError(
            f"{len(changed_records)} shared example records differ; first IDs: "
            f"{changed_records[:10]}."
        )
    canonical_id_sha256 = hashlib.sha256(
        ("\n".join(sorted(candidate_ids)) + "\n").encode("utf-8")
    ).hexdigest()
    return {
        "status": "pass",
        "mode": mode,
        "reference_examples": len(reference),
        "candidate_examples": len(candidate),
        "candidate_type_counts": dict(sorted(actual_type_counts.items())),
        "candidate_ids_sha256": canonical_id_sha256,
        "candidate_is_phase1_subset": candidate_ids.issubset(reference_ids),
        "records_identical_for_shared_ids": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-examples", type=Path, required=True)
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument("--candidate-examples", type=Path)
    candidate_group.add_argument("--source-data-path", type=Path)
    parser.add_argument("--mode", choices=("exact", "subset"), required=True)
    parser.add_argument("--type-limit", action="append", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260718)
    parser.add_argument("--split", default="train13b")
    parser.add_argument("--expected-reference-sha256")
    parser.add_argument("--write-report", type=Path)
    parser.add_argument(
        "--write-candidate-examples",
        type=Path,
        help="Write the validated candidate cohort as an exclusive JSONL manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    type_limits = parse_type_limits(args.type_limit)
    reference_sha256 = sha256_file(args.reference_examples)
    if (
        args.expected_reference_sha256
        and reference_sha256 != args.expected_reference_sha256
    ):
        raise ValueError(
            "Phase-1 reference hash mismatch: "
            f"expected {args.expected_reference_sha256}, got {reference_sha256}."
        )
    reference_rows = read_jsonl(args.reference_examples)
    if args.candidate_examples is not None:
        candidate_rows = read_jsonl(args.candidate_examples)
        candidate_source = str(args.candidate_examples)
    else:
        candidate_rows = load_common_examples(
            dataset="bioasq_medical_uq",
            data_path=args.source_data_path,
            split=args.split,
            limit=sum(type_limits.values()),
            bioasq_type_limits=type_limits,
            selection_seed=args.selection_seed,
        )
        # The no-evidence pipeline adds this provenance field after dataset
        # selection. Add the frozen value so the preflight can compare the
        # source-derived records with accepted saved examples byte-for-field.
        for row in candidate_rows:
            row["prompt_evidence_mode"] = "none"
        candidate_source = str(args.source_data_path)
    report = compare_cohorts(
        reference_rows=reference_rows,
        candidate_rows=candidate_rows,
        mode=args.mode,
        expected_type_limits=type_limits,
    )
    report.update(
        {
            "schema_version": "phase1_cohort_alignment_v1",
            "reference_path": str(args.reference_examples),
            "reference_sha256": reference_sha256,
            "candidate_source": candidate_source,
            "selection_seed": args.selection_seed,
            "split": args.split,
        }
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(rendered, encoding="utf-8")
    if args.write_candidate_examples is not None:
        write_jsonl_exclusive(args.write_candidate_examples, candidate_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
