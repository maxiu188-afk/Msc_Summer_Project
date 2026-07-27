#!/usr/bin/env python3
"""Repair sentence-derived fields from saved summary-length answers.

This deterministic repair changes no generation, UQ score, cluster, label, or
model artifact.  It archives each original condition score table before
rewriting only text-length and sentence-compliance fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl
from archehr_sebaseline.summary_length import text_length_stats


CONDITIONS = ("current", "short")
REPAIRED_FIELDS = (
    "main_word_count",
    "main_sentence_count",
    "main_one_or_two_sentence_compliant",
    "sample_word_count_mean",
    "sample_sentence_count_mean",
    "sample_one_or_two_sentence_compliance_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _answer(record: dict[str, Any]) -> str:
    value = str(record.get("clean_answer") or record.get("raw_answer") or "").strip()
    if not value:
        raise ValueError(f"Empty saved answer for {record.get('example_id')}.")
    return value


def recompute_condition_rows(
    score_rows: list[dict[str, str]],
    main_generations: list[dict[str, Any]],
    sampled_generations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    main_by_id = {str(row["example_id"]): row for row in main_generations}
    samples_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sampled_generations:
        samples_by_id[str(row["example_id"])].append(row)
    score_ids = {str(row["example_id"]) for row in score_rows}
    if set(main_by_id) != score_ids or set(samples_by_id) != score_ids:
        raise ValueError("Score, main-answer, and sampled-answer IDs do not match.")

    repaired = []
    for row in score_rows:
        example_id = str(row["example_id"])
        samples = samples_by_id[example_id]
        if len(samples) != 10:
            raise ValueError(f"Expected ten samples for {example_id}, got {len(samples)}.")
        main_stats = text_length_stats(_answer(main_by_id[example_id]))
        sample_stats = [text_length_stats(_answer(sample)) for sample in samples]
        repaired.append(
            {
                **row,
                "main_word_count": int(main_stats["word_count"]),
                "main_sentence_count": int(main_stats["sentence_count"]),
                "main_one_or_two_sentence_compliant": int(
                    main_stats["one_or_two_sentence_compliant"]
                ),
                "sample_word_count_mean": fmean(
                    int(stats["word_count"]) for stats in sample_stats
                ),
                "sample_sentence_count_mean": fmean(
                    int(stats["sentence_count"]) for stats in sample_stats
                ),
                "sample_one_or_two_sentence_compliance_rate": fmean(
                    bool(stats["one_or_two_sentence_compliant"])
                    for stats in sample_stats
                ),
            }
        )
    return repaired


def main() -> int:
    args = parse_args()
    metadata_path = args.collection_dir / "sentence_count_repair.json"
    archive_dir = args.collection_dir / "archive_sentence_count_pre_fix"
    if metadata_path.exists() or archive_dir.exists():
        raise FileExistsError("Sentence-count repair has already been applied or archived.")
    archive_dir.mkdir()

    metadata: dict[str, Any] = {
        "schema_version": "bioasq_summary_length_sentence_count_repair_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "derived text-length fields only; no generation, UQ, cluster, or label changes",
        "repaired_fields": list(REPAIRED_FIELDS),
        "conditions": {},
    }
    for condition in CONDITIONS:
        condition_dir = args.collection_dir / condition
        score_path = condition_dir / "condition_scores.csv"
        fieldnames, score_rows = _read_csv(score_path)
        if not score_rows or any(field not in fieldnames for field in REPAIRED_FIELDS):
            raise ValueError(f"Condition score schema is incomplete: {score_path}")
        before_hash = sha256_file(score_path)
        archive_path = archive_dir / f"{condition}_condition_scores.csv"
        shutil.copy2(score_path, archive_path)
        repaired = recompute_condition_rows(
            score_rows,
            read_jsonl(condition_dir / "best_generations.jsonl"),
            read_jsonl(condition_dir / "generations.jsonl"),
        )
        _write_csv(score_path, fieldnames, repaired)
        metadata["conditions"][condition] = {
            "rows": len(repaired),
            "before_sha256": before_hash,
            "archive_sha256": sha256_file(archive_path),
            "after_sha256": sha256_file(score_path),
            "main_compliant": sum(
                int(row["main_one_or_two_sentence_compliant"]) for row in repaired
            ),
            "sample_compliance_rate_mean": fmean(
                float(row["sample_one_or_two_sentence_compliance_rate"])
                for row in repaired
            ),
        }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
