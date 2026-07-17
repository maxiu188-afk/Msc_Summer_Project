"""Compute question-level UQ metrics from Claude labels on best generations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import write_csv
from archehr_sebaseline.evaluation.uncertainty_metrics import (
    AUROC_FIELDS,
    REJECTION_CURVE_FIELDS,
    build_auroc_rows,
    build_rejection_curve_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    judge_dir = args.run_dir / "claude_judge"
    labels_path = judge_dir / "claude_generation_labels.csv"
    se_path = args.run_dir / "bioasq_eval" / "bioasq_se_eval_examples.csv"
    labels = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    by_example: dict[str, dict[str, str]] = {}
    for row in labels:
        if str(row.get("label_valid")).lower() != "true":
            raise ValueError(f"Claude label is invalid for {row.get('example_id')}; retry it before evaluation.")
        example_id = str(row["example_id"])
        if example_id in by_example:
            raise ValueError(f"More than one Claude label for {example_id}; expected one best generation.")
        by_example[example_id] = row
    se_rows = list(csv.DictReader(se_path.open(encoding="utf-8")))
    if len(by_example) != len(se_rows):
        raise ValueError(f"Claude labels ({len(by_example)}) do not match UQ rows ({len(se_rows)}).")
    rows = []
    for row in se_rows:
        label = by_example.get(str(row["example_id"]), {}).get("label")
        if label not in {"good", "partial", "poor"}:
            raise ValueError(f"Missing valid Claude label for {row['example_id']}.")
        rows.append({**row, "claude_label": label, "is_low_quality": str(label == "poor").lower()})
    auroc_rows = build_auroc_rows(rows)
    rejection_rows = build_rejection_curve_rows(rows)
    write_csv(rows, judge_dir / "claude_uq_examples.csv", list(rows[0]), overwrite=args.overwrite)
    write_csv(auroc_rows, judge_dir / "claude_uq_auroc.csv", AUROC_FIELDS, overwrite=args.overwrite)
    write_csv(rejection_rows, judge_dir / "claude_uq_rejection_curve.csv", REJECTION_CURVE_FIELDS, overwrite=args.overwrite)
    print(f"wrote Claude-label UQ evaluation for {len(rows)} questions to {judge_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
