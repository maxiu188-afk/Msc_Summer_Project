"""Compute question-level UQ metrics from Claude labels on best generations."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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


AURAC_FIELDS = [
    "score_name",
    "num_examples",
    "num_positive",
    "minimum_coverage",
    "maximum_coverage",
    "aurac",
]

UQ_SCORE_NAMES = [
    "normalized_discrete_semantic_entropy",
    "discrete_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "num_clusters",
    "mean_token_entropy",
    "mean_normalized_nll",
    "mean_sequence_nll",
    "avg_token_logprob_uncertainty",
    "verbalized_confidence_uncertainty",
    "p_true_blind_uncertainty",
    "p_true_with_samples_uncertainty",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument(
        "--allow_incomplete_labels",
        action="store_true",
        help="Exclude unanswered Claude cases instead of failing the evaluation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_aurac_rows(
    rejection_rows: list[dict[str, str]],
    *,
    num_examples: int,
    num_positive: int,
) -> list[dict[str, str]]:
    """Integrate risk (1 - retained accuracy) over the reported coverage range."""

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rejection_rows:
        grouped[row["score_name"]].append(row)
    results: list[dict[str, str]] = []
    for score_name, curve in grouped.items():
        points = sorted(
            (float(row["coverage"]), 1.0 - float(row["accuracy"]))
            for row in curve
            if row.get("coverage") and row.get("accuracy")
        )
        area = sum(
            (points[index][0] - points[index - 1][0])
            * (points[index][1] + points[index - 1][1])
            / 2.0
            for index in range(1, len(points))
        )
        results.append(
            {
                "score_name": score_name,
                "num_examples": str(num_examples),
                "num_positive": str(num_positive),
                "minimum_coverage": f"{points[0][0]:.10f}" if points else "",
                "maximum_coverage": f"{points[-1][0]:.10f}" if points else "",
                "aurac": f"{area:.10f}" if points else "",
            }
        )
    return results


def load_uq_rows(run_dir: Path) -> list[dict[str, str]]:
    """Merge the native Level 4 and post-hoc UQ artifacts by example ID."""

    source_paths = [
        run_dir / "se_scores.csv",
        run_dir / "example_uq.csv",
        run_dir / "uq_baselines" / "self_report_examples.csv",
    ]
    merged: dict[str, dict[str, str]] = {}
    for path in source_paths:
        if not path.exists():
            if path.name == "self_report_examples.csv":
                continue
            raise FileNotFoundError(f"Required Level 4 UQ artifact is missing: {path}")
        for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
            example_id = str(row.get("example_id") or "")
            if not example_id:
                raise ValueError(f"UQ artifact has an empty example_id: {path}")
            merged.setdefault(example_id, {}).update(row)
    if not merged:
        raise ValueError(f"No UQ rows found under {run_dir}.")
    return [merged[example_id] for example_id in sorted(merged)]


def main() -> int:
    args = parse_args()
    judge_dir = args.run_dir / "claude_binary_main_answer_judge"
    labels_path = judge_dir / "claude_generation_labels.csv"
    labels = list(csv.DictReader(labels_path.open(encoding="utf-8")))
    by_example: dict[str, dict[str, str]] = {}
    for row in labels:
        if str(row.get("label_valid")).lower() != "true":
            if args.allow_incomplete_labels:
                continue
            raise ValueError(f"Claude label is invalid for {row.get('example_id')}; retry it before evaluation.")
        example_id = str(row["example_id"])
        if example_id in by_example:
            raise ValueError(f"More than one Claude label for {example_id}; expected one best generation.")
        by_example[example_id] = row
    se_rows = load_uq_rows(args.run_dir)
    if not args.allow_incomplete_labels and len(by_example) != len(se_rows):
        raise ValueError(f"Claude labels ({len(by_example)}) do not match UQ rows ({len(se_rows)}).")
    rows = []
    excluded_example_ids = []
    for row in se_rows:
        label = by_example.get(str(row["example_id"]), {}).get("label")
        if label not in {"correct", "incorrect"}:
            if args.allow_incomplete_labels:
                excluded_example_ids.append(str(row["example_id"]))
                continue
            raise ValueError(f"Missing valid Claude label for {row['example_id']}.")
        rows.append(
            {
                **row,
                "claude_label": label,
                "is_low_quality": str(label == "incorrect").lower(),
            }
        )
    if not rows:
        raise ValueError("No valid Claude labels are available for evaluation.")
    auroc_rows = build_auroc_rows(rows, score_names=UQ_SCORE_NAMES)
    rejection_rows = build_rejection_curve_rows(rows, score_names=UQ_SCORE_NAMES)
    aurac_rows = build_aurac_rows(
        rejection_rows,
        num_examples=len(rows),
        num_positive=sum(row["is_low_quality"] == "true" for row in rows),
    )
    write_csv(rows, judge_dir / "claude_uq_examples.csv", list(rows[0]), overwrite=args.overwrite)
    write_csv(auroc_rows, judge_dir / "claude_uq_auroc.csv", AUROC_FIELDS, overwrite=args.overwrite)
    write_csv(rejection_rows, judge_dir / "claude_uq_rejection_curve.csv", REJECTION_CURVE_FIELDS, overwrite=args.overwrite)
    write_csv(aurac_rows, judge_dir / "claude_uq_aurac.csv", AURAC_FIELDS, overwrite=args.overwrite)
    summary = {
        "num_total_questions": len(se_rows),
        "num_evaluated_questions": len(rows),
        "num_excluded_questions": len(excluded_example_ids),
        "excluded_example_ids": excluded_example_ids,
        "positive_labels": ["incorrect"],
        "aurac_coverage_range": [0.5, 1.0],
    }
    (judge_dir / "claude_uq_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"wrote Claude-label UQ evaluation for {len(rows)}/{len(se_rows)} questions "
        f"to {judge_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
