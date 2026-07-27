#!/usr/bin/env python3
"""Analyze the paired BioASQ current-versus-short summary intervention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_probe import binary_ranking_metrics


METHOD_FIELDS = {
    "blind_p_true": "blind_p_true_uncertainty",
    "discrete_semantic_entropy": "discrete_semantic_entropy",
    "num_clusters": "num_clusters",
    "normalized_nll_10_samples": "normalized_nll_10_samples",
}
LENGTH_FIELDS = (
    "main_generated_tokens",
    "main_word_count",
    "main_sentence_count",
    "main_one_or_two_sentence_compliant",
    "sample_generated_tokens_mean",
    "sample_word_count_mean",
    "sample_sentence_count_mean",
    "sample_one_or_two_sentence_compliance_rate",
    "largest_cluster_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--current-labels", type=Path, required=True)
    parser.add_argument("--short-labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return number


def load_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    for row in _read_csv(path):
        valid = str(row.get("label_valid") or "").strip().lower()
        label = str(row.get("label") or "").strip().lower()
        if valid not in {"true", "1", "yes"} or label not in {"correct", "incorrect"}:
            continue
        example_id = str(row.get("example_id") or "")
        sample_id = int(row.get("sample_id") or 0)
        if sample_id != 0:
            continue
        if example_id in labels:
            raise ValueError(f"Duplicate valid label for {example_id} in {path}.")
        labels[example_id] = int(label == "incorrect")
    return labels


def percentile_interval(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot calculate an interval from no values.")
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_ranking_metrics(labels, scores)["auroc"])


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(binary_ranking_metrics(labels, scores)["average_precision"])


def analyze(
    *,
    current_rows: list[dict[str, str]],
    short_rows: list[dict[str, str]],
    current_labels: dict[str, int],
    short_labels: dict[str, int],
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive.")
    by_condition = {
        "current": {str(row["example_id"]): row for row in current_rows},
        "short": {str(row["example_id"]): row for row in short_rows},
    }
    collection_ids = sorted(by_condition["current"])
    expected_ids = set(collection_ids)
    if set(by_condition["short"]) != expected_ids:
        raise ValueError("Current and short score tables must contain exactly the same IDs.")
    paired_ids = sorted(expected_ids.intersection(current_labels, short_labels))
    if not paired_ids:
        raise ValueError("No question has valid labels in both conditions.")

    per_example = []
    arrays: dict[str, dict[str, np.ndarray]] = {"current": {}, "short": {}}
    for condition, labels in (("current", current_labels), ("short", short_labels)):
        arrays[condition]["incorrect"] = np.asarray(
            [labels[example_id] for example_id in paired_ids], dtype=np.int64
        )
        for method, field in METHOD_FIELDS.items():
            arrays[condition][method] = np.asarray(
                [
                    _finite(by_condition[condition][example_id][field], field)
                    for example_id in paired_ids
                ],
                dtype=np.float64,
            )
    length_arrays: dict[str, dict[str, np.ndarray]] = {"current": {}, "short": {}}
    for condition in ("current", "short"):
        for field in LENGTH_FIELDS:
            length_arrays[condition][field] = np.asarray(
                [
                    _finite(by_condition[condition][example_id][field], field)
                    for example_id in collection_ids
                ],
                dtype=np.float64,
            )

    for example_id in collection_ids:
        current_label = current_labels.get(example_id)
        short_label = short_labels.get(example_id)
        if current_label is None or short_label is None:
            paired_correctness = "missing_paired_label"
        elif not current_label and not short_label:
            paired_correctness = "both_correct"
        elif not current_label and short_label:
            paired_correctness = "current_only_correct"
        elif current_label and not short_label:
            paired_correctness = "short_only_correct"
        else:
            paired_correctness = "both_incorrect"
        row: dict[str, Any] = {
            "example_id": example_id,
            "current_incorrect": current_label,
            "short_incorrect": short_label,
            "paired_correctness": paired_correctness,
        }
        for condition in ("current", "short"):
            for method, field in METHOD_FIELDS.items():
                row[f"{condition}_{method}"] = _finite(
                    by_condition[condition][example_id][field], field
                )
            for field in LENGTH_FIELDS:
                row[f"{condition}_{field}"] = _finite(
                    by_condition[condition][example_id][field], field
                )
        per_example.append(row)

    metric_rows: list[dict[str, Any]] = []
    for condition in ("current", "short"):
        labels = arrays[condition]["incorrect"]
        for method in METHOD_FIELDS:
            metrics = binary_ranking_metrics(labels, arrays[condition][method])
            metric_rows.append(
                {
                    "condition": condition,
                    "method": method,
                    "examples": len(paired_ids),
                    "incorrect": int(labels.sum()),
                    "auroc": metrics["auroc"],
                    "average_precision": metrics["average_precision"],
                }
            )

    current_d = _auroc(
        arrays["current"]["incorrect"], arrays["current"]["blind_p_true"]
    ) - _auroc(
        arrays["current"]["incorrect"], arrays["current"]["discrete_semantic_entropy"]
    )
    short_d = _auroc(
        arrays["short"]["incorrect"], arrays["short"]["blind_p_true"]
    ) - _auroc(
        arrays["short"]["incorrect"], arrays["short"]["discrete_semantic_entropy"]
    )
    observed = {
        "current_accuracy": float(1.0 - arrays["current"]["incorrect"].mean()),
        "short_accuracy": float(1.0 - arrays["short"]["incorrect"].mean()),
        "accuracy_difference_short_minus_current": float(
            arrays["current"]["incorrect"].mean() - arrays["short"]["incorrect"].mean()
        ),
        "main_word_difference_short_minus_current": float(
            np.mean(
                length_arrays["short"]["main_word_count"]
                - length_arrays["current"]["main_word_count"]
            )
        ),
        "main_sentence_difference_short_minus_current": float(
            np.mean(
                length_arrays["short"]["main_sentence_count"]
                - length_arrays["current"]["main_sentence_count"]
            )
        ),
        "d_current": current_d,
        "d_short": short_d,
        "length_effect_d_short_minus_d_current": short_d - current_d,
    }
    for method in METHOD_FIELDS:
        current_auroc = _auroc(
            arrays["current"]["incorrect"], arrays["current"][method]
        )
        short_auroc = _auroc(arrays["short"]["incorrect"], arrays["short"][method])
        current_ap = _average_precision(
            arrays["current"]["incorrect"], arrays["current"][method]
        )
        short_ap = _average_precision(
            arrays["short"]["incorrect"], arrays["short"][method]
        )
        observed.update(
            {
                f"current_{method}_auroc": current_auroc,
                f"short_{method}_auroc": short_auroc,
                f"{method}_auroc_short_minus_current": short_auroc - current_auroc,
                f"current_{method}_average_precision": current_ap,
                f"short_{method}_average_precision": short_ap,
                f"{method}_average_precision_short_minus_current": short_ap - current_ap,
            }
        )

    rng = np.random.default_rng(bootstrap_seed)
    bootstrap: dict[str, list[float]] = {name: [] for name in observed}
    valid_ranking_resamples = 0
    for _ in range(bootstrap_resamples):
        length_sample = rng.integers(
            0, len(collection_ids), size=len(collection_ids)
        )
        paired_sample = rng.integers(0, len(paired_ids), size=len(paired_ids))
        current_y = arrays["current"]["incorrect"][paired_sample]
        short_y = arrays["short"]["incorrect"][paired_sample]
        bootstrap["current_accuracy"].append(float(1.0 - current_y.mean()))
        bootstrap["short_accuracy"].append(float(1.0 - short_y.mean()))
        bootstrap["accuracy_difference_short_minus_current"].append(
            float(current_y.mean() - short_y.mean())
        )
        bootstrap["main_word_difference_short_minus_current"].append(
            float(
                np.mean(
                    length_arrays["short"]["main_word_count"][length_sample]
                    - length_arrays["current"]["main_word_count"][length_sample]
                )
            )
        )
        bootstrap["main_sentence_difference_short_minus_current"].append(
            float(
                np.mean(
                    length_arrays["short"]["main_sentence_count"][length_sample]
                    - length_arrays["current"]["main_sentence_count"][length_sample]
                )
            )
        )
        if len(np.unique(current_y)) < 2 or len(np.unique(short_y)) < 2:
            continue
        for method in METHOD_FIELDS:
            current_scores = arrays["current"][method][paired_sample]
            short_scores = arrays["short"][method][paired_sample]
            current_auroc = _auroc(current_y, current_scores)
            short_auroc = _auroc(short_y, short_scores)
            current_ap = _average_precision(current_y, current_scores)
            short_ap = _average_precision(short_y, short_scores)
            bootstrap[f"current_{method}_auroc"].append(current_auroc)
            bootstrap[f"short_{method}_auroc"].append(short_auroc)
            bootstrap[f"{method}_auroc_short_minus_current"].append(
                short_auroc - current_auroc
            )
            bootstrap[f"current_{method}_average_precision"].append(current_ap)
            bootstrap[f"short_{method}_average_precision"].append(short_ap)
            bootstrap[
                f"{method}_average_precision_short_minus_current"
            ].append(short_ap - current_ap)
        current_gap = _auroc(
            current_y, arrays["current"]["blind_p_true"][paired_sample]
        ) - _auroc(
            current_y,
            arrays["current"]["discrete_semantic_entropy"][paired_sample],
        )
        short_gap = _auroc(
            short_y, arrays["short"]["blind_p_true"][paired_sample]
        ) - _auroc(
            short_y, arrays["short"]["discrete_semantic_entropy"][paired_sample]
        )
        bootstrap["d_current"].append(current_gap)
        bootstrap["d_short"].append(short_gap)
        bootstrap["length_effect_d_short_minus_d_current"].append(
            short_gap - current_gap
        )
        valid_ranking_resamples += 1

    interval_rows = []
    for metric, value in observed.items():
        lower, upper = percentile_interval(bootstrap[metric])
        interval_rows.append(
            {
                "metric": metric,
                "observed": value,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "valid_resamples": len(bootstrap[metric]),
                "requested_resamples": bootstrap_resamples,
                "seed": bootstrap_seed,
            }
        )
    summary = {
        "schema_version": "bioasq_summary_length_comparison_v1",
        "collection_examples": len(collection_ids),
        "paired_valid_label_examples": len(paired_ids),
        "missing_paired_label_ids": sorted(expected_ids.difference(paired_ids)),
        "positive_class": "Claude incorrect",
        "average_precision_caveat": (
            "AP changes partly with condition-specific error prevalence; use AUROC "
            "for the primary cross-condition ranking comparison."
        ),
        "observed": observed,
        "paired_correctness_counts": {
            category: sum(row["paired_correctness"] == category for row in per_example)
            for category in (
                "both_correct",
                "current_only_correct",
                "short_only_correct",
                "both_incorrect",
                "missing_paired_label",
            )
        },
        "length_and_compliance": {
            condition: {
                field: float(length_arrays[condition][field].mean())
                for field in LENGTH_FIELDS
            }
            for condition in ("current", "short")
        },
        "bootstrap": {
            "requested_resamples": bootstrap_resamples,
            "valid_ranking_resamples": valid_ranking_resamples,
            "seed": bootstrap_seed,
            "paired_question_id_resampling": True,
        },
    }
    return per_example, metric_rows + interval_rows, summary


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    current_rows = _read_csv(args.collection_dir / "current" / "condition_scores.csv")
    short_rows = _read_csv(args.collection_dir / "short" / "condition_scores.csv")
    per_example, result_rows, summary = analyze(
        current_rows=current_rows,
        short_rows=short_rows,
        current_labels=load_labels(args.current_labels),
        short_labels=load_labels(args.short_labels),
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary["input_sha256"] = {
        "collection_summary": sha256_file(args.collection_dir / "collection_summary.json"),
        "sentence_count_repair": sha256_file(
            args.collection_dir / "sentence_count_repair.json"
        ),
        "current_scores": sha256_file(
            args.collection_dir / "current" / "condition_scores.csv"
        ),
        "short_scores": sha256_file(
            args.collection_dir / "short" / "condition_scores.csv"
        ),
        "current_labels": sha256_file(args.current_labels),
        "short_labels": sha256_file(args.short_labels),
    }
    _write_csv(args.output_dir / "paired_predictions.csv", per_example)
    metric_rows = [row for row in result_rows if "condition" in row]
    interval_rows = [row for row in result_rows if "metric" in row]
    _write_csv(args.output_dir / "uq_metrics.csv", metric_rows)
    _write_csv(args.output_dir / "paired_bootstrap.csv", interval_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed paired summary-length analysis: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
