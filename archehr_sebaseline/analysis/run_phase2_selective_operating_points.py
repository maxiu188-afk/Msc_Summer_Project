#!/usr/bin/env python3
"""Evaluate validation-fixed selective-prediction operating points on BioASQ.

Thresholds are selected from the frozen Phase-2 validation scores only and are
then applied unchanged to the frozen Phase-2 test scores.  Test labels never
fit or alter a threshold.  Equal-to-threshold examples are retained, so tied
scores can make achieved coverage exceed the requested validation coverage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


METHOD_FIELDS = {
    "discrete_semantic_entropy": "discrete_semantic_entropy",
    "blind_p_true": "blind_p_true_uncertainty",
    "accuracy_probe": "accuracy_probe_uncertainty",
    "p_true_probe": "p_true_probe_uncertainty",
}
PRIMARY_METHODS = (
    "discrete_semantic_entropy",
    "blind_p_true",
    "accuracy_probe",
)
ALL_METHODS = PRIMARY_METHODS + ("p_true_probe",)
TARGET_COVERAGES = (0.80, 0.90, 0.95)
EXPECTED_SHA256 = {
    "validation": "cc86e63cd1197d3b896fc8e14808c2fa3be308ce96961b40a2b3d5e0962d6245",
    "test": "181e5a77e3a7ba13a61577b6edb1d19c25539cdaf9a788b7ebe5b59785fad760",
}
EXPECTED_TYPE_COUNTS = {
    "validation": {"factoid": 157, "list": 104, "summary": 123},
    "test": {"factoid": 158, "list": 103, "summary": 123},
}
EXPECTED_ERRORS = {"validation": 265, "test": 252}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validation-uq-scores",
        type=Path,
        default=project_root
        / "analysis_outputs"
        / "bioasq_phase2_uq_calibration_validation384_20260801"
        / "uq_scores.csv",
    )
    parser.add_argument(
        "--test-uq-scores",
        type=Path,
        default=project_root
        / "analysis_outputs"
        / "bioasq_phase2_uq_efficiency_seed31_test384_20260723"
        / "uq_scores.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260820)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_path(path: Path, *, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def finite_array(values: list[Any], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value.") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def load_uq_split(
    path: Path,
    *,
    split: str,
    expected_sha256: str | None,
    expected_examples: int = 384,
    expected_type_counts: dict[str, int] | None = None,
    expected_errors: int | None = None,
) -> dict[str, Any]:
    if expected_sha256 is not None:
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{split} UQ SHA-256 mismatch: {observed_sha256} != {expected_sha256}."
            )
    else:
        observed_sha256 = sha256_file(path)

    rows = read_csv(path)
    if len(rows) != expected_examples:
        raise ValueError(
            f"Expected {expected_examples} {split} rows, got {len(rows)}."
        )
    ids = [str(row.get("example_id") or "") for row in rows]
    if any(not example_id for example_id in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{split} contains an empty or duplicate example ID.")

    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    labels = np.asarray([int(row["incorrect"]) for row in ordered], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{split} labels do not contain both classes.")
    if expected_errors is not None and int(np.sum(labels)) != expected_errors:
        raise ValueError(
            f"Expected {expected_errors} {split} errors, got {int(np.sum(labels))}."
        )

    bioasq_types = [str(row["bioasq_type"]) for row in ordered]
    observed_type_counts = dict(Counter(bioasq_types))
    if expected_type_counts is not None and observed_type_counts != expected_type_counts:
        raise ValueError(
            f"{split} type counts differ: {observed_type_counts} != {expected_type_counts}."
        )

    scores = {
        method: finite_array(
            [row[field] for row in ordered], name=f"{split} {method}"
        )
        for method, field in METHOD_FIELDS.items()
    }
    return {
        "split": split,
        "path": path,
        "sha256": observed_sha256,
        "example_ids": [str(row["example_id"]) for row in ordered],
        "bioasq_types": bioasq_types,
        "labels": labels,
        "scores": scores,
    }


def select_inclusive_threshold(scores: np.ndarray, target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("Target coverage must be in (0, 1].")
    retained = max(1, int(math.ceil(len(scores) * target_coverage)))
    return float(np.sort(scores, kind="stable")[retained - 1])


def evaluate_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int | None]:
    accepted = scores <= threshold
    retained_examples = int(np.sum(accepted))
    rejected_examples = int(len(labels) - retained_examples)
    if retained_examples == 0:
        raise ValueError("The fixed threshold retained no examples.")
    retained_errors = int(np.sum(labels[accepted]))
    rejected_errors = int(np.sum(labels[~accepted]))
    full_risk = float(np.mean(labels))
    retained_risk = retained_errors / retained_examples
    rejected_risk = rejected_errors / rejected_examples if rejected_examples else None
    absolute_reduction = full_risk - retained_risk
    return {
        "examples": int(len(labels)),
        "errors": int(np.sum(labels)),
        "full_error_risk": full_risk,
        "retained_examples": retained_examples,
        "retained_errors": retained_errors,
        "achieved_coverage": retained_examples / len(labels),
        "retained_error_risk": retained_risk,
        "retained_accuracy": 1.0 - retained_risk,
        "rejected_examples": rejected_examples,
        "rejected_errors": rejected_errors,
        "rejected_error_risk": rejected_risk,
        "absolute_risk_reduction": absolute_reduction,
        "relative_risk_reduction": absolute_reduction / full_risk,
        "accepted_error_burden": retained_errors / len(labels),
    }


def bootstrap_intervals(
    labels: np.ndarray,
    accepted: np.ndarray,
    indexes: np.ndarray,
) -> list[dict[str, Any]]:
    sampled_labels = labels[indexes]
    sampled_accepted = accepted[indexes]
    retained_counts = np.sum(sampled_accepted, axis=1)
    rejected_counts = len(labels) - retained_counts
    retained_errors = np.sum(sampled_labels * sampled_accepted, axis=1)
    rejected_errors = np.sum(sampled_labels * ~sampled_accepted, axis=1)
    full_risk = np.mean(sampled_labels, axis=1)
    retained_risk = retained_errors / retained_counts
    rejected_risk = np.divide(
        rejected_errors,
        rejected_counts,
        out=np.full(len(indexes), np.nan, dtype=np.float64),
        where=rejected_counts > 0,
    )
    absolute_reduction = full_risk - retained_risk
    relative_reduction = np.divide(
        absolute_reduction,
        full_risk,
        out=np.full(len(indexes), np.nan, dtype=np.float64),
        where=full_risk > 0,
    )
    values = {
        "achieved_coverage": retained_counts / len(labels),
        "retained_error_risk": retained_risk,
        "retained_accuracy": 1.0 - retained_risk,
        "rejected_error_risk": rejected_risk,
        "absolute_risk_reduction": absolute_reduction,
        "relative_risk_reduction": relative_reduction,
    }
    rows = []
    for metric, distribution in values.items():
        finite = distribution[np.isfinite(distribution)]
        rows.append(
            {
                "metric": metric,
                "ci_lower_95": (
                    float(np.quantile(finite, 0.025)) if len(finite) else None
                ),
                "ci_upper_95": (
                    float(np.quantile(finite, 0.975)) if len(finite) else None
                ),
                "valid_resamples": int(len(finite)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_plot(
    svg_path: Path,
    png_path: Path,
    rows: list[dict[str, Any]],
    full_risk: float,
) -> None:
    import matplotlib.pyplot as plt

    display = {
        "accuracy_probe": "Accuracy-Probe",
        "blind_p_true": "Blind P(True)",
        "p_true_probe": "P(True)-Probe",
        "discrete_semantic_entropy": "Semantic Entropy",
    }
    figure, axis = plt.subplots(figsize=(5.8, 4.8))
    for method in ALL_METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: float(row["target_coverage"]))
        axis.plot(
            [float(row["test_achieved_coverage"]) for row in method_rows],
            [float(row["test_retained_error_risk"]) for row in method_rows],
            marker="o",
            linewidth=1.6,
            label=display[method],
        )
    axis.axhline(
        full_risk,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="Full-coverage risk",
    )
    axis.set_xlabel("Test coverage from validation-fixed threshold")
    axis.set_ylabel("Retained test error risk")
    axis.set_xlim(0.74, 1.01)
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.2)
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(svg_path)
    figure.savefig(png_path, dpi=180)
    plt.close(figure)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    output_dir = args.output_dir.resolve()
    project_root = Path(__file__).resolve().parents[1]
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}.")
    output_dir.mkdir(parents=True, exist_ok=True)

    validation = load_uq_split(
        args.validation_uq_scores.resolve(),
        split="validation",
        expected_sha256=EXPECTED_SHA256["validation"],
        expected_type_counts=EXPECTED_TYPE_COUNTS["validation"],
        expected_errors=EXPECTED_ERRORS["validation"],
    )
    test = load_uq_split(
        args.test_uq_scores.resolve(),
        split="test",
        expected_sha256=EXPECTED_SHA256["test"],
        expected_type_counts=EXPECTED_TYPE_COUNTS["test"],
        expected_errors=EXPECTED_ERRORS["test"],
    )
    overlap = set(validation["example_ids"]) & set(test["example_ids"])
    if overlap:
        raise ValueError(f"Validation/test ID overlap: {len(overlap)} examples.")

    rng = np.random.default_rng(args.random_seed)
    bootstrap_indexes = rng.integers(
        0,
        len(test["labels"]),
        size=(args.bootstrap_samples, len(test["labels"])),
        dtype=np.int32,
    )

    overall_rows = []
    type_rows = []
    bootstrap_rows = []
    prediction_rows = []
    test_types = np.asarray(test["bioasq_types"], dtype=object)
    for method in ALL_METHODS:
        validation_scores = validation["scores"][method]
        test_scores = test["scores"][method]
        for target_coverage in TARGET_COVERAGES:
            threshold = select_inclusive_threshold(validation_scores, target_coverage)
            validation_metrics = evaluate_threshold(
                validation["labels"], validation_scores, threshold
            )
            test_metrics = evaluate_threshold(test["labels"], test_scores, threshold)
            overall_rows.append(
                {
                    "method": method,
                    "role": "primary" if method in PRIMARY_METHODS else "auxiliary",
                    "target_coverage": target_coverage,
                    "validation_score_threshold": threshold,
                    "validation_scores_below_threshold": int(
                        np.sum(validation_scores < threshold)
                    ),
                    "validation_scores_equal_threshold": int(
                        np.sum(validation_scores == threshold)
                    ),
                    "test_scores_below_threshold": int(np.sum(test_scores < threshold)),
                    "test_scores_equal_threshold": int(np.sum(test_scores == threshold)),
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in test_metrics.items()},
                    "test_coverage_drift": float(test_metrics["achieved_coverage"])
                    - target_coverage,
                }
            )

            accepted = test_scores <= threshold
            for interval in bootstrap_intervals(
                test["labels"], accepted, bootstrap_indexes
            ):
                bootstrap_rows.append(
                    {
                        "method": method,
                        "role": "primary" if method in PRIMARY_METHODS else "auxiliary",
                        "target_coverage": target_coverage,
                        "validation_score_threshold": threshold,
                        "uncertainty_scope": "fixed_validation_threshold_test_row_bootstrap",
                        "requested_resamples": args.bootstrap_samples,
                        **interval,
                    }
                )

            for bioasq_type in ("factoid", "list", "summary"):
                mask = test_types == bioasq_type
                metrics = evaluate_threshold(
                    test["labels"][mask], test_scores[mask], threshold
                )
                type_rows.append(
                    {
                        "method": method,
                        "role": "primary" if method in PRIMARY_METHODS else "auxiliary",
                        "target_coverage": target_coverage,
                        "validation_score_threshold": threshold,
                        "bioasq_type": bioasq_type,
                        **metrics,
                    }
                )

            for index, example_id in enumerate(test["example_ids"]):
                prediction_rows.append(
                    {
                        "example_id": example_id,
                        "bioasq_type": test["bioasq_types"][index],
                        "incorrect": int(test["labels"][index]),
                        "method": method,
                        "role": "primary" if method in PRIMARY_METHODS else "auxiliary",
                        "target_coverage": target_coverage,
                        "validation_score_threshold": threshold,
                        "raw_uncertainty_score": float(test_scores[index]),
                        "retained": bool(accepted[index]),
                    }
                )

    overall_path = output_dir / "operating_points_overall.csv"
    type_path = output_dir / "operating_points_by_type.csv"
    bootstrap_path = output_dir / "bootstrap_intervals.csv"
    predictions_path = output_dir / "operating_point_predictions.csv"
    plot_path = output_dir / "operating_points.svg"
    plot_png_path = output_dir / "operating_points.png"
    write_csv(overall_path, overall_rows)
    write_csv(type_path, type_rows)
    write_csv(bootstrap_path, bootstrap_rows)
    write_csv(predictions_path, prediction_rows)
    write_plot(
        plot_path,
        plot_png_path,
        overall_rows,
        float(np.mean(test["labels"])),
    )

    summary = {
        "status": "complete",
        "protocol": {
            "selection_split": "BioASQ Phase-2 validation only",
            "evaluation_split": "BioASQ Phase-2 test only",
            "target_coverages": list(TARGET_COVERAGES),
            "threshold_rule": "ceil(N * target coverage) validation order statistic",
            "retention_rule": "retain raw uncertainty score <= validation threshold",
            "tie_policy": "retain every score equal to the threshold; report achieved coverage",
            "test_used_for_threshold_fitting_or_selection": False,
            "type_specific_thresholds": False,
            "calibration_refit": False,
        },
        "methods": {
            "primary": list(PRIMARY_METHODS),
            "auxiliary": ["p_true_probe"],
        },
        "inputs": {
            "validation_uq_scores": {
                "path": portable_path(
                    args.validation_uq_scores, project_root=project_root
                ),
                "sha256": validation["sha256"],
                "examples": len(validation["labels"]),
                "errors": int(np.sum(validation["labels"])),
            },
            "test_uq_scores": {
                "path": portable_path(args.test_uq_scores, project_root=project_root),
                "sha256": test["sha256"],
                "examples": len(test["labels"]),
                "errors": int(np.sum(test["labels"])),
            },
            "validation_test_id_overlap": 0,
        },
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "random_seed": args.random_seed,
            "unit": "paired test row",
            "thresholds_refitted_inside_bootstrap": False,
        },
        "overall_results": overall_rows,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in (
                overall_path,
                type_path,
                bootstrap_path,
                predictions_path,
                plot_path,
                plot_png_path,
            )
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
