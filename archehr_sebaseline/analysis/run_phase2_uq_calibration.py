#!/usr/bin/env python3
"""Calibrate and evaluate the frozen Phase-2 UQ methods without test fitting.

The scalar calibration models are fitted once on the labelled Phase-2
validation split and applied unchanged to the Phase-2 test split.  The same
BioASQ-fitted mappings can optionally be applied to PubMedQA as a zero-refit
calibration-transfer diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


METHOD_FIELDS = {
    "discrete_semantic_entropy": "discrete_semantic_entropy",
    "blind_p_true": "blind_p_true_uncertainty",
    "accuracy_probe": "accuracy_probe",
    "p_true_probe": "p_true_probe",
}
PRIMARY_METHODS = ("discrete_semantic_entropy", "blind_p_true", "accuracy_probe")
ALL_METHODS = PRIMARY_METHODS + ("p_true_probe",)
NATIVE_PROBABILITY_METHODS = ("blind_p_true", "accuracy_probe", "p_true_probe")
PUBMEDQA_FIELDS = {
    "blind_p_true": "p_true_blind_uncertainty",
    "accuracy_probe": "accuracy_probe_score",
    "p_true_probe": "p_true_probe_score",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-predictions",
        type=Path,
        default=project_root
        / "analysis_outputs"
        / "bioasq_phase2_probe_completion_20260725"
        / "predictions.csv",
    )
    parser.add_argument("--validation-uq-scores", type=Path, default=None)
    parser.add_argument("--test-uq-scores", type=Path, default=None)
    parser.add_argument(
        "--exclude-semantic-entropy",
        action="store_true",
        help="Run the predeclared interim analysis for P(True)/Probes only.",
    )
    parser.add_argument("--pubmedqa-predictions", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260801)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def by_id(rows: list[dict[str, str]], *, source: str) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        example_id = str(row.get("example_id") or "")
        if not example_id or example_id in result:
            raise ValueError(f"{source} contains an empty or duplicate example ID.")
        result[example_id] = row
    return result


def finite_array(values: list[Any], *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-numeric value.") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional array.")
    return array


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}.")
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_plots(
    output_dir: Path,
    reliability_rows: list[dict[str, Any]],
    risk_rows: list[dict[str, Any]],
    *,
    overwrite: bool,
) -> None:
    import matplotlib.pyplot as plt

    reliability_path = output_dir / "reliability_diagram.svg"
    risk_path = output_dir / "risk_coverage.svg"
    for path in (reliability_path, risk_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}.")

    display = {
        "accuracy_probe": "Accuracy-Probe",
        "blind_p_true": "Blind P(True)",
        "p_true_probe": "P(True)-Probe",
        "discrete_semantic_entropy": "Semantic Entropy",
    }
    methods = [method for method in ALL_METHODS if any(row["method"] == method for row in reliability_rows)]

    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    axis.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0)
    for method in methods:
        rows = [row for row in reliability_rows if row["method"] == method]
        rows.sort(key=lambda row: int(row["bin_index"]))
        axis.plot(
            [float(row["mean_predicted_risk"]) for row in rows],
            [float(row["observed_error_rate"]) for row in rows],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=display[method],
        )
    axis.set_xlabel("Mean calibrated error probability")
    axis.set_ylabel("Observed error rate")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(reliability_path)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.4, 4.8))
    for method in methods:
        rows = [row for row in risk_rows if row["method"] == method]
        rows.sort(key=lambda row: float(row["coverage"]))
        axis.plot(
            [float(row["coverage"]) for row in rows],
            [float(row["retained_error_risk"]) for row in rows],
            marker="o",
            linewidth=1.6,
            markersize=4,
            label=display[method],
        )
    axis.set_xlabel("Coverage")
    axis.set_ylabel("Retained error risk")
    axis.set_xlim(0.5, 1.0)
    axis.set_ylim(bottom=0.0)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(risk_path)
    plt.close(figure)


class ScalarLogisticCalibrator:
    """One-dimensional monotonic logistic calibration fitted on validation."""

    def __init__(self, *, random_seed: int) -> None:
        self.random_seed = random_seed
        self.mean_: float = 0.0
        self.scale_: float = 1.0
        self.model_: LogisticRegression | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "ScalarLogisticCalibrator":
        self.mean_ = float(np.mean(scores))
        scale = float(np.std(scores))
        self.scale_ = scale if scale > 0.0 else 1.0
        standardized = ((scores - self.mean_) / self.scale_).reshape(-1, 1)
        model = LogisticRegression(
            C=1_000_000.0,
            solver="lbfgs",
            max_iter=10_000,
            random_state=self.random_seed,
        )
        model.fit(standardized, labels)
        coefficient = float(model.coef_[0, 0])
        if not math.isfinite(coefficient) or coefficient <= 0.0:
            raise ValueError(
                "Calibration slope is not positive; refusing a post-hoc score-direction flip."
            )
        self.model_ = model
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Calibrator has not been fitted.")
        standardized = ((scores - self.mean_) / self.scale_).reshape(-1, 1)
        probabilities = self.model_.predict_proba(standardized)[:, 1]
        return np.asarray(probabilities, dtype=np.float64)

    def parameters(self) -> dict[str, float]:
        if self.model_ is None:
            raise RuntimeError("Calibrator has not been fitted.")
        return {
            "score_mean": self.mean_,
            "score_scale": self.scale_,
            "standardized_coefficient": float(self.model_.coef_[0, 0]),
            "intercept": float(self.model_.intercept_[0]),
        }


def load_split(
    base_rows: list[dict[str, str]],
    uq_path: Path | None,
    split: str,
    *,
    methods: tuple[str, ...] = ALL_METHODS,
) -> dict[str, Any]:
    base = [row for row in base_rows if str(row.get("split")) == split]
    uq = by_id(read_csv(uq_path), source=str(uq_path)) if uq_path is not None else None
    if len(base) != 384:
        raise ValueError(f"Expected 384 base {split} rows, got {len(base)}.")
    base_ids = {str(row["example_id"]) for row in base}
    if uq is not None and base_ids != set(uq):
        raise ValueError(
            f"{split} base/UQ IDs differ: base_only={len(base_ids - set(uq))}, "
            f"uq_only={len(set(uq) - base_ids)}."
        )
    ordered = sorted(base, key=lambda row: str(row["example_id"]))
    labels = np.asarray([int(row["incorrect"]) for row in ordered], dtype=np.int64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError(f"{split} labels do not contain both classes.")
    scores: dict[str, np.ndarray] = {}
    for method in methods:
        if method == "discrete_semantic_entropy":
            if uq is None:
                raise ValueError(
                    f"{split} Semantic Entropy requires the matched UQ-score file."
                )
            values = [uq[str(row["example_id"])]["discrete_semantic_entropy"] for row in ordered]
        else:
            values = [row[METHOD_FIELDS[method]] for row in ordered]
        scores[method] = finite_array(values, name=f"{split} {method}")

    consistency_fields = {
        "blind_p_true": "blind_p_true_uncertainty",
        "accuracy_probe": "accuracy_probe_uncertainty",
        "p_true_probe": "p_true_probe_uncertainty",
    }
    for method, uq_field in consistency_fields.items():
        if uq is None or method not in methods:
            continue
        replay = finite_array(
            [uq[str(row["example_id"])][uq_field] for row in ordered],
            name=f"{split} replay {method}",
        )
        if not np.allclose(scores[method], replay, rtol=0.0, atol=1e-8):
            raise ValueError(f"{split} saved and replayed {method} scores differ.")
    return {
        "split": split,
        "example_ids": [str(row["example_id"]) for row in ordered],
        "bioasq_types": [str(row["bioasq_type"]) for row in ordered],
        "labels": labels,
        "scores": scores,
    }


def clipped(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(probabilities, 1e-8, 1.0 - 1e-8)


def log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = clipped(probabilities)
    return float(
        -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    )


def equal_frequency_ece(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> tuple[float, list[dict[str, Any]]]:
    if bins <= 1 or bins > len(labels):
        raise ValueError("bins must be between 2 and the number of examples.")
    order = np.argsort(probabilities, kind="mergesort")
    groups = np.array_split(order, bins)
    ece = 0.0
    rows = []
    for index, group in enumerate(groups):
        mean_probability = float(np.mean(probabilities[group]))
        observed_risk = float(np.mean(labels[group]))
        weight = len(group) / len(labels)
        ece += weight * abs(mean_probability - observed_risk)
        rows.append(
            {
                "bin_index": index,
                "examples": len(group),
                "minimum_probability": float(np.min(probabilities[group])),
                "maximum_probability": float(np.max(probabilities[group])),
                "mean_predicted_risk": mean_probability,
                "observed_error_rate": observed_risk,
            }
        )
    return float(ece), rows


def probability_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> dict[str, float]:
    brier = float(brier_score_loss(labels, probabilities))
    prevalence = float(np.mean(labels))
    baseline_brier = prevalence * (1.0 - prevalence)
    ece, _ = equal_frequency_ece(labels, probabilities, bins=bins)
    return {
        "examples": int(len(labels)),
        "errors": int(np.sum(labels)),
        "error_prevalence": prevalence,
        "brier": brier,
        "brier_constant_baseline": baseline_brier,
        "brier_skill": 1.0 - brier / baseline_brier,
        "log_loss": log_loss(labels, probabilities),
        "ece_equal_frequency": ece,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }


def risk_coverage(
    labels: np.ndarray,
    scores: np.ndarray,
) -> tuple[list[dict[str, float]], float]:
    order = np.argsort(scores, kind="mergesort")
    rows = []
    for index in range(11):
        requested = round(0.50 + 0.05 * index, 2)
        retained = min(
            len(labels), max(1, int(math.ceil(len(labels) * requested)))
        )
        indexes = order[:retained]
        rows.append(
            {
                "requested_coverage": requested,
                "coverage": retained / len(labels),
                "retained_examples": retained,
                "retained_error_risk": float(np.mean(labels[indexes])),
                "score_threshold": float(scores[indexes[-1]]),
            }
        )
    area = float(
        np.trapezoid(
            [row["retained_error_risk"] for row in rows],
            [row["coverage"] for row in rows],
        )
    )
    return rows, area


def method_metrics(
    labels: np.ndarray,
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int,
) -> dict[str, float]:
    _, aurac = risk_coverage(labels, raw_scores)
    return {
        **ranking_metrics(labels, raw_scores),
        **probability_metrics(labels, probabilities, bins=bins),
        "aurac_coverage_0p5_to_1p0": aurac,
    }


def bootstrap_differences(
    labels: np.ndarray,
    raw_scores: dict[str, np.ndarray],
    calibrated: dict[str, np.ndarray],
    *,
    methods: tuple[str, ...],
    samples: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive.")
    pairs = [("accuracy_probe", "blind_p_true")]
    if "discrete_semantic_entropy" in methods:
        pairs.extend(
            [
                ("accuracy_probe", "discrete_semantic_entropy"),
                ("blind_p_true", "discrete_semantic_entropy"),
            ]
        )
    metric_names = ("auroc", "brier", "log_loss", "aurac_coverage_0p5_to_1p0")

    def values(indexes: np.ndarray, method: str) -> dict[str, float]:
        y = labels[indexes]
        score = raw_scores[method][indexes]
        probability = calibrated[method][indexes]
        _, aurac = risk_coverage(y, score)
        return {
            "auroc": float(roc_auc_score(y, score)),
            "brier": float(brier_score_loss(y, probability)),
            "log_loss": log_loss(y, probability),
            "aurac_coverage_0p5_to_1p0": aurac,
        }

    observed = {method: values(np.arange(len(labels)), method) for method in methods}
    distributions: dict[tuple[str, str, str], list[float]] = {
        (candidate, comparator, metric): []
        for candidate, comparator in pairs
        for metric in metric_names
    }
    rng = np.random.default_rng(random_seed)
    skipped = 0
    for _ in range(samples):
        indexes = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[indexes])) != 2:
            skipped += 1
            continue
        sampled = {method: values(indexes, method) for method in methods}
        for candidate, comparator in pairs:
            for metric in metric_names:
                distributions[(candidate, comparator, metric)].append(
                    sampled[candidate][metric] - sampled[comparator][metric]
                )
    rows = []
    for candidate, comparator in pairs:
        for metric in metric_names:
            distribution = finite_array(
                distributions[(candidate, comparator, metric)],
                name=f"bootstrap {candidate} {comparator} {metric}",
            )
            rows.append(
                {
                    "candidate": candidate,
                    "comparator": comparator,
                    "metric": metric,
                    "difference": observed[candidate][metric] - observed[comparator][metric],
                    "ci_lower_95": float(np.quantile(distribution, 0.025)),
                    "ci_upper_95": float(np.quantile(distribution, 0.975)),
                    "requested_resamples": samples,
                    "valid_resamples": len(distribution),
                    "skipped_single_class": skipped,
                    "difference_direction": "candidate_minus_comparator",
                    "higher_is_better": metric == "auroc",
                }
            )
    return rows


def pubmedqa_transfer(
    path: Path,
    calibrators: dict[str, ScalarLogisticCalibrator],
    *,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(path)
    if len(rows) != 500:
        raise ValueError(f"Expected 500 PubMedQA rows, got {len(rows)}.")
    labels = np.asarray([int(row["incorrect"]) for row in rows], dtype=np.int64)
    metric_rows = []
    prediction_rows = []
    for method, field in PUBMEDQA_FIELDS.items():
        raw = finite_array([row[field] for row in rows], name=f"PubMedQA {method}")
        calibrated = calibrators[method].predict(raw)
        for calibration, probabilities in (("native", raw), ("bioasq_validation", calibrated)):
            metric_rows.append(
                {
                    "dataset": "pubmedqa_v2",
                    "method": method,
                    "calibration": calibration,
                    **ranking_metrics(labels, raw),
                    **probability_metrics(labels, probabilities, bins=bins),
                }
            )
        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "example_id": row["example_id"],
                    "incorrect": int(labels[index]),
                    "method": method,
                    "raw_score": float(raw[index]),
                    "bioasq_calibrated_error_probability": float(calibrated[index]),
                }
            )
    return metric_rows, prediction_rows


def main() -> int:
    args = parse_args()
    if args.bins != 10:
        raise ValueError("The frozen primary protocol requires exactly 10 bins.")
    active_primary_methods = (
        ("blind_p_true", "accuracy_probe")
        if args.exclude_semantic_entropy
        else PRIMARY_METHODS
    )
    active_methods = active_primary_methods + ("p_true_probe",)
    if not args.exclude_semantic_entropy and (
        args.validation_uq_scores is None or args.test_uq_scores is None
    ):
        raise ValueError(
            "Semantic Entropy analysis requires --validation-uq-scores and "
            "--test-uq-scores."
        )
    base_rows = read_csv(args.phase2_predictions.resolve())
    validation_uq_path = (
        args.validation_uq_scores.resolve()
        if args.validation_uq_scores is not None
        else None
    )
    test_uq_path = args.test_uq_scores.resolve() if args.test_uq_scores is not None else None
    validation = load_split(
        base_rows, validation_uq_path, "validation", methods=active_methods
    )
    test = load_split(base_rows, test_uq_path, "test", methods=active_methods)

    calibrators = {}
    calibrated = {}
    parameters = {}
    for method in active_methods:
        calibrator = ScalarLogisticCalibrator(random_seed=args.random_seed).fit(
            validation["scores"][method], validation["labels"]
        )
        calibrators[method] = calibrator
        calibrated[method] = calibrator.predict(test["scores"][method])
        parameters[method] = calibrator.parameters()

    metric_rows = []
    reliability_rows = []
    risk_rows = []
    type_rows = []
    prediction_rows = []
    for method in active_methods:
        raw = test["scores"][method]
        probability = calibrated[method]
        metric_rows.append(
            {
                "dataset": "bioasq_test",
                "method": method,
                "calibration": "validation_logistic",
                "role": "primary" if method in active_primary_methods else "auxiliary_cross_target",
                **method_metrics(test["labels"], raw, probability, bins=args.bins),
            }
        )
        if method in NATIVE_PROBABILITY_METHODS:
            metric_rows.append(
                {
                    "dataset": "bioasq_test",
                    "method": method,
                    "calibration": "native_uncalibrated",
                    "role": "diagnostic",
                    **method_metrics(test["labels"], raw, raw, bins=args.bins),
                }
            )
        _, bins = equal_frequency_ece(test["labels"], probability, bins=args.bins)
        for row in bins:
            reliability_rows.append(
                {"dataset": "bioasq_test", "method": method, **row}
            )
        curve, _ = risk_coverage(test["labels"], raw)
        for row in curve:
            risk_rows.append({"dataset": "bioasq_test", "method": method, **row})
        for index, example_id in enumerate(test["example_ids"]):
            prediction_rows.append(
                {
                    "split": "test",
                    "example_id": example_id,
                    "bioasq_type": test["bioasq_types"][index],
                    "incorrect": int(test["labels"][index]),
                    "method": method,
                    "raw_score": float(raw[index]),
                    "calibrated_error_probability": float(probability[index]),
                }
            )
        types = np.asarray(test["bioasq_types"], dtype=object)
        for question_type in ("factoid", "list", "summary"):
            selected = types == question_type
            type_rows.append(
                {
                    "dataset": "bioasq_test",
                    "bioasq_type": question_type,
                    "method": method,
                    "calibration": "global_validation_logistic",
                    **method_metrics(
                        test["labels"][selected], raw[selected], probability[selected], bins=args.bins
                    ),
                }
            )

    bootstrap_rows = bootstrap_differences(
        test["labels"],
        {method: test["scores"][method] for method in active_primary_methods},
        {method: calibrated[method] for method in active_primary_methods},
        methods=active_primary_methods,
        samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "metrics_overall.csv", metric_rows, overwrite=args.overwrite)
    write_csv(output_dir / "metrics_by_type.csv", type_rows, overwrite=args.overwrite)
    write_csv(output_dir / "reliability_bins.csv", reliability_rows, overwrite=args.overwrite)
    write_csv(output_dir / "risk_coverage.csv", risk_rows, overwrite=args.overwrite)
    write_csv(output_dir / "calibrated_predictions.csv", prediction_rows, overwrite=args.overwrite)
    write_csv(output_dir / "paired_bootstrap.csv", bootstrap_rows, overwrite=args.overwrite)
    write_plots(
        output_dir,
        reliability_rows,
        risk_rows,
        overwrite=args.overwrite,
    )

    pubmedqa_metrics = []
    if args.pubmedqa_predictions is not None:
        pubmedqa_metrics, pubmedqa_predictions = pubmedqa_transfer(
            args.pubmedqa_predictions.resolve(), calibrators, bins=args.bins
        )
        write_csv(
            output_dir / "pubmedqa_transfer_metrics.csv",
            pubmedqa_metrics,
            overwrite=args.overwrite,
        )
        write_csv(
            output_dir / "pubmedqa_transfer_predictions.csv",
            pubmedqa_predictions,
            overwrite=args.overwrite,
        )

    inputs = {
        "phase2_predictions": args.phase2_predictions.resolve(),
    }
    if validation_uq_path is not None:
        inputs["validation_uq_scores"] = validation_uq_path
    if test_uq_path is not None:
        inputs["test_uq_scores"] = test_uq_path
    if args.pubmedqa_predictions is not None:
        inputs["pubmedqa_predictions"] = args.pubmedqa_predictions.resolve()
    summary = {
        "schema_version": "phase2_uq_calibration_v1",
        "status": "partial_without_semantic_entropy" if args.exclude_semantic_entropy else "complete",
        "target": "incorrect=1",
        "primary_methods": list(active_primary_methods),
        "auxiliary_method": "p_true_probe",
        "semantic_entropy_included": not args.exclude_semantic_entropy,
        "calibration_fit_split": "validation",
        "evaluation_split": "test",
        "test_used_for_fitting_or_selection": False,
        "calibrator": "z-scored scalar logistic regression; fixed C=1e6; positive slope required",
        "reliability_bins": "10 equal-frequency bins on the evaluation split",
        "figures": ["reliability_diagram.svg", "risk_coverage.svg"],
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "random_seed": args.random_seed,
        },
        "rows": {"validation": 384, "test": 384, "pubmedqa": 500 if pubmedqa_metrics else 0},
        "calibrator_parameters": parameters,
        "input_sha256": {name: sha256_file(path) for name, path in inputs.items()},
    }
    write_json(output_dir / "summary.json", summary, overwrite=args.overwrite)
    print(
        f"calibration=PASS validation=384 test=384 se={not args.exclude_semantic_entropy} "
        f"pubmedqa={summary['rows']['pubmedqa']} "
        f"output={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
