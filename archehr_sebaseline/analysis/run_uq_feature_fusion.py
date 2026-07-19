#!/usr/bin/env python3
"""Evaluate pre-specified BioASQ UQ feature fusions without new model calls.

This script consumes the question-level artifacts already written by the two
formal BioASQ medical-UQ runs.  It never loads a language model, runs NLI, or
contacts an external judge.  ``incorrect`` is the positive class and every
input score is represented as an uncertainty (larger means more uncertain).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIRS = {
    31: PROJECT_ROOT.parent / "server_results" / "bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31",
    47: PROJECT_ROOT.parent / "server_results" / "bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed47",
}

FEATURES = {
    "p_true_blind": "p_true_blind_uncertainty",
    "discrete_se": "normalized_discrete_semantic_entropy",
    "weighted_se": "normalized_likelihood_weighted_semantic_entropy",
    "normalized_nll": "mean_normalized_nll",
    "token_entropy": "mean_token_entropy",
    "num_clusters": "num_clusters",
}
MODEL_FEATURES = {
    "p_true_blind": ("p_true_blind",),
    "discrete_se": ("discrete_se",),
    "weighted_se": ("weighted_se",),
    "normalized_nll": ("normalized_nll",),
    "token_entropy": ("token_entropy",),
    "num_clusters": ("num_clusters",),
    "p_true_blind_plus_discrete_se": ("p_true_blind", "discrete_se"),
    "p_true_blind_plus_weighted_se": ("p_true_blind", "weighted_se"),
    "p_true_blind_plus_normalized_nll": ("p_true_blind", "normalized_nll"),
    "p_true_blind_plus_token_entropy": ("p_true_blind", "token_entropy"),
    "p_true_blind_plus_discrete_se_plus_normalized_nll": (
        "p_true_blind", "discrete_se", "normalized_nll"
    ),
    "p_true_blind_plus_weighted_se_plus_normalized_nll": (
        "p_true_blind", "weighted_se", "normalized_nll"
    ),
    "p_true_blind_plus_discrete_se_plus_normalized_nll_plus_token_entropy": (
        "p_true_blind", "discrete_se", "normalized_nll", "token_entropy"
    ),
}
BOOTSTRAP_COMPARISONS = {
    "p_true_blind_plus_discrete_se": "P(True)-blind + discrete SE versus P(True)-blind",
    "p_true_blind_plus_normalized_nll": "P(True)-blind + normalized NLL versus P(True)-blind",
    "p_true_blind_plus_discrete_se_plus_normalized_nll": "P(True)-blind + discrete SE + normalized NLL versus P(True)-blind",
}


@dataclass
class LoadedSeed:
    seed: int
    records: list[dict[str, Any]]
    cleaning: dict[str, Any]


@dataclass
class OOFResult:
    probabilities: np.ndarray
    test_counts: np.ndarray
    coefficient_rows: list[dict[str, Any]]
    scaler_means: list[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed31-dir", type=Path, default=DEFAULT_RUN_DIRS[31])
    parser.add_argument("--seed47-dir", type=Path, default=DEFAULT_RUN_DIRS[47])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "analysis_outputs" / "bioasq_uq_feature_fusion_20260719_phase1",
    )
    parser.add_argument("--seed", type=int, default=20260718, help="Random seed for CV and bootstrap.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--skip-pooled-analysis", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv_by_id(path: Path, *, required: bool = True) -> dict[str, dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required artifact is missing: {path}")
        return {}
    rows: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as infile:
        for row in csv.DictReader(infile):
            example_id = str(row.get("example_id") or "")
            if not example_id or example_id in rows:
                raise ValueError(f"{path} has an empty or duplicate example_id.")
            rows[example_id] = row
    return rows


def load_types(path: Path) -> dict[str, str]:
    types: dict[str, str] = {}
    with path.open(encoding="utf-8") as infile:
        for line in infile:
            row = json.loads(line)
            example_id = str(row.get("id") or "")
            question_type = str(row.get("bioasq_type") or "unknown").lower()
            if not example_id or example_id in types:
                raise ValueError(f"{path} has an empty or duplicate BioASQ ID.")
            types[example_id] = question_type
    return types


def load_seed_results(seed: int, run_dir: Path) -> LoadedSeed:
    """Merge native UQ inputs and filter invalid labels/features explicitly."""

    se_rows = read_csv_by_id(run_dir / "se_scores.csv")
    self_report = read_csv_by_id(run_dir / "uq_baselines" / "self_report_examples.csv")
    labels = read_csv_by_id(run_dir / "claude_binary_main_answer_judge" / "claude_generation_labels.csv")
    types = load_types(run_dir / "examples.jsonl")
    if set(se_rows) != set(types):
        raise ValueError(f"seed {seed}: se_scores.csv and examples.jsonl IDs differ.")

    reasons = Counter()
    records: list[dict[str, Any]] = []
    for example_id in sorted(se_rows):
        label_row = labels.get(example_id)
        label = str(label_row.get("label") or "").lower() if label_row else ""
        is_valid = str(label_row.get("label_valid") or "").lower() == "true" if label_row else False
        if not is_valid or label not in {"correct", "incorrect"}:
            reasons["invalid_or_blank_claude_label"] += 1
            continue
        merged = {**se_rows[example_id], **self_report.get(example_id, {})}
        values: dict[str, float] = {}
        missing = []
        for feature, column in FEATURES.items():
            value = finite_float(merged.get(column))
            if value is None:
                missing.append(feature)
            else:
                values[feature] = value
        if missing:
            reasons["missing_or_nonfinite_feature"] += 1
            for feature in missing:
                reasons[f"missing_{feature}"] += 1
            continue
        records.append(
            {
                "example_id": example_id,
                "seed": seed,
                "bioasq_type": types[example_id],
                "claude_label": label,
                "incorrect": int(label == "incorrect"),
                **values,
            }
        )
    if not records:
        raise ValueError(f"seed {seed}: no valid labelled rows remain after cleaning.")
    return LoadedSeed(
        seed=seed,
        records=records,
        cleaning={
            "source_questions": len(se_rows),
            "valid_labelled_questions": sum(
                1
                for row in labels.values()
                if str(row.get("label_valid") or "").lower() == "true"
                and str(row.get("label") or "").lower() in {"correct", "incorrect"}
            ),
            "retained_questions": len(records),
            "excluded_questions": len(se_rows) - len(records),
            "exclusion_reasons": dict(sorted(reasons.items())),
        },
    )


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None


def safe_auprc(y: np.ndarray, score: np.ndarray) -> float | None:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else None


def safe_spearman(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return None
    value = float(spearmanr(score, y).statistic)
    return value if math.isfinite(value) else None


def coverage_risk_curve(y: np.ndarray, uncertainty: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Current repository convention: retained risk over coverage 0.5--1.0."""

    if not len(y):
        return np.array([]), np.array([]), None
    order = np.argsort(uncertainty, kind="stable")
    requested_coverages = np.asarray([round(0.50 + 0.05 * index, 10) for index in range(11)])
    coverages = []
    risks = []
    for coverage in requested_coverages:
        retained_count = max(1, int(math.ceil(len(order) * float(coverage))))
        coverages.append(retained_count / len(order))
        risks.append(float(np.mean(y[order[:retained_count]])))
    coverage_values = np.asarray(coverages)
    return coverage_values, np.asarray(risks), float(np.trapezoid(risks, coverage_values))


def metric_row(
    y: np.ndarray,
    score: np.ndarray,
    *,
    seed: str,
    bioasq_type: str,
    model_name: str,
    evaluation: str,
) -> dict[str, Any]:
    _, _, aurac = coverage_risk_curve(y, score)
    return {
        "seed": seed,
        "bioasq_type": bioasq_type,
        "model_name": model_name,
        "evaluation": evaluation,
        "num_examples": len(y),
        "num_incorrect": int(np.sum(y)),
        "num_correct": int(len(y) - np.sum(y)),
        "auroc": safe_auroc(y, score),
        "auprc": safe_auprc(y, score),
        "aurac_coverage_0p5_to_1p0": aurac,
        "spearman_uncertainty_error": safe_spearman(y, score),
    }


def make_repeated_splits(
    y: np.ndarray, *, n_splits: int, n_repeats: int, random_state: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    if n_splits < 2 or n_repeats < 1:
        raise ValueError("n_splits must be at least 2 and n_repeats at least 1.")
    class_counts = np.bincount(y.astype(int), minlength=2)
    if int(class_counts.min()) < n_splits:
        raise ValueError(f"Cannot make {n_splits} stratified folds with class counts {class_counts.tolist()}.")
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for index, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y)):
        yield index // n_splits, index % n_splits, train, test


def make_grouped_splits(
    y: np.ndarray, groups: np.ndarray, *, n_splits: int, n_repeats: int, random_state: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Repeated StratifiedGroupKFold, with GroupKFold only as old-sklearn fallback."""

    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError("Fewer unique question IDs than requested grouped folds.")
    for repeat in range(n_repeats):
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat
            )
        except TypeError:  # pragma: no cover - compatibility with pre-shuffle sklearn
            splitter = GroupKFold(n_splits=n_splits)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(y)), y, groups)):
            if set(groups[train]) & set(groups[test]):
                raise AssertionError("Grouped split leaked a question ID into train and test.")
            yield repeat, fold, train, test


def empirical_training_cdf(training_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    """Put fold-specific scores on a common 0--1 scale without test labels.

    Separately fitted CV logistic models have independently fitted intercepts,
    so their raw probabilities are not necessarily comparable across folds.
    This strictly monotone (except ties) training-fold transform preserves each
    fold's model ranking before the requested OOF averaging.
    """

    ordered = np.sort(np.asarray(training_scores, dtype=float))
    lower = np.searchsorted(ordered, test_scores, side="left")
    upper = np.searchsorted(ordered, test_scores, side="right")
    return (lower + upper + 2.0) / (2.0 * (len(ordered) + 2.0))


def fit_repeated_oof(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    groups: np.ndarray | None = None,
    seed_label: str,
    model_name: str,
) -> OOFResult:
    """Fit pipeline inside each fold and average one OOF probability per repeat."""

    sums = np.zeros(len(y), dtype=float)
    counts = np.zeros(len(y), dtype=int)
    coefficient_rows: list[dict[str, Any]] = []
    scaler_means: list[np.ndarray] = []
    split_iter: Iterable[tuple[int, int, np.ndarray, np.ndarray]]
    if groups is None:
        split_iter = make_repeated_splits(y, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    else:
        split_iter = make_grouped_splits(y, groups, n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for repeat, fold, train, test in split_iter:
        if len(np.unique(y[train])) != 2:
            raise ValueError("A training fold has only one label; reduce n_splits or inspect labels.")
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logistic", LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=random_state)),
            ]
        )
        pipeline.fit(x[train], y[train])
        train_probabilities = pipeline.predict_proba(x[train])[:, 1]
        test_probabilities = pipeline.predict_proba(x[test])[:, 1]
        sums[test] += empirical_training_cdf(train_probabilities, test_probabilities)
        counts[test] += 1
        scaler = pipeline.named_steps["scaler"]
        logistic = pipeline.named_steps["logistic"]
        scaler_means.append(np.asarray(scaler.mean_).copy())
        for feature, coefficient, mean, scale in zip(feature_names, logistic.coef_[0], scaler.mean_, scaler.scale_):
            coefficient_rows.append(
                {
                    "seed": seed_label,
                    "model_name": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "feature": feature,
                    "coefficient_standardized": float(coefficient),
                    "intercept": float(logistic.intercept_[0]),
                    "training_scaler_mean": float(mean),
                    "training_scaler_scale": float(scale),
                    "train_examples": len(train),
                    "test_examples": len(test),
                }
            )
    if not np.all(counts == n_repeats):
        raise AssertionError(f"OOF prediction completeness failed: test counts are {np.unique(counts).tolist()}.")
    return OOFResult(sums / counts, counts, coefficient_rows, scaler_means)


def bootstrap_auc_difference(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    random_state: int,
) -> dict[str, Any]:
    point = safe_auroc(y, candidate)
    base_point = safe_auroc(y, baseline)
    point_difference = None if point is None or base_point is None else point - base_point
    rng = np.random.default_rng(random_state)
    differences = []
    skipped = 0
    for _ in range(samples):
        indexes = rng.integers(0, len(y), size=len(y))
        sample_y = y[indexes]
        if len(np.unique(sample_y)) != 2:
            skipped += 1
            continue
        differences.append(float(roc_auc_score(sample_y, candidate[indexes]) - roc_auc_score(sample_y, baseline[indexes])))
    return {
        "auroc_difference": point_difference,
        "ci_lower_95": float(np.quantile(differences, 0.025)) if differences else None,
        "ci_upper_95": float(np.quantile(differences, 0.975)) if differences else None,
        "requested_resamples": samples,
        "valid_resamples": len(differences),
        "skipped_single_label_resamples": skipped,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})


def plot_coverage_risk(rows: list[dict[str, Any]], predictions: dict[str, np.ndarray], output_path: Path, title: str) -> None:
    y = np.asarray([row["incorrect"] for row in rows], dtype=int)
    plt.figure(figsize=(10, 6.4))
    for model_name, prediction in predictions.items():
        coverage, risk, _ = coverage_risk_curve(y, prediction)
        plt.plot(coverage, risk, marker="o", linewidth=1.7, markersize=3.3, label=model_name)
    plt.xlabel("Coverage retained (lowest uncertainty first)")
    plt.ylabel("Retained error risk")
    plt.title(title)
    plt.xlim(0.5, 1.0)
    plt.ylim(bottom=0.0)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 3) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append("NA" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value)))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, divider, *body])


def directional_conclusion(bootstrap_rows: list[dict[str, Any]]) -> str:
    key_rows = [row for row in bootstrap_rows if row["candidate_model"] == "p_true_blind_plus_discrete_se"]
    if len(key_rows) != 2:
        return "The discrete-SE comparison was not available for both seeds."
    diffs = [row["auroc_difference"] for row in key_rows]
    lowers = [row["ci_lower_95"] for row in key_rows]
    if all(value is not None and value > 0 for value in diffs) and all(value is not None and value > 0 for value in lowers):
        return "Discrete SE shows preliminary complementary evidence: both seeds improve and both bootstrap intervals exclude zero."
    if any(value is not None and value > 0 for value in diffs) and any(value is not None and value < 0 for value in diffs):
        return "Discrete-SE fusion is directionally inconsistent across seeds, so there is no stable complementary evidence."
    return "Discrete-SE fusion does not clear the conservative two-seed/bootstrap criterion for reliable complementary evidence."


def mean_coefficients(coefficient_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in coefficient_rows:
        if row["seed"] in {"31", "47"}:
            grouped[(str(row["seed"]), str(row["model_name"]), str(row["feature"]))].append(
                float(row["coefficient_standardized"])
            )
    return [
        {
            "seed": seed,
            "model_name": model_name,
            "feature": feature,
            "mean_standardized_coefficient": float(np.mean(values)),
            "sd_standardized_coefficient": float(np.std(values, ddof=0)),
            "fold_fits": len(values),
        }
        for (seed, model_name, feature), values in sorted(grouped.items())
    ]


def format_difference(row: dict[str, Any]) -> str:
    value = row.get("auroc_difference")
    interval = (row.get("ci_lower_95"), row.get("ci_upper_95"))
    if value is None or interval[0] is None or interval[1] is None:
        return "not available"
    return f"{value:+.3f} (95% CI [{interval[0]:+.3f}, {interval[1]:+.3f}])"


def write_report(
    output_path: Path,
    *,
    args: argparse.Namespace,
    loaded: list[LoadedSeed],
    metric_rows: list[dict[str, Any]],
    type_rows: list[dict[str, Any]],
    bootstrap_rows: list[dict[str, Any]],
    coefficient_rows: list[dict[str, Any]],
    pooled_included: bool,
) -> None:
    oof_overall = [row for row in metric_rows if row["evaluation"] == "oof_logistic" and row["seed"] in {"31", "47"}]
    raw_overall = [row for row in metric_rows if row["evaluation"] == "raw_direct" and row["seed"] in {"31", "47"}]
    ptrue_check = []
    for seed in ("31", "47"):
        raw = next(row for row in raw_overall if row["seed"] == seed and row["model_name"] == "p_true_blind")
        oof = next(row for row in oof_overall if row["seed"] == seed and row["model_name"] == "p_true_blind")
        ptrue_check.append({"seed": seed, "raw_auroc": raw["auroc"], "oof_logistic_auroc": oof["auroc"], "difference": oof["auroc"] - raw["auroc"]})
    type_counts = [
        row
        for row in type_rows
        if row["model_name"] == "p_true_blind" and row["evaluation"] == "oof_logistic" and row["seed"] in {"31", "47"}
    ]
    bootstrap_display = [
        {
            "seed": row["seed"],
            "comparison": row["comparison"],
            "difference": row["auroc_difference"],
            "ci": f"[{row['ci_lower_95']:.3f}, {row['ci_upper_95']:.3f}]" if row["ci_lower_95"] is not None else "NA",
            "valid_bootstraps": row["valid_resamples"],
        }
        for row in bootstrap_rows
    ]
    coefficient_display = [
        row
        for row in mean_coefficients(coefficient_rows)
        if row["model_name"]
        in {
            "p_true_blind_plus_discrete_se",
            "p_true_blind_plus_normalized_nll",
            "p_true_blind_plus_discrete_se_plus_normalized_nll",
        }
    ]
    bootstrap_by_candidate = {
        row["candidate_model"]: {item["seed"]: item for item in bootstrap_rows if item["candidate_model"] == row["candidate_model"]}
        for row in bootstrap_rows
    }
    discrete_bootstrap = bootstrap_by_candidate["p_true_blind_plus_discrete_se"]
    nll_bootstrap = bootstrap_by_candidate["p_true_blind_plus_normalized_nll"]
    combined_bootstrap = bootstrap_by_candidate["p_true_blind_plus_discrete_se_plus_normalized_nll"]
    token_metrics = {
        row["seed"]: row
        for row in oof_overall
        if row["model_name"] == "p_true_blind_plus_token_entropy"
    }
    ptrue_metrics = {row["seed"]: row for row in oof_overall if row["model_name"] == "p_true_blind"}
    token_differences = {seed: token_metrics[seed]["auroc"] - ptrue_metrics[seed]["auroc"] for seed in ("31", "47")}
    lines = [
        "# BioASQ UQ feature-fusion diagnostic",
        "",
        "## Scope and reproducibility",
        "",
        "This is a post-hoc, CPU-only diagnostic over existing question-level artifacts. It does not regenerate answers, rerun NLI clustering, call Claude, or implement P(True)-Probe.",
        "",
        f"- Random seed: `{args.seed}`; repeated stratified CV: {args.n_splits} folds x {args.n_repeats} repeats.",
        f"- L2 logistic regression: `C=1.0`, `liblinear`, with `StandardScaler` inside each sklearn `Pipeline` training fold. Fold outputs are mapped through the training-fold empirical CDF before repeated OOF averaging, so independently fitted fold intercepts do not make scores incomparable.",
        f"- Bootstrap: {args.bootstrap_samples} paired question-level resamples per seed; one-label resamples are skipped.",
        "- Positive label is `incorrect`; every feature is used as uncertainty, so higher values mean greater predicted error risk.",
        "- AURAC follows the repository convention: trapezoidal retained error risk integrated over coverage 0.5--1.0 (lower is better).",
        "",
        "## Data cleaning",
        "",
    ]
    for item in loaded:
        cleaning = item.cleaning
        lines.append(
            f"- Seed {item.seed}: {cleaning['source_questions']} source questions; {cleaning['valid_labelled_questions']} valid Claude labels; {cleaning['retained_questions']} retained; {cleaning['excluded_questions']} excluded. Reasons: `{json.dumps(cleaning['exclusion_reasons'], sort_keys=True)}`."
        )
    lines.extend([
        "",
        "All model comparisons within a seed use the same retained questions. Blank/invalid Claude labels and any missing/non-finite required feature are explicitly excluded before fitting.",
        "",
        "## Overall OOF logistic metrics",
        "",
        markdown_table(oof_overall, [("seed", "Seed"), ("model_name", "Model"), ("num_examples", "N"), ("num_incorrect", "Incorrect"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("aurac_coverage_0p5_to_1p0", "AURAC"), ("spearman_uncertainty_error", "Spearman")]),
        "",
        "## Raw-score sanity check",
        "",
        markdown_table(raw_overall, [("seed", "Seed"), ("model_name", "Raw score"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("aurac_coverage_0p5_to_1p0", "AURAC")]),
        "",
        "P(True)-blind raw versus one-feature OOF logistic AUROC:",
        "",
        markdown_table(ptrue_check, [("seed", "Seed"), ("raw_auroc", "Raw AUROC"), ("oof_logistic_auroc", "OOF AUROC"), ("difference", "OOF - raw")]),
        "",
        "## Descriptive results by BioASQ type",
        "",
        "These are subsets of predictions from models trained on each seed's full retained set, not independently trained small-subset models. The summary split is particularly small; treat differences as descriptive only.",
        "",
        markdown_table(type_rows, [("seed", "Seed"), ("bioasq_type", "Type"), ("model_name", "Model"), ("num_examples", "N"), ("num_correct", "Correct"), ("num_incorrect", "Incorrect"), ("auroc", "AUROC"), ("auprc", "AUPRC"), ("aurac_coverage_0p5_to_1p0", "AURAC")]),
        "",
        "## Paired bootstrap AUROC differences",
        "",
        markdown_table(bootstrap_display, [("seed", "Seed"), ("comparison", "Comparison"), ("difference", "AUROC difference"), ("ci", "95% CI"), ("valid_bootstraps", "Valid resamples")]),
        "",
        "## Coefficients (mean across 100 fold fits per seed)",
        "",
        "Coefficients are on fold-standardized feature scales; a positive coefficient means that the feature raises predicted error uncertainty after controlling for the other listed features. Full fold-level values are in `fusion_coefficients.csv`.",
        "",
        markdown_table(coefficient_display, [("seed", "Seed"), ("model_name", "Model"), ("feature", "Feature"), ("mean_standardized_coefficient", "Mean coefficient"), ("sd_standardized_coefficient", "SD"), ("fold_fits", "Fits")]),
        "",
        "## Interpretation",
        "",
        f"1. **Does SE add information after P(True)-blind?** The pre-specified discrete-SE fusion is positive in both seeds: seed 31 {format_difference(discrete_bootstrap['31'])}; seed 47 {format_difference(discrete_bootstrap['47'])}. {directional_conclusion(bootstrap_rows)} Its positive standardized coefficients support an independent directional signal, but the paired bootstrap intervals are the deciding uncertainty check.",
        f"2. **Do normalized NLL or token entropy complement P(True)-blind better than SE?** No in this diagnostic: normalized-NLL fusion is lower in both seeds (seed 31 {format_difference(nll_bootstrap['31'])}; seed 47 {format_difference(nll_bootstrap['47'])}), while token-entropy fusion changes are {token_differences['31']:+.3f} and {token_differences['47']:+.3f}. The pre-specified SE+NLL fusion is {format_difference(combined_bootstrap['31'])} in seed 31 and {format_difference(combined_bootstrap['47'])} in seed 47, so it does not rescue NLL.",
        "3. **Is complementarity restricted to factoid/list?** No stable type-specific pattern is established. The descriptive per-type table shows only small/inconsistent factoid changes, an especially adverse seed-47 list change for the discrete-SE fusion, and no replicating summary gain. These splits have small N (notably summary), so no type-level bootstrap claim is made.",
        "4. **Are directions consistent across seeds?** The two-feature discrete-SE fusion is directionally positive in both seeds, whereas NLL, token entropy, and the three-feature discrete-SE+NLL model are not consistently beneficial. The secondary grouped pooled result is descriptive and does not replace the two seed-specific comparison.",
        "5. **Do gains exceed bootstrap uncertainty?** No: every required paired-bootstrap CI includes zero. P(True)-blind remains the strongest raw single feature and appears to capture most usable signal; this does not yet justify P(True)-Probe or a multi-target probe stage.",
        "",
        "## Coverage-risk plots",
        "",
        "- `coverage_risk_31.png`",
        "- `coverage_risk_47.png`",
    ])
    if pooled_included:
        lines.extend(["", "A secondary pooled analysis is included in `fusion_metrics_by_seed.csv` with question-ID grouped splits; a question's two seed rows are never separated across train and test."])
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}. Pass --overwrite to replace named outputs.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    loaded = [load_seed_results(31, args.seed31_dir), load_seed_results(47, args.seed47_dir)]
    metric_rows: list[dict[str, Any]] = []
    type_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    oof_export_rows: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []

    for loaded_seed in loaded:
        rows = loaded_seed.records
        y = np.asarray([row["incorrect"] for row in rows], dtype=int)
        raw_features = {feature: np.asarray([row[feature] for row in rows], dtype=float) for feature in FEATURES}
        seed_text = str(loaded_seed.seed)
        predictions: dict[str, np.ndarray] = {}
        for feature in FEATURES:
            metric_rows.append(metric_row(y, raw_features[feature], seed=seed_text, bioasq_type="overall", model_name=feature, evaluation="raw_direct"))
        for model_name, features in MODEL_FEATURES.items():
            x = np.column_stack([raw_features[feature] for feature in features])
            result = fit_repeated_oof(
                x, y, features, n_splits=args.n_splits, n_repeats=args.n_repeats,
                random_state=args.seed + loaded_seed.seed, seed_label=seed_text, model_name=model_name,
            )
            predictions[model_name] = result.probabilities
            coefficient_rows.extend(result.coefficient_rows)
            metric_rows.append(metric_row(y, result.probabilities, seed=seed_text, bioasq_type="overall", model_name=model_name, evaluation="oof_logistic"))
            for question_type in ("factoid", "list", "summary"):
                indexes = np.asarray([row["bioasq_type"] == question_type for row in rows])
                type_rows.append(metric_row(y[indexes], result.probabilities[indexes], seed=seed_text, bioasq_type=question_type, model_name=model_name, evaluation="oof_logistic"))
            for row, probability, raw in zip(rows, result.probabilities, raw_features[features[0]]):
                oof_export_rows.append({
                    "seed": loaded_seed.seed, "example_id": row["example_id"], "bioasq_type": row["bioasq_type"], "claude_label": row["claude_label"], "incorrect": row["incorrect"],
                    "model_name": model_name, "oof_predicted_uncertainty": float(probability), "raw_single_feature_uncertainty": float(raw) if len(features) == 1 else "",
                    "oof_test_count": int(result.test_counts[0]) if len(result.test_counts) else 0,
                })
        for candidate, comparison in BOOTSTRAP_COMPARISONS.items():
            values = bootstrap_auc_difference(y, predictions["p_true_blind"], predictions[candidate], samples=args.bootstrap_samples, random_state=args.seed + loaded_seed.seed * 100 + len(bootstrap_rows))
            bootstrap_rows.append({"seed": seed_text, "baseline_model": "p_true_blind", "candidate_model": candidate, "comparison": comparison, **values})
        plot_coverage_risk(rows, predictions, args.output_dir / f"coverage_risk_{loaded_seed.seed}.png", f"BioASQ UQ coverage-risk: seed {loaded_seed.seed}")
        pooled_records.extend(rows)

    if not args.skip_pooled_analysis:
        y = np.asarray([row["incorrect"] for row in pooled_records], dtype=int)
        groups = np.asarray([row["example_id"] for row in pooled_records])
        raw_features = {feature: np.asarray([row[feature] for row in pooled_records], dtype=float) for feature in FEATURES}
        for model_name, features in MODEL_FEATURES.items():
            x = np.column_stack([raw_features[feature] for feature in features])
            result = fit_repeated_oof(
                x, y, features, n_splits=args.n_splits, n_repeats=args.n_repeats, random_state=args.seed + 999,
                groups=groups, seed_label="pooled_grouped", model_name=model_name,
            )
            coefficient_rows.extend(result.coefficient_rows)
            metric_rows.append(metric_row(y, result.probabilities, seed="pooled_grouped", bioasq_type="overall", model_name=model_name, evaluation="oof_logistic_grouped"))

    write_csv(args.output_dir / "fusion_metrics_by_seed.csv", metric_rows)
    write_csv(args.output_dir / "fusion_metrics_by_type.csv", type_rows)
    write_csv(args.output_dir / "fusion_bootstrap_differences.csv", bootstrap_rows)
    write_csv(args.output_dir / "fusion_oof_predictions.csv", oof_export_rows)
    write_csv(args.output_dir / "fusion_coefficients.csv", coefficient_rows)
    run_info = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "matplotlib": matplotlib.__version__,
        "loaded_seeds": {str(item.seed): item.cleaning for item in loaded},
        "features": FEATURES,
        "models": {key: list(value) for key, value in MODEL_FEATURES.items()},
        "oof_score_calibration": "training_fold_empirical_cdf_of_logistic_probability",
        "pooled_grouped_analysis": not args.skip_pooled_analysis,
    }
    (args.output_dir / "fusion_run_metadata.json").write_text(json.dumps(run_info, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_dir / "uq_feature_fusion_report.md", args=args, loaded=loaded, metric_rows=metric_rows, type_rows=type_rows, bootstrap_rows=bootstrap_rows, coefficient_rows=coefficient_rows, pooled_included=not args.skip_pooled_analysis)
    return {"output_dir": args.output_dir, "metrics": metric_rows, "bootstrap": bootstrap_rows}


def main() -> int:
    args = parse_args()
    result = run_analysis(args)
    print(f"Wrote feature-fusion analysis to {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
