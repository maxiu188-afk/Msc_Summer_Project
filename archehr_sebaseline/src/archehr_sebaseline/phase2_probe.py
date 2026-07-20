"""Small, leakage-safe utilities for the Phase-2 linear Probe study.

This module deliberately contains no model loading or hidden-state I/O.  It
defines the targets and metrics shared by the command-line runner so that the
protocol-critical thresholding can be unit tested without a GPU or PyTorch.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


ThresholdMethod = Literal["even", "minimum_within_variance"]


@dataclass(frozen=True)
class ThresholdSpec:
    """A scalar train-derived uncertainty threshold and its train partition."""

    method: ThresholdMethod
    threshold: float
    train_low_count: int
    train_high_count: int
    within_group_sse: float | None

    def labels(self, values: np.ndarray) -> np.ndarray:
        """Return 1 for uncertainty at or above the recorded threshold."""

        values = _finite_vector(values, name="values")
        return (values >= self.threshold).astype(np.int64)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).reshape(-1)
    if result.size < 2:
        raise ValueError(f"{name} needs at least two values.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a missing or non-finite value.")
    return result


def _candidate_partitions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sorted values and valid scalar-threshold split locations.

    A split is valid only between distinct adjacent values.  This guarantees
    that the saved threshold can label validation/test examples without a
    train-only tie-breaking rule.
    """

    ordered = np.sort(_finite_vector(values, name="train uncertainty"))
    split_indices = np.flatnonzero(ordered[:-1] < ordered[1:]) + 1
    if split_indices.size == 0:
        raise ValueError("Cannot form a scalar binary threshold from a constant target.")
    thresholds = (ordered[split_indices - 1] + ordered[split_indices]) / 2.0
    return ordered, split_indices, thresholds


def fit_even_threshold(train_uncertainty: np.ndarray) -> ThresholdSpec:
    """Choose the closest feasible scalar threshold to an even train split."""

    ordered, split_indices, thresholds = _candidate_partitions(train_uncertainty)
    middle = ordered.size / 2.0
    distances = np.abs(split_indices - middle)
    candidates = np.flatnonzero(distances == distances.min())
    # Deterministic resolution if two adjacent split ranks are equally close.
    chosen = int(candidates[0])
    split = int(split_indices[chosen])
    return ThresholdSpec(
        method="even",
        threshold=float(thresholds[chosen]),
        train_low_count=split,
        train_high_count=int(ordered.size - split),
        within_group_sse=None,
    )


def fit_minimum_within_variance_threshold(train_uncertainty: np.ndarray) -> ThresholdSpec:
    """Fit the minimum within-group sum-of-squares scalar threshold."""

    ordered, split_indices, thresholds = _candidate_partitions(train_uncertainty)
    cumulative_sum = np.concatenate(([0.0], np.cumsum(ordered)))
    cumulative_square_sum = np.concatenate(([0.0], np.cumsum(ordered * ordered)))

    total_sum = cumulative_sum[-1]
    total_square_sum = cumulative_square_sum[-1]
    costs: list[float] = []
    for split in split_indices:
        low_sum = cumulative_sum[split]
        low_square_sum = cumulative_square_sum[split]
        high_sum = total_sum - low_sum
        high_square_sum = total_square_sum - low_square_sum
        low_sse = low_square_sum - (low_sum * low_sum / split)
        high_count = ordered.size - split
        high_sse = high_square_sum - (high_sum * high_sum / high_count)
        costs.append(float(max(0.0, low_sse + high_sse)))
    chosen = int(np.argmin(np.asarray(costs)))
    split = int(split_indices[chosen])
    return ThresholdSpec(
        method="minimum_within_variance",
        threshold=float(thresholds[chosen]),
        train_low_count=split,
        train_high_count=int(ordered.size - split),
        within_group_sse=costs[chosen],
    )


def fit_threshold(method: ThresholdMethod, train_uncertainty: np.ndarray) -> ThresholdSpec:
    if method == "even":
        return fit_even_threshold(train_uncertainty)
    if method == "minimum_within_variance":
        return fit_minimum_within_variance_threshold(train_uncertainty)
    raise ValueError(f"Unsupported threshold method: {method!r}.")


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """Return standard binary-score metrics, leaving undefined AUCs as null."""

    labels = _finite_vector(labels, name="labels").astype(np.int64)
    scores = _finite_vector(scores, name="scores")
    if labels.shape != scores.shape or not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary labels/scores must have equal length and labels in {0, 1}.")
    has_two_classes = np.unique(labels).size == 2
    return {
        "auroc": float(roc_auc_score(labels, scores)) if has_two_classes else None,
        "average_precision": float(average_precision_score(labels, scores)) if has_two_classes else None,
        "brier": float(brier_score_loss(labels, scores)),
    }


def continuous_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    """Return continuous-target fidelity metrics for P(True)-Probe."""

    target = _finite_vector(target, name="target")
    prediction = _finite_vector(prediction, name="prediction")
    if target.shape != prediction.shape:
        raise ValueError("Continuous target and prediction must have equal length.")
    correlation = (
        spearmanr(target, prediction).statistic
        if np.unique(target).size > 1 and np.unique(prediction).size > 1
        else None
    )
    return {
        "mae": float(mean_absolute_error(target, prediction)),
        "rmse": float(mean_squared_error(target, prediction) ** 0.5),
        "spearman": float(correlation) if correlation is not None and np.isfinite(correlation) else None,
        "r2": float(r2_score(target, prediction)) if np.unique(target).size > 1 else None,
    }
