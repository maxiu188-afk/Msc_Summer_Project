"""Lightweight uncertainty evaluation helpers.

The functions here are dataset-agnostic: callers provide a binary risk label
and one or more uncertainty scores. Higher scores are interpreted as higher
uncertainty/risk.
"""

from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


AUROC_FIELDS = [
    "score_name",
    "num_examples",
    "num_positive",
    "num_negative",
    "auroc",
]

ECE_FIELDS = [
    "score_name",
    "num_examples",
    "num_bins",
    "ece",
    "mean_confidence",
    "accuracy",
]

REJECTION_CURVE_FIELDS = [
    "score_name",
    "rejection_fraction",
    "coverage",
    "num_retained",
    "accuracy",
    "mean_quality_score",
    "threshold",
]

RELIABILITY_BIN_FIELDS = [
    "score_name",
    "bin_index",
    "bin_lower",
    "bin_upper",
    "num_examples",
    "accuracy",
    "mean_confidence",
]

DEFAULT_SCORE_NAMES = [
    "normalized_semantic_entropy",
    "semantic_entropy",
    "num_clusters",
    "normalized_citation_set_entropy",
    "mean_token_entropy",
    "mean_normalized_nll",
]

SCORE_DISPLAY_NAMES = {
    "normalized_semantic_entropy": "Answer SE norm.",
    "semantic_entropy": "Answer SE",
    "num_clusters": "Clusters",
    "normalized_citation_set_entropy": "Citation SE norm.",
    "mean_token_entropy": "Token entropy",
    "mean_normalized_nll": "Norm. NLL",
}

SCORE_COLORS = {
    "normalized_semantic_entropy": "#2f6f9f",
    "semantic_entropy": "#6f4aa8",
    "num_clusters": "#c45a2f",
    "normalized_citation_set_entropy": "#2c8a63",
    "mean_token_entropy": "#9b6b1f",
    "mean_normalized_nll": "#6b7280",
}


def finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def fail_if_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite if replacing it is intended."
        )


def auroc(y_true: list[int], y_score: list[float]) -> float | None:
    """Compute AUROC without sklearn, using average ranks for tied scores."""

    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    n_pos = sum(1 for value in y_true if value == 1)
    n_neg = sum(1 for value in y_true if value == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    rank_sum_pos = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum_pos += average_rank * sum(1 for _, label in pairs[index:end] if label == 1)
        index = end
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def minmax_uncertainties(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _score_values(
    example_rows: list[dict[str, Any]],
    score_name: str,
) -> tuple[list[dict[str, Any]], list[float], list[float]]:
    scored: list[dict[str, Any]] = []
    raw_scores: list[float] = []
    for row in example_rows:
        score = finite_float(row.get(score_name))
        if score is None:
            continue
        scored.append(row)
        raw_scores.append(score)

    if score_name.startswith("normalized_"):
        uncertainties = [_clip01(value) for value in raw_scores]
    else:
        uncertainties = minmax_uncertainties(raw_scores)
    return scored, raw_scores, uncertainties


def build_auroc_rows(
    example_rows: list[dict[str, Any]],
    *,
    score_names: list[str] | None = None,
    risk_field: str = "is_low_quality",
) -> list[dict[str, str]]:
    score_names = score_names or DEFAULT_SCORE_NAMES
    rows: list[dict[str, str]] = []
    for score_name in score_names:
        scored, raw_scores, _ = _score_values(example_rows, score_name)
        labels = [1 if str(row.get(risk_field)).lower() == "true" else 0 for row in scored]
        value = auroc(labels, raw_scores) if labels else None
        row = {
            "score_name": score_name,
            "num_examples": len(labels),
            "num_positive": sum(labels),
            "num_negative": len(labels) - sum(labels),
            "auroc": value,
        }
        rows.append({field: fmt(row.get(field)) for field in AUROC_FIELDS})
    return rows


def build_ece_rows(
    example_rows: list[dict[str, Any]],
    *,
    score_names: list[str] | None = None,
    risk_field: str = "is_low_quality",
    num_bins: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Build ECE and reliability-bin rows.

    Uncertainty scores are mapped to confidence as ``1 - normalized_uncertainty``.
    Correctness is ``not risk_field``.
    """

    score_names = score_names or DEFAULT_SCORE_NAMES
    ece_rows: list[dict[str, str]] = []
    bin_rows: list[dict[str, str]] = []
    for score_name in score_names:
        scored, _, uncertainties = _score_values(example_rows, score_name)
        if not scored:
            row = {
                "score_name": score_name,
                "num_examples": 0,
                "num_bins": num_bins,
                "ece": None,
                "mean_confidence": None,
                "accuracy": None,
            }
            ece_rows.append({field: fmt(row.get(field)) for field in ECE_FIELDS})
            continue

        confidences = [1.0 - value for value in uncertainties]
        correct = [0 if str(row.get(risk_field)).lower() == "true" else 1 for row in scored]
        total = len(scored)
        ece = 0.0
        for bin_index in range(num_bins):
            lower = bin_index / num_bins
            upper = (bin_index + 1) / num_bins
            indexes = []
            for index, confidence in enumerate(confidences):
                if bin_index == num_bins - 1:
                    in_bin = lower <= confidence <= upper
                else:
                    in_bin = lower <= confidence < upper
                if in_bin:
                    indexes.append(index)
            if not indexes:
                accuracy = None
                mean_confidence = None
            else:
                accuracy = sum(correct[index] for index in indexes) / len(indexes)
                mean_confidence = sum(confidences[index] for index in indexes) / len(indexes)
                ece += len(indexes) / total * abs(accuracy - mean_confidence)
            bin_row = {
                "score_name": score_name,
                "bin_index": bin_index,
                "bin_lower": lower,
                "bin_upper": upper,
                "num_examples": len(indexes),
                "accuracy": accuracy,
                "mean_confidence": mean_confidence,
            }
            bin_rows.append({field: fmt(bin_row.get(field)) for field in RELIABILITY_BIN_FIELDS})

        row = {
            "score_name": score_name,
            "num_examples": total,
            "num_bins": num_bins,
            "ece": ece,
            "mean_confidence": sum(confidences) / total,
            "accuracy": sum(correct) / total,
        }
        ece_rows.append({field: fmt(row.get(field)) for field in ECE_FIELDS})
    return ece_rows, bin_rows


def build_rejection_curve_rows(
    example_rows: list[dict[str, Any]],
    *,
    score_names: list[str] | None = None,
    risk_field: str = "is_low_quality",
    quality_field: str = "quality_score",
    max_rejection_fraction: float = 0.5,
    step: float = 0.05,
) -> list[dict[str, str]]:
    score_names = score_names or DEFAULT_SCORE_NAMES
    fractions = []
    current = 0.0
    while current <= max_rejection_fraction + 1e-12:
        fractions.append(round(current, 10))
        current += step

    rows: list[dict[str, str]] = []
    for score_name in score_names:
        scored = []
        for row in example_rows:
            score = finite_float(row.get(score_name))
            if score is None:
                continue
            is_risk = str(row.get(risk_field)).lower() == "true"
            quality = finite_float(row.get(quality_field))
            scored.append((score, is_risk, quality))
        scored.sort(key=lambda item: item[0])
        total = len(scored)
        if total == 0:
            continue
        for fraction in fractions:
            num_retained = max(1, int(math.ceil(total * (1.0 - fraction))))
            retained = scored[:num_retained]
            correct_count = sum(1 for _, is_risk, _ in retained if not is_risk)
            quality_values = [quality for _, _, quality in retained if quality is not None]
            row = {
                "score_name": score_name,
                "rejection_fraction": fraction,
                "coverage": num_retained / total,
                "num_retained": num_retained,
                "accuracy": correct_count / num_retained,
                "mean_quality_score": (
                    sum(quality_values) / len(quality_values) if quality_values else None
                ),
                "threshold": retained[-1][0],
            }
            rows.append({field: fmt(row.get(field)) for field in REJECTION_CURVE_FIELDS})
    return rows


def _plot_points(
    values: list[tuple[float, float]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
) -> str:
    points = []
    for x_value, y_value in values:
        x = x0 + x_value * width
        y = y0 + (1.0 - y_value) * height
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def write_auroc_bar_svg(
    auroc_rows: list[dict[str, str]],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_if_exists(output_path, overwrite)

    values = {
        row["score_name"]: finite_float(row.get("auroc"))
        for row in auroc_rows
        if row.get("score_name")
    }
    score_names = [name for name in DEFAULT_SCORE_NAMES if name in values]
    width, height = 820, 430
    left, top, plot_width, plot_height = 76, 58, 650, 270
    bar_gap = 16
    bar_width = (plot_width - bar_gap * max(0, len(score_names) - 1)) / max(len(score_names), 1)
    baseline_y = top + 0.5 * plot_height

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="76" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">AUROC for Low-Quality Answer Detection</text>',
        '<text x="76" y="50" font-family="Arial, sans-serif" font-size="12" fill="#5f6368">Higher means the uncertainty score ranks weak answers above strong answers.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" y2="{baseline_y:.2f}" stroke="#b3261e" stroke-width="1" stroke-dasharray="4 4"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + (1.0 - tick) * plot_height
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5f6368">{tick:.2f}</text>'
        )
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#5f6368" stroke-width="1"/>'
        )

    for index, score_name in enumerate(score_names):
        value = values.get(score_name)
        bar_value = max(0.0, min(1.0, value if value is not None else 0.0))
        x = left + index * (bar_width + bar_gap)
        y = top + (1.0 - bar_value) * plot_height
        label = html.escape(SCORE_DISPLAY_NAMES.get(score_name, score_name))
        color = SCORE_COLORS.get(score_name, "#4a5568")
        display_value = "" if value is None else f"{value:.3f}"
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{plot_height * bar_value:.2f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#202124">{display_value}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#202124">{label}</text>',
            ]
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_rejection_curve_svg(
    rejection_rows: list[dict[str, str]],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_if_exists(output_path, overwrite)

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rejection_rows:
        score_name = str(row.get("score_name") or "")
        rejection_fraction = finite_float(row.get("rejection_fraction"))
        accuracy = finite_float(row.get("accuracy"))
        if score_name and rejection_fraction is not None and accuracy is not None:
            grouped[score_name].append((rejection_fraction, accuracy))

    width, height = 820, 500
    left, top, plot_width, plot_height = 78, 58, 570, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="78" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">Rejection Curve</text>',
        '<text x="78" y="50" font-family="Arial, sans-serif" font-size="12" fill="#5f6368">Reject high-uncertainty examples first; higher retained accuracy is better.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_width
        y = top + (1.0 - tick) * plot_height
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 4}" stroke="#5f6368" stroke-width="1"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#5f6368">{tick:.2f}</text>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5f6368">{tick:.2f}</text>',
                f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#5f6368" stroke-width="1"/>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Rejection fraction</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Retained accuracy</text>',
        ]
    )

    legend_x = left + plot_width + 34
    legend_y = top + 22
    legend_index = 0
    for score_name in DEFAULT_SCORE_NAMES:
        values = sorted(grouped.get(score_name, []))
        if not values:
            continue
        color = SCORE_COLORS.get(score_name, "#4a5568")
        points = _plot_points(values, x0=left, y0=top, width=plot_width, height=plot_height)
        label = html.escape(SCORE_DISPLAY_NAMES.get(score_name, score_name))
        y = legend_y + legend_index * 28
        legend_index += 1
        parts.extend(
            [
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>',
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#202124">{label}</text>',
            ]
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_reliability_svg(
    reliability_rows: list[dict[str, str]],
    path: str | Path,
    *,
    score_name: str = "normalized_semantic_entropy",
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_if_exists(output_path, overwrite)

    rows = [row for row in reliability_rows if row.get("score_name") == score_name]
    width, height = 560, 420
    left, top, plot_width, plot_height = 70, 54, 360, 280
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="70" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">Reliability: {html.escape(SCORE_DISPLAY_NAMES.get(score_name, score_name))}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top}" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="4 4"/>',
    ]
    for row in rows:
        count = int(finite_float(row.get("num_examples")) or 0)
        if count == 0:
            continue
        lower = finite_float(row.get("bin_lower")) or 0.0
        upper = finite_float(row.get("bin_upper")) or 0.0
        accuracy = finite_float(row.get("accuracy"))
        if accuracy is None:
            continue
        x = left + lower * plot_width
        bar_width = max(1.0, (upper - lower) * plot_width - 2)
        y = top + (1.0 - accuracy) * plot_height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{plot_height * accuracy:.2f}" fill="#2f6f9f" opacity="0.82"/>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 34}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Confidence bin</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Accuracy</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")
