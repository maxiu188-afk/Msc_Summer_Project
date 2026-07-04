"""PubMedQA yes/no/maybe label evaluation for Level 4 artifacts."""

from __future__ import annotations

import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..data_io import read_jsonl, write_csv


PUBMEDQA_LABELS = ("yes", "no", "maybe")
UNKNOWN_LABEL = "unknown"

LABEL_PREDICTION_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "sample_id",
    "gold_label",
    "predicted_label",
    "sample_correct",
    "majority_label",
    "majority_correct",
    "majority_count",
    "known_vote_count",
    "total_vote_count",
    "is_tie",
    "clean_answer",
]

REJECTION_CURVE_FIELDS = [
    "score_name",
    "rejection_fraction",
    "coverage",
    "num_retained",
    "accuracy",
    "correct_count",
    "total_count",
    "threshold",
]

DEFAULT_UNCERTAINTY_FIELDS = [
    "normalized_discrete_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "mean_normalized_nll",
    "mean_token_entropy",
]

SCORE_DISPLAY_NAMES = {
    "normalized_discrete_semantic_entropy": "Discrete SE",
    "normalized_likelihood_weighted_semantic_entropy": "Weighted SE",
    "predictive_entropy": "Predictive Ent.",
    "mean_normalized_nll": "Norm. NLL",
    "mean_token_entropy": "Token Ent.",
}

SCORE_COLORS = {
    "normalized_discrete_semantic_entropy": "#2f6f9f",
    "normalized_likelihood_weighted_semantic_entropy": "#6f4aa8",
    "predictive_entropy": "#c45a2f",
    "mean_normalized_nll": "#2c8a63",
    "mean_token_entropy": "#9b6b1f",
}


_BRACKET_LABEL_RE = re.compile(r"\[(yes|no|maybe)\]", re.IGNORECASE)
_EXPLICIT_LABEL_RE = re.compile(
    r"\b(?:answer|label|decision)\s*(?:is|was)?\s*(?::|=)?\s*(?:\[)?(yes|no|maybe)\b",
    re.IGNORECASE,
)
_LEADING_LABEL_RE = re.compile(r"^\s*(?:\[)?(yes|no|maybe)(?:\])?[\s,.:;\-]", re.IGNORECASE)
_CONCLUSION_LABEL_RE = re.compile(
    r"\b(?:therefore|thus|overall|so|in conclusion)\b[^.\n]{0,120}\b(yes|no|maybe)\b",
    re.IGNORECASE,
)


def normalize_pubmedqa_label(label: Any) -> str:
    normalized = str(label or "").strip().lower()
    return normalized if normalized in PUBMEDQA_LABELS else UNKNOWN_LABEL


def extract_pubmedqa_label(answer: str) -> str:
    """Extract a conservative PubMedQA decision label from generated text."""

    text = str(answer or "")
    signals: list[tuple[int, str]] = []
    for pattern in (
        _BRACKET_LABEL_RE,
        _EXPLICIT_LABEL_RE,
        _LEADING_LABEL_RE,
        _CONCLUSION_LABEL_RE,
    ):
        for match in pattern.finditer(text):
            signals.append((match.start(), match.group(1).lower()))
    if not signals:
        return UNKNOWN_LABEL
    return sorted(signals, key=lambda item: item[0])[-1][1]


def majority_vote(labels: list[str]) -> tuple[str, int, int, bool]:
    """Return label, winning count, known vote count, and whether a tie occurred."""

    known_labels = [label for label in labels if label in PUBMEDQA_LABELS]
    if not known_labels:
        return UNKNOWN_LABEL, 0, 0, False
    counts = Counter(known_labels)
    max_count = max(counts.values())
    winners = [label for label in PUBMEDQA_LABELS if counts[label] == max_count]
    is_tie = len(winners) > 1
    return (UNKNOWN_LABEL if is_tie else winners[0]), max_count, len(known_labels), is_tie


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


def _fail_if_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite if replacing it is intended."
        )


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def build_label_prediction_rows(
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    examples_by_id = {str(example["id"]): example for example in examples}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)

    prediction_rows: list[dict[str, str]] = []
    example_results: list[dict[str, Any]] = []
    for example_id in sorted(grouped):
        example = examples_by_id.get(example_id, {})
        gold_label = normalize_pubmedqa_label(example.get("label"))
        records = sorted(grouped[example_id], key=lambda item: int(item.get("sample_id", 0)))
        predicted_labels = [
            extract_pubmedqa_label(str(record.get("clean_answer") or record.get("raw_answer") or ""))
            for record in records
        ]
        majority_label, majority_count, known_count, is_tie = majority_vote(predicted_labels)
        majority_correct = gold_label in PUBMEDQA_LABELS and majority_label == gold_label
        example_results.append(
            {
                "example_id": example_id,
                "dataset": example.get("dataset") or records[0].get("dataset"),
                "split": example.get("split") or records[0].get("split"),
                "gold_label": gold_label,
                "predicted_labels": predicted_labels,
                "majority_label": majority_label,
                "majority_correct": majority_correct,
                "majority_count": majority_count,
                "known_vote_count": known_count,
                "total_vote_count": len(records),
                "is_tie": is_tie,
            }
        )

        for record, predicted_label in zip(records, predicted_labels):
            sample_correct = gold_label in PUBMEDQA_LABELS and predicted_label == gold_label
            row = {
                "example_id": example_id,
                "dataset": example.get("dataset") or record.get("dataset"),
                "split": example.get("split") or record.get("split"),
                "sample_id": record.get("sample_id"),
                "gold_label": gold_label,
                "predicted_label": predicted_label,
                "sample_correct": sample_correct,
                "majority_label": majority_label,
                "majority_correct": majority_correct,
                "majority_count": majority_count,
                "known_vote_count": known_count,
                "total_vote_count": len(records),
                "is_tie": is_tie,
                "clean_answer": record.get("clean_answer") or "",
            }
            prediction_rows.append({field: _fmt(row.get(field)) for field in LABEL_PREDICTION_FIELDS})

    return prediction_rows, example_results


def _score_table(score_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row["example_id"]): row for row in score_rows}


def compute_auroc_table(
    example_results: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    *,
    score_names: list[str] | None = None,
) -> dict[str, float | None]:
    score_names = score_names or DEFAULT_UNCERTAINTY_FIELDS
    scores_by_id = _score_table(score_rows)
    output: dict[str, float | None] = {}
    for score_name in score_names:
        y_true: list[int] = []
        y_score: list[float] = []
        for result in example_results:
            score = finite_float(scores_by_id.get(result["example_id"], {}).get(score_name))
            if score is None:
                continue
            y_true.append(0 if result["majority_correct"] else 1)
            y_score.append(score)
        output[score_name] = auroc(y_true, y_score) if y_true else None
    return output


def build_rejection_curve_rows(
    example_results: list[dict[str, Any]],
    score_rows: list[dict[str, str]],
    *,
    score_names: list[str] | None = None,
) -> list[dict[str, str]]:
    score_names = score_names or DEFAULT_UNCERTAINTY_FIELDS
    scores_by_id = _score_table(score_rows)
    rows: list[dict[str, str]] = []
    for score_name in score_names:
        scored = []
        for result in example_results:
            score = finite_float(scores_by_id.get(result["example_id"], {}).get(score_name))
            if score is None:
                continue
            scored.append((score, bool(result["majority_correct"])))
        scored.sort(key=lambda item: item[0])
        total = len(scored)
        if total == 0:
            continue
        for rejected in range(total):
            retained = scored[: total - rejected]
            correct_count = sum(1 for _, correct in retained if correct)
            num_retained = len(retained)
            row = {
                "score_name": score_name,
                "rejection_fraction": rejected / total,
                "coverage": num_retained / total,
                "num_retained": num_retained,
                "accuracy": correct_count / num_retained,
                "correct_count": correct_count,
                "total_count": total,
                "threshold": retained[-1][0],
            }
            rows.append({field: _fmt(row.get(field)) for field in REJECTION_CURVE_FIELDS})
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
    aurocs: dict[str, float | None],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write a small dependency-free SVG bar chart for AUROC results."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fail_if_exists(output_path, overwrite)

    width, height = 760, 430
    left, top, plot_width, plot_height = 76, 54, 620, 270
    baseline_y = top + (1.0 - 0.5) * plot_height
    bar_gap = 22
    score_names = [name for name in DEFAULT_UNCERTAINTY_FIELDS if name in aurocs]
    bar_width = (plot_width - bar_gap * (len(score_names) - 1)) / max(len(score_names), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="76" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">AUROC for Detecting Incorrect Majority Answers</text>',
        '<text x="76" y="50" font-family="Arial, sans-serif" font-size="12" fill="#5f6368">Higher is better; 0.5 is random ranking.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{baseline_y:.2f}" x2="{left + plot_width}" y2="{baseline_y:.2f}" stroke="#b3261e" stroke-width="1" stroke-dasharray="4 4"/>',
        f'<text x="{left + plot_width + 8}" y="{baseline_y + 4:.2f}" font-family="Arial, sans-serif" font-size="11" fill="#b3261e">0.50</text>',
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
        value = aurocs.get(score_name)
        bar_value = max(0.0, min(1.0, value if value is not None else 0.0))
        x = left + index * (bar_width + bar_gap)
        y = top + (1.0 - bar_value) * plot_height
        label = html.escape(SCORE_DISPLAY_NAMES.get(score_name, score_name))
        color = SCORE_COLORS.get(score_name, "#4a5568")
        parts.extend(
            [
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{plot_height * bar_value:.2f}" fill="{color}"/>',
                f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#202124">{bar_value:.3f}</text>',
                f'<text x="{x + bar_width / 2:.2f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#202124">{label}</text>',
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
    """Write a dependency-free SVG selective-prediction curve."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fail_if_exists(output_path, overwrite)

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rejection_rows:
        rejection_fraction = finite_float(row.get("rejection_fraction"))
        accuracy = finite_float(row.get("accuracy"))
        score_name = str(row.get("score_name") or "")
        if rejection_fraction is None or accuracy is None or not score_name:
            continue
        grouped[score_name].append((rejection_fraction, accuracy))

    width, height = 820, 500
    left, top, plot_width, plot_height = 78, 58, 570, 340
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="78" y="30" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#202124">Selective Prediction Curve</text>',
        '<text x="78" y="50" font-family="Arial, sans-serif" font-size="12" fill="#5f6368">Reject most uncertain examples first; higher retained accuracy is better.</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#5f6368" stroke-width="1"/>',
    ]
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        x = left + tick * plot_width
        y = top + (1.0 - tick) * plot_height
        parts.append(
            f'<line x1="{x:.2f}" y1="{top + plot_height}" x2="{x:.2f}" y2="{top + plot_height + 4}" stroke="#5f6368" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#5f6368">{tick:.2f}</text>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#5f6368">{tick:.2f}</text>'
        )
        parts.append(
            f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#5f6368" stroke-width="1"/>'
        )
    parts.extend(
        [
            f'<text x="{left + plot_width / 2:.2f}" y="{height - 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Rejection fraction</text>',
            f'<text x="20" y="{top + plot_height / 2:.2f}" transform="rotate(-90 20 {top + plot_height / 2:.2f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#202124">Retained accuracy</text>',
        ]
    )

    legend_x = left + plot_width + 34
    legend_y = top + 22
    for index, score_name in enumerate(DEFAULT_UNCERTAINTY_FIELDS):
        values = sorted(grouped.get(score_name, []))
        if not values:
            continue
        color = SCORE_COLORS.get(score_name, "#4a5568")
        points = _plot_points(values, x0=left, y0=top, width=plot_width, height=plot_height)
        label = html.escape(SCORE_DISPLAY_NAMES.get(score_name, score_name))
        y = legend_y + index * 28
        parts.extend(
            [
                f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>',
                f'<circle cx="{left + values[0][0] * plot_width:.2f}" cy="{top + (1.0 - values[0][1]) * plot_height:.2f}" r="3" fill="{color}"/>',
                f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 22}" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 30}" y="{y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#202124">{label}</text>',
            ]
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def summarize_label_evaluation(
    example_results: list[dict[str, Any]],
    prediction_rows: list[dict[str, str]],
    aurocs: dict[str, float | None],
) -> dict[str, Any]:
    total_examples = len(example_results)
    majority_correct = sum(1 for result in example_results if result["majority_correct"])
    valid_sample_rows = [
        row for row in prediction_rows
        if row["gold_label"] in PUBMEDQA_LABELS and row["predicted_label"] in PUBMEDQA_LABELS
    ]
    known_correct = sum(1 for row in valid_sample_rows if row["sample_correct"] == "true")
    unknown_samples = sum(1 for row in prediction_rows if row["predicted_label"] == UNKNOWN_LABEL)
    label_counts = Counter(result["gold_label"] for result in example_results)
    majority_counts = Counter(result["majority_label"] for result in example_results)

    return {
        "dataset": "pubmedqa",
        "num_examples": total_examples,
        "num_generations": len(prediction_rows),
        "gold_label_counts": dict(sorted(label_counts.items())),
        "majority_label_counts": dict(sorted(majority_counts.items())),
        "majority_correct": majority_correct,
        "majority_accuracy": majority_correct / total_examples if total_examples else None,
        "majority_unknown": sum(1 for result in example_results if result["majority_label"] == UNKNOWN_LABEL),
        "majority_ties": sum(1 for result in example_results if result["is_tie"]),
        "known_sample_predictions": len(valid_sample_rows),
        "unknown_sample_predictions": unknown_samples,
        "per_sample_accuracy_known": (
            known_correct / len(valid_sample_rows) if valid_sample_rows else None
        ),
        "auroc_incorrect_by_score": aurocs,
    }


def evaluate_level4_pubmedqa_labels(
    output_dir: str | Path,
    *,
    predictions_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    rejection_curve_path: str | Path | None = None,
    auroc_plot_path: str | Path | None = None,
    rejection_plot_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate PubMedQA yes/no/maybe labels from a Level 4 output directory."""

    output_path = Path(output_dir)
    predictions_path = predictions_path or output_path / "pubmedqa_label_predictions.csv"
    summary_path = summary_path or output_path / "pubmedqa_eval_summary.json"
    rejection_curve_path = rejection_curve_path or output_path / "rejection_curve.csv"
    auroc_plot_path = auroc_plot_path or output_path / "auroc_bar.svg"
    rejection_plot_path = rejection_plot_path or output_path / "rejection_curve.svg"

    examples = read_jsonl(output_path / "examples.jsonl")
    generations = read_jsonl(output_path / "cleaned_generations.jsonl")
    score_rows = read_csv_rows(output_path / "se_scores.csv")

    prediction_rows, example_results = build_label_prediction_rows(examples, generations)
    aurocs = compute_auroc_table(example_results, score_rows)
    rejection_rows = build_rejection_curve_rows(example_results, score_rows)
    summary = summarize_label_evaluation(example_results, prediction_rows, aurocs)

    write_csv(prediction_rows, predictions_path, LABEL_PREDICTION_FIELDS, overwrite=overwrite)
    write_csv(rejection_rows, rejection_curve_path, REJECTION_CURVE_FIELDS, overwrite=overwrite)
    write_auroc_bar_svg(aurocs, auroc_plot_path, overwrite=overwrite)
    write_rejection_curve_svg(rejection_rows, rejection_plot_path, overwrite=overwrite)

    summary_output_path = Path(summary_path)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{summary_output_path} already exists. Pass --overwrite if replacing it is intended."
        )
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "prediction_rows": prediction_rows,
        "rejection_curve_rows": rejection_rows,
        "paths": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "rejection_curve": str(rejection_curve_path),
            "auroc_plot": str(auroc_plot_path),
            "rejection_plot": str(rejection_plot_path),
        },
    }
