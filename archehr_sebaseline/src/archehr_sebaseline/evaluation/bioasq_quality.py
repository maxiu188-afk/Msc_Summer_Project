"""Lightweight BioASQ Task B quality evaluation for Level 4 outputs.

This module is deliberately a transparent approximation, not a replacement for
the official BioASQ evaluation service.  It uses the same *families* of
automatic metrics where practical: ROUGE-2 and ROUGE-SU4 for ideal (summary)
answers, accuracy for yes/no, exact-answer matching for factoids, and set
precision/recall/F1 for lists.  The local implementation has no stemming,
synonym resource, manual readability assessment, or official ranked-answer
format; those limitations are recorded in the output report.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from ..data_io import read_jsonl, write_csv
from .uncertainty_metrics import (
    AUROC_FIELDS,
    REJECTION_CURVE_FIELDS,
    auroc,
    fail_if_exists,
    finite_float,
    fmt,
)


DEFAULT_QUALITY_THRESHOLD = 0.15
DEFAULT_UNCERTAINTY_FIELDS = [
    "normalized_discrete_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "mean_normalized_nll",
    "mean_token_entropy",
]

GENERATION_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "bioasq_type",
    "sample_id",
    "quality_metric",
    "quality_score",
    "rouge_2_f1",
    "rouge_su4_f1",
    "yesno_accuracy",
    "factoid_strict_accuracy",
    "factoid_lenient_accuracy",
    "list_precision",
    "list_recall",
    "list_f1",
    "has_snippet_citation",
    "reference_count",
    "answer_text",
]

EXAMPLE_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "bioasq_type",
    "num_generations",
    "quality_target",
    "quality_threshold",
    "quality_metric",
    "quality_score",
    "sample0_quality_score",
    "mean_quality_score",
    "best_quality_score",
    "is_low_quality",
    "quality_band",
    "mean_rouge_2_f1",
    "mean_rouge_su4_f1",
    "mean_yesno_accuracy",
    "mean_factoid_lenient_accuracy",
    "mean_list_f1",
    "snippet_citation_rate",
    "reference_count",
]

SE_EVAL_FIELDS = [
    *EXAMPLE_QUALITY_FIELDS,
    "discrete_semantic_entropy",
    "normalized_discrete_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "num_clusters",
    "mean_normalized_nll",
    "mean_token_entropy",
]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CITATION_RE = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]", re.IGNORECASE)
_YESNO_RE = re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?(yes|no)\b", re.IGNORECASE)


def _answer_text(record: dict[str, Any]) -> str:
    return str(record.get("clean_answer") or record.get("raw_answer") or "").strip()


def strip_snippet_citations(text: str) -> str:
    """Remove the project's ``[S1]``-style citations before lexical scoring."""

    return _CITATION_RE.sub(" ", str(text or ""))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(strip_snippet_citations(text).lower())


def _counter_prf(predicted: Counter[Any], reference: Counter[Any]) -> tuple[float, float, float]:
    predicted_total = sum(predicted.values())
    reference_total = sum(reference.values())
    if not predicted_total and not reference_total:
        return 1.0, 1.0, 1.0
    if not predicted_total or not reference_total:
        return 0.0, 0.0, 0.0
    overlap = sum((predicted & reference).values())
    precision = overlap / predicted_total
    recall = overlap / reference_total
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_n_f1(prediction: str, reference: str, *, n: int = 2) -> float:
    """Unstemmed ROUGE-N F1 approximation with multiset n-gram matching."""

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    predicted = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
    gold = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
    return _counter_prf(predicted, gold)[2]


def _su4_units(tokens: list[str]) -> Counter[tuple[str, ...]]:
    """Return unigrams plus skip-bigrams with at most four skipped positions."""

    units: Counter[tuple[str, ...]] = Counter((token,) for token in tokens)
    for start, left in enumerate(tokens):
        for end in range(start + 1, min(len(tokens), start + 6)):
            units[(left, tokens[end])] += 1
    return units


def rouge_su4_f1(prediction: str, reference: str) -> float:
    """Unstemmed ROUGE-SU4 F1 approximation used for BioASQ ideal answers."""

    return _counter_prf(_su4_units(tokenize(prediction)), _su4_units(tokenize(reference)))[2]


def _best_reference_score(
    prediction: str,
    references: list[str],
    scorer: Callable[[str, str], float],
) -> float | None:
    return max((scorer(prediction, reference) for reference in references), default=None)


def _normalize_entity(text: str) -> str:
    return " ".join(tokenize(text))


def extract_yesno(answer: str) -> str:
    match = _YESNO_RE.match(strip_snippet_citations(answer))
    return match.group(1).lower() if match else "unknown"


def _first_phrase(text: str) -> str:
    cleaned = strip_snippet_citations(text).strip()
    return re.split(r"[.\n;:]", cleaned, maxsplit=1)[0].strip()


def _contains_alias(answer: str, aliases: set[str]) -> bool:
    normalized_answer = _normalize_entity(answer)
    return any(alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_answer) for alias in aliases)


def _list_items(answer: str) -> set[str]:
    first_line = strip_snippet_citations(answer).split("\n", maxsplit=1)[0]
    pieces = re.split(r"[,;•]|\band\b", first_line, flags=re.IGNORECASE)
    return {normalized for piece in pieces if (normalized := _normalize_entity(piece))}


def _set_prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _quality_band(score: float, *, quality_threshold: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= quality_threshold:
        return "medium"
    return "low"


def evaluate_generation_quality(generation: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    """Score one generated answer using the BioASQ type preserved in examples."""

    answer = _answer_text(generation)
    question_type = str(example.get("bioasq_type") or "summary").lower()
    ideal_answers = [str(item).strip() for item in example.get("ideal_answers") or [] if str(item).strip()]
    exact_answers = [str(item).strip() for item in example.get("exact_answers") or [] if str(item).strip()]
    base = {
        "example_id": generation.get("example_id"),
        "dataset": generation.get("dataset") or example.get("dataset"),
        "split": generation.get("split") or example.get("split"),
        "bioasq_type": question_type,
        "sample_id": generation.get("sample_id"),
        "rouge_2_f1": None,
        "rouge_su4_f1": None,
        "yesno_accuracy": None,
        "factoid_strict_accuracy": None,
        "factoid_lenient_accuracy": None,
        "list_precision": None,
        "list_recall": None,
        "list_f1": None,
        "has_snippet_citation": bool(_CITATION_RE.search(answer)),
        "reference_count": len(ideal_answers) if question_type == "summary" else len(exact_answers),
        "answer_text": answer,
    }
    if question_type == "summary":
        rouge2 = _best_reference_score(answer, ideal_answers, rouge_n_f1)
        su4 = _best_reference_score(answer, ideal_answers, rouge_su4_f1)
        quality = (rouge2 + su4) / 2 if rouge2 is not None and su4 is not None else 0.0
        return {**base, "quality_metric": "mean_rouge2_su4_f1", "quality_score": quality, "rouge_2_f1": rouge2, "rouge_su4_f1": su4}
    if question_type == "yesno":
        gold = _normalize_entity(exact_answers[0]) if exact_answers else ""
        accuracy = float(extract_yesno(answer) == gold) if gold else 0.0
        return {**base, "quality_metric": "yesno_accuracy", "quality_score": accuracy, "yesno_accuracy": accuracy}
    if question_type == "factoid":
        aliases = {_normalize_entity(item) for item in exact_answers if _normalize_entity(item)}
        strict = float(_normalize_entity(_first_phrase(answer)) in aliases) if aliases else 0.0
        lenient = float(_contains_alias(answer, aliases)) if aliases else 0.0
        return {**base, "quality_metric": "factoid_lenient_accuracy", "quality_score": lenient, "factoid_strict_accuracy": strict, "factoid_lenient_accuracy": lenient}
    if question_type == "list":
        gold = {_normalize_entity(item) for item in exact_answers if _normalize_entity(item)}
        precision, recall, f1 = _set_prf(_list_items(answer), gold)
        return {**base, "quality_metric": "list_f1", "quality_score": f1, "list_precision": precision, "list_recall": recall, "list_f1": f1}
    return {**base, "quality_metric": "unsupported", "quality_score": 0.0}


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for value in (finite_float(row.get(field)) for row in rows) if value is not None]
    return sum(values) / len(values) if values else None


def build_quality_rows(
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_target: str = "mean",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    examples_by_id = {str(example["id"]): example for example in examples}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)
    generation_rows: list[dict[str, str]] = []
    example_rows: list[dict[str, str]] = []
    for example_id in sorted(grouped):
        example = examples_by_id.get(example_id, {"id": example_id})
        raw_rows = [evaluate_generation_quality(record, example) for record in sorted(grouped[example_id], key=lambda item: int(item.get("sample_id", 0)))]
        generation_rows.extend({field: fmt(row.get(field)) for field in GENERATION_QUALITY_FIELDS} for row in raw_rows)
        values = [float(row["quality_score"]) for row in raw_rows]
        target = sum(values) / len(values) if quality_target == "mean" else values[0]
        row = {
            "example_id": example_id,
            "dataset": example.get("dataset"),
            "split": example.get("split"),
            "bioasq_type": example.get("bioasq_type"),
            "num_generations": len(raw_rows),
            "quality_target": quality_target,
            "quality_threshold": quality_threshold,
            "quality_metric": raw_rows[0]["quality_metric"],
            "quality_score": target,
            "sample0_quality_score": values[0],
            "mean_quality_score": sum(values) / len(values),
            "best_quality_score": max(values),
            "is_low_quality": target < quality_threshold,
            "quality_band": _quality_band(target, quality_threshold=quality_threshold),
            "mean_rouge_2_f1": _mean(raw_rows, "rouge_2_f1"),
            "mean_rouge_su4_f1": _mean(raw_rows, "rouge_su4_f1"),
            "mean_yesno_accuracy": _mean(raw_rows, "yesno_accuracy"),
            "mean_factoid_lenient_accuracy": _mean(raw_rows, "factoid_lenient_accuracy"),
            "mean_list_f1": _mean(raw_rows, "list_f1"),
            "snippet_citation_rate": sum(bool(record["has_snippet_citation"]) for record in raw_rows) / len(raw_rows),
            "reference_count": raw_rows[0]["reference_count"],
        }
        example_rows.append({field: fmt(row.get(field)) for field in EXAMPLE_QUALITY_FIELDS})
    return generation_rows, example_rows


def _merge_se_rows(
    example_rows: list[dict[str, str]], score_rows: list[dict[str, str]], generation_uq_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    scores_by_id = {str(row.get("example_id")): row for row in score_rows}
    grouped_uq: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in generation_uq_rows:
        grouped_uq[str(row.get("example_id"))].append(row)
    output = []
    for row in example_rows:
        example_id = str(row["example_id"])
        merged = {**row, **scores_by_id.get(example_id, {})}
        for output_field, generation_field in (
            ("mean_normalized_nll", "normalized_nll"),
            ("mean_token_entropy", "mean_token_entropy"),
        ):
            values = [
                value
                for value in (
                    finite_float(item.get(generation_field))
                    for item in grouped_uq.get(example_id, [])
                )
                if value is not None
            ]
            if values:
                merged[output_field] = sum(values) / len(values)
        output.append({field: fmt(merged.get(field)) for field in SE_EVAL_FIELDS})
    return output


def _build_auroc_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for score_name in DEFAULT_UNCERTAINTY_FIELDS:
        scored = [(finite_float(row.get(score_name)), str(row.get("is_low_quality")).lower() == "true") for row in rows]
        scored = [(score, risk) for score, risk in scored if score is not None]
        labels = [int(risk) for _, risk in scored]
        output.append({field: fmt({"score_name": score_name, "num_examples": len(labels), "num_positive": sum(labels), "num_negative": len(labels) - sum(labels), "auroc": auroc(labels, [score for score, _ in scored]) if labels else None}.get(field)) for field in AUROC_FIELDS})
    return output


def _build_rejection_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for score_name in DEFAULT_UNCERTAINTY_FIELDS:
        scored = [(finite_float(row.get(score_name)), str(row.get("is_low_quality")).lower() == "true", finite_float(row.get("quality_score"))) for row in rows]
        scored = sorted((item for item in scored if item[0] is not None), key=lambda item: item[0])
        for fraction in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            retained = scored[:max(1, round(len(scored) * (1 - fraction)))] if scored else []
            quality = [value for _, _, value in retained if value is not None]
            data = {"score_name": score_name, "rejection_fraction": fraction, "coverage": len(retained) / len(scored) if scored else None, "num_retained": len(retained), "accuracy": sum(not risk for _, risk, _ in retained) / len(retained) if retained else None, "mean_quality_score": sum(quality) / len(quality) if quality else None, "threshold": retained[-1][0] if retained else None}
            output.append({field: fmt(data.get(field)) for field in REJECTION_CURVE_FIELDS})
    return output


def _summary(generation_rows: list[dict[str, str]], example_rows: list[dict[str, str]], auroc_rows: list[dict[str, str]]) -> dict[str, Any]:
    quality = [value for value in (finite_float(row.get("quality_score")) for row in example_rows) if value is not None]
    type_counts = Counter(row.get("bioasq_type") or "unknown" for row in example_rows)
    return {
        "dataset": "bioasq",
        "evaluator": "lightweight_bioasq_v1",
        "official_metric_status": "local approximation; not the official BioASQ evaluation service",
        "num_examples": len(example_rows),
        "num_generations": len(generation_rows),
        "question_type_counts": dict(sorted(type_counts.items())),
        "quality_threshold": finite_float(example_rows[0].get("quality_threshold")) if example_rows else None,
        "low_quality_examples": sum(str(row.get("is_low_quality")).lower() == "true" for row in example_rows),
        "low_quality_rate": sum(str(row.get("is_low_quality")).lower() == "true" for row in example_rows) / len(example_rows) if example_rows else None,
        "mean_example_quality_score": sum(quality) / len(quality) if quality else None,
        "median_example_quality_score": sorted(quality)[len(quality) // 2] if quality else None,
        "auroc_low_quality_by_score": {row["score_name"]: finite_float(row.get("auroc")) for row in auroc_rows},
    }


def _write_report(summary: dict[str, Any], paths: dict[str, str], output_path: Path, *, overwrite: bool) -> None:
    fail_if_exists(output_path, overwrite)
    lines = [
        "# Lightweight BioASQ Quality Evaluation",
        "",
        "## Scope and limits",
        "",
        "This local evaluator follows BioASQ metric families but is not the official service: it uses unstemmed lexical ROUGE-2 and ROUGE-SU4 F1 approximations, no manual ideal-answer review, no synonym matching, and no official ranked-answer protocol.",
        "The `quality_threshold` is an operational low-quality cut-off for SE analysis, not an official BioASQ pass mark.",
        "",
        "## Summary",
        "",
    ]
    for key in ("num_examples", "num_generations", "question_type_counts", "quality_threshold", "low_quality_examples", "low_quality_rate", "mean_example_quality_score", "median_example_quality_score"):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_level4_bioasq(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_target: str = "mean",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate a BioASQ Level 4 run and write quality/SE analysis artifacts."""

    run_path = Path(run_dir)
    output_path = Path(output_dir) if output_dir else run_path / "bioasq_eval"
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "generation_quality": output_path / "bioasq_quality_generations.csv",
        "example_quality": output_path / "bioasq_quality_examples.csv",
        "se_eval_examples": output_path / "bioasq_se_eval_examples.csv",
        "auroc": output_path / "bioasq_se_auroc.csv",
        "rejection_curve": output_path / "bioasq_se_rejection_curve.csv",
        "summary": output_path / "bioasq_eval_summary.json",
        "report": output_path / "bioasq_eval_report.md",
    }
    examples = read_jsonl(run_path / "examples.jsonl")
    generations = read_jsonl(run_path / "cleaned_generations.jsonl")
    with (run_path / "se_scores.csv").open("r", encoding="utf-8", newline="") as infile:
        score_rows = list(csv.DictReader(infile))
    generation_uq_path = run_path / "generation_uq.csv"
    if generation_uq_path.exists():
        with generation_uq_path.open("r", encoding="utf-8", newline="") as infile:
            generation_uq_rows = list(csv.DictReader(infile))
    else:
        generation_uq_rows = []
    generation_rows, example_rows = build_quality_rows(examples, generations, quality_threshold=quality_threshold, quality_target=quality_target)
    se_rows = _merge_se_rows(example_rows, score_rows, generation_uq_rows)
    auroc_rows = _build_auroc_rows(se_rows)
    rejection_rows = _build_rejection_rows(se_rows)
    summary = _summary(generation_rows, example_rows, auroc_rows)
    write_csv(generation_rows, paths["generation_quality"], GENERATION_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(example_rows, paths["example_quality"], EXAMPLE_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(se_rows, paths["se_eval_examples"], SE_EVAL_FIELDS, overwrite=overwrite)
    write_csv(auroc_rows, paths["auroc"], AUROC_FIELDS, overwrite=overwrite)
    write_csv(rejection_rows, paths["rejection_curve"], REJECTION_CURVE_FIELDS, overwrite=overwrite)
    fail_if_exists(paths["summary"], overwrite)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    string_paths = {key: str(value) for key, value in paths.items()}
    _write_report(summary, string_paths, paths["report"], overwrite=overwrite)
    return {"summary": summary, "paths": string_paths}
