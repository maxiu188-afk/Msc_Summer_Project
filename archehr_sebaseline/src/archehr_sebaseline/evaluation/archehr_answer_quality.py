"""Lightweight ArchEHR-QA answer quality evaluation.

This module keeps the current baseline intentionally simple. It evaluates
answers against clinician references and sentence relevance labels when the
ArchEHR-QA key file is available.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..data_io import read_jsonl, write_csv
from .uncertainty_metrics import (
    AUROC_FIELDS,
    DEFAULT_SCORE_NAMES,
    ECE_FIELDS,
    REJECTION_CURVE_FIELDS,
    RELIABILITY_BIN_FIELDS,
    build_auroc_rows,
    build_ece_rows,
    build_rejection_curve_rows,
    fail_if_exists,
    finite_float,
    fmt,
    read_csv_rows,
    write_auroc_bar_svg,
    write_rejection_curve_svg,
    write_reliability_svg,
)


GENERATION_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "sample_id",
    "parse_status",
    "has_gold_evidence",
    "num_predicted_citations",
    "num_gold_essential",
    "num_gold_lenient",
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "lenient_precision",
    "lenient_recall",
    "lenient_f1",
    "citation_score",
    "token_precision",
    "token_recall",
    "token_f1",
    "rouge_l_precision",
    "rouge_l_recall",
    "rouge_l",
    "answer_coverage",
    "relevance_score",
    "quality_score",
    "quality_band",
    "is_low_quality",
    "answer_text",
]

EXAMPLE_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "num_generations",
    "quality_target",
    "quality_threshold",
    "has_gold_evidence",
    "quality_score",
    "sample0_quality_score",
    "mean_quality_score",
    "best_quality_score",
    "sample0_low_quality",
    "mean_low_quality",
    "is_low_quality",
    "mean_strict_f1",
    "mean_lenient_f1",
    "mean_citation_score",
    "mean_answer_coverage",
    "mean_token_f1",
    "mean_rouge_l",
    "mean_relevance_score",
    "sample0_strict_f1",
    "sample0_lenient_f1",
    "sample0_citation_score",
    "sample0_answer_coverage",
    "sample0_token_f1",
    "sample0_rouge_l",
    "sample0_relevance_score",
    "quality_band",
    "num_gold_essential",
    "num_gold_lenient",
    "gold_answer",
]

SE_EVAL_EXAMPLE_FIELDS = [
    *EXAMPLE_QUALITY_FIELDS,
    "normalized_semantic_entropy",
    "semantic_entropy",
    "num_clusters",
    "normalized_citation_set_entropy",
    "citation_set_entropy",
    "num_unique_citation_sets",
    "mean_pairwise_citation_jaccard",
    "mean_token_entropy",
    "mean_normalized_nll",
    "max_token_entropy",
]

ANSWER_FIELD_CANDIDATES = [
    "clinician_answer",
    "gold_answer",
    "reference_answer",
    "answer",
]

ID_FIELD_CANDIDATES = [
    "id",
    "case_id",
    "example_id",
    "qid",
    "question_id",
]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_BRACKET_CITATION_RE = re.compile(r"\[([0-9,\s;]+)\]")


def normalize_sentence_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.lower().startswith("s") and text[1:].isdigit():
        return f"S{int(text[1:])}"
    if text.isdigit():
        return f"S{int(text)}"
    return text


def token_sort_key(value: str) -> tuple[int, str]:
    return (len(value), value)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0 if gold else 1.0, 0.0
    if not gold:
        return 0.0, 1.0, 0.0
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def token_f1(prediction: str, reference: str) -> float:
    precision, recall, f1 = token_overlap_metrics(prediction, reference)
    return f1


def token_overlap_metrics(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    pred_counts = Counter(pred_tokens)
    ref_counts = Counter(ref_tokens)
    overlap = sum((pred_counts & ref_counts).values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def rouge_l_f1(prediction: str, reference: str) -> float:
    precision, recall, f1 = rouge_l_metrics(prediction, reference)
    return f1


def rouge_l_metrics(prediction: str, reference: str) -> tuple[float, float, float]:
    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0
    previous = [0] * (len(ref_tokens) + 1)
    for pred_token in pred_tokens:
        current = [0]
        for index, ref_token in enumerate(ref_tokens, start=1):
            if pred_token == ref_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def quality_band(
    score: float,
    *,
    medium_threshold: float = 0.4,
    high_threshold: float = 0.7,
) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def _first_present(record: dict[str, Any], candidates: list[str]) -> Any:
    for candidate in candidates:
        if candidate in record and record[candidate] not in (None, ""):
            return record[candidate]
    return None


def _as_records(loaded: Any) -> list[dict[str, Any]]:
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    if not isinstance(loaded, dict):
        return []
    for key in ("examples", "data", "records", "questions", "cases"):
        value = loaded.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            return [
                {"id": record_id, **record}
                for record_id, record in value.items()
                if isinstance(record, dict)
            ]
    return [
        {"id": record_id, **record}
        for record_id, record in loaded.items()
        if isinstance(record, dict)
    ]


def _normalize_answer(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item not in (None, ""))
    if isinstance(value, dict):
        for candidate in ANSWER_FIELD_CANDIDATES:
            if candidate in value:
                return _normalize_answer(value[candidate])
        return " ".join(str(item) for item in value.values() if item not in (None, ""))
    return str(value or "").strip()


def _ids_from_value(value: Any) -> set[str]:
    if value in (None, ""):
        return set()
    if isinstance(value, (list, tuple, set)):
        return {normalize_sentence_id(item) for item in value if normalize_sentence_id(item)}
    if isinstance(value, dict):
        output = set()
        for key, item in value.items():
            if isinstance(item, bool) and item:
                output.add(normalize_sentence_id(key))
            elif isinstance(item, (int, float)) and item > 0:
                output.add(normalize_sentence_id(key))
            elif isinstance(item, str) and item.strip().lower() not in {"", "0", "false", "no", "none"}:
                output.add(normalize_sentence_id(key))
        return {item for item in output if item}
    return {
        normalize_sentence_id(item)
        for item in re.split(r"[,;\s]+", str(value))
        if normalize_sentence_id(item)
    }


def _extract_relevance(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    essential: set[str] = set()
    supplementary: set[str] = set()

    for key in ("gold_relevant_sentence_ids", "relevant_sentence_ids", "essential_sentence_ids"):
        essential |= _ids_from_value(record.get(key))
    for key in ("supplementary_sentence_ids", "additional_sentence_ids"):
        supplementary |= _ids_from_value(record.get(key))

    for key in (
        "answers",
        "sentence_relevance",
        "sentence_labels",
        "evidence_relevance",
        "relevance",
        "labels",
    ):
        value = record.get(key)
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, list):
            items = []
            for index, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    sentence_id = (
                        item.get("sentence_id")
                        or item.get("id")
                        or item.get("sentence")
                        or item.get("sentence_no")
                        or index
                    )
                    label = (
                        item.get("label")
                        or item.get("relevance")
                        or item.get("value")
                        or item.get("category")
                    )
                    items.append((sentence_id, label))
                else:
                    items.append((index, item))
        else:
            continue
        for sentence_id, label in items:
            normalized_id = normalize_sentence_id(sentence_id)
            normalized_label = str(label or "").strip().lower()
            if not normalized_id:
                continue
            if "essential" in normalized_label or normalized_label in {"1", "true", "relevant"}:
                essential.add(normalized_id)
            elif "supp" in normalized_label or normalized_label in {"2", "additional"}:
                supplementary.add(normalized_id)

    if not essential:
        essential |= extract_citations_from_answer_text(
            _normalize_answer(_first_present(record, ANSWER_FIELD_CANDIDATES))
        )

    supplementary -= essential
    return essential, supplementary


def extract_citations_from_answer_text(text: str) -> set[str]:
    """Extract sentence ids from clinician answer citations like ``[2, 5]``."""

    citations: set[str] = set()
    for match in _BRACKET_CITATION_RE.finditer(str(text or "")):
        for item in re.split(r"[,;\s]+", match.group(1)):
            normalized = normalize_sentence_id(item)
            if normalized:
                citations.add(normalized)
    return citations


def load_archehr_key(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a flexible ArchEHR-QA key JSON into example-id keyed references."""

    with Path(path).open("r", encoding="utf-8") as infile:
        loaded = json.load(infile)
    references = {}
    for index, record in enumerate(_as_records(loaded), start=1):
        example_id = _first_present(record, ID_FIELD_CANDIDATES) or index
        answer = _normalize_answer(_first_present(record, ANSWER_FIELD_CANDIDATES))
        essential, supplementary = _extract_relevance(record)
        references[str(example_id)] = {
            "example_id": str(example_id),
            "gold_answer": answer,
            "essential_sentence_ids": sorted(essential, key=token_sort_key),
            "supplementary_sentence_ids": sorted(supplementary, key=token_sort_key),
            "lenient_sentence_ids": sorted(essential | supplementary, key=token_sort_key),
        }
    return references


def _reference_for_example(
    example: dict[str, Any],
    references: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    example_id = str(example.get("id") or example.get("example_id"))
    reference = references.get(example_id, {}).copy()
    reference.setdefault("example_id", example_id)
    reference.setdefault("gold_answer", str(example.get("gold_answer") or ""))
    reference.setdefault("essential_sentence_ids", [])
    reference.setdefault(
        "lenient_sentence_ids",
        reference.get("essential_sentence_ids", []),
    )
    return reference


def _mean_present(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [
        value for value in (finite_float(row.get(field)) for row in rows)
        if value is not None
    ]
    return sum(values) / len(values) if values else None


def evaluate_generation_quality(
    generation: dict[str, Any],
    reference: dict[str, Any],
    *,
    quality_threshold: float,
    reference_only_threshold: float = 0.2,
) -> dict[str, Any]:
    predicted = {
        normalize_sentence_id(citation)
        for citation in generation.get("citation_ids") or []
        if normalize_sentence_id(citation)
    }
    essential = {
        normalize_sentence_id(citation)
        for citation in reference.get("essential_sentence_ids") or []
        if normalize_sentence_id(citation)
    }
    lenient = {
        normalize_sentence_id(citation)
        for citation in reference.get("lenient_sentence_ids") or []
        if normalize_sentence_id(citation)
    }
    strict_precision, strict_recall, strict_f1 = prf(predicted, essential)
    lenient_precision, lenient_recall, lenient_f1 = prf(predicted, lenient)
    has_gold_evidence = bool(essential or lenient)
    citation_score = max(strict_f1, lenient_f1) if has_gold_evidence else None

    answer_text = str(
        generation.get("clean_answer")
        or generation.get("answer_text")
        or generation.get("raw_answer")
        or ""
    )
    gold_answer = str(reference.get("gold_answer") or "")
    if gold_answer:
        token_precision, token_recall, text_token_f1 = token_overlap_metrics(
            answer_text, gold_answer
        )
        rouge_precision, rouge_recall, text_rouge_l = rouge_l_metrics(answer_text, gold_answer)
    else:
        token_precision = token_recall = text_token_f1 = 0.0
        rouge_precision = rouge_recall = text_rouge_l = 0.0
    answer_coverage = 0.5 * token_recall + 0.5 * rouge_recall
    lexical_similarity = 0.5 * text_token_f1 + 0.5 * text_rouge_l
    relevance = 0.75 * answer_coverage + 0.25 * lexical_similarity
    quality = (0.75 * citation_score + 0.25 * relevance) if has_gold_evidence else relevance
    applied_threshold = quality_threshold if has_gold_evidence else reference_only_threshold
    band = (
        quality_band(quality)
        if has_gold_evidence
        else quality_band(quality, medium_threshold=reference_only_threshold, high_threshold=0.35)
    )

    row = {
        "example_id": generation.get("example_id"),
        "dataset": generation.get("dataset"),
        "split": generation.get("split"),
        "sample_id": generation.get("sample_id"),
        "parse_status": generation.get("parse_status", ""),
        "has_gold_evidence": has_gold_evidence,
        "num_predicted_citations": len(predicted),
        "num_gold_essential": len(essential),
        "num_gold_lenient": len(lenient),
        "strict_precision": strict_precision if has_gold_evidence else None,
        "strict_recall": strict_recall if has_gold_evidence else None,
        "strict_f1": strict_f1 if has_gold_evidence else None,
        "lenient_precision": lenient_precision if has_gold_evidence else None,
        "lenient_recall": lenient_recall if has_gold_evidence else None,
        "lenient_f1": lenient_f1 if has_gold_evidence else None,
        "citation_score": citation_score,
        "token_precision": token_precision,
        "token_recall": token_recall,
        "token_f1": text_token_f1,
        "rouge_l_precision": rouge_precision,
        "rouge_l_recall": rouge_recall,
        "rouge_l": text_rouge_l,
        "answer_coverage": answer_coverage,
        "relevance_score": relevance,
        "quality_score": quality,
        "quality_band": band,
        "is_low_quality": quality < applied_threshold,
        "answer_text": answer_text,
    }
    return row


def build_quality_rows(
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    references: dict[str, dict[str, Any]],
    *,
    quality_threshold: float = 0.4,
    reference_only_threshold: float = 0.2,
    quality_target: str = "sample0",
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    examples_by_id = {str(example["id"]): example for example in examples}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)

    generation_rows: list[dict[str, str]] = []
    example_rows: list[dict[str, str]] = []
    for example_id in sorted(grouped, key=token_sort_key):
        example = examples_by_id.get(example_id, {"id": example_id})
        reference = _reference_for_example(example, references)
        records = sorted(grouped[example_id], key=lambda item: int(item.get("sample_id", 0)))
        raw_rows = [
            evaluate_generation_quality(
                record,
                reference,
                quality_threshold=quality_threshold,
                reference_only_threshold=reference_only_threshold,
            )
            for record in records
        ]
        generation_rows.extend(
            {field: fmt(row.get(field)) for field in GENERATION_QUALITY_FIELDS}
            for row in raw_rows
        )

        sample0 = raw_rows[0]
        quality_values = [float(row["quality_score"]) for row in raw_rows]
        has_gold_evidence = str(sample0.get("has_gold_evidence")).lower() == "true"
        applied_threshold = quality_threshold if has_gold_evidence else reference_only_threshold
        mean_quality = sum(quality_values) / len(quality_values)
        target_quality = mean_quality if quality_target == "mean" else float(sample0["quality_score"])
        row = {
            "example_id": example_id,
            "dataset": example.get("dataset") or sample0.get("dataset"),
            "split": example.get("split") or sample0.get("split"),
            "num_generations": len(raw_rows),
            "quality_target": quality_target,
            "quality_threshold": applied_threshold,
            "has_gold_evidence": has_gold_evidence,
            "quality_score": target_quality,
            "sample0_quality_score": sample0["quality_score"],
            "mean_quality_score": mean_quality,
            "best_quality_score": max(quality_values),
            "sample0_low_quality": bool(sample0["is_low_quality"]),
            "mean_low_quality": mean_quality < applied_threshold,
            "is_low_quality": target_quality < applied_threshold,
            "mean_strict_f1": _mean_present(raw_rows, "strict_f1"),
            "mean_lenient_f1": _mean_present(raw_rows, "lenient_f1"),
            "mean_citation_score": _mean_present(raw_rows, "citation_score"),
            "mean_answer_coverage": _mean_present(raw_rows, "answer_coverage"),
            "mean_token_f1": _mean_present(raw_rows, "token_f1"),
            "mean_rouge_l": _mean_present(raw_rows, "rouge_l"),
            "mean_relevance_score": _mean_present(raw_rows, "relevance_score"),
            "sample0_strict_f1": sample0["strict_f1"],
            "sample0_lenient_f1": sample0["lenient_f1"],
            "sample0_citation_score": sample0["citation_score"],
            "sample0_answer_coverage": sample0["answer_coverage"],
            "sample0_token_f1": sample0["token_f1"],
            "sample0_rouge_l": sample0["rouge_l"],
            "sample0_relevance_score": sample0["relevance_score"],
            "quality_band": (
                quality_band(target_quality)
                if has_gold_evidence
                else quality_band(
                    target_quality,
                    medium_threshold=reference_only_threshold,
                    high_threshold=0.35,
                )
            ),
            "num_gold_essential": sample0["num_gold_essential"],
            "num_gold_lenient": sample0["num_gold_lenient"],
            "gold_answer": reference.get("gold_answer", ""),
        }
        example_rows.append({field: fmt(row.get(field)) for field in EXAMPLE_QUALITY_FIELDS})
    return generation_rows, example_rows


def _rows_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {str(row.get("example_id")): row for row in rows if row.get("example_id")}


def _aggregate_generation_uq(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("example_id"))].append(row)
    output = {}
    for example_id, records in grouped.items():
        mean_entropy_values = [
            value for value in (finite_float(row.get("mean_token_entropy")) for row in records)
            if value is not None
        ]
        normalized_nll_values = [
            value for value in (finite_float(row.get("normalized_nll")) for row in records)
            if value is not None
        ]
        max_entropy_values = [
            value for value in (finite_float(row.get("max_token_entropy")) for row in records)
            if value is not None
        ]
        output[example_id] = {
            "mean_token_entropy": fmt(
                sum(mean_entropy_values) / len(mean_entropy_values)
                if mean_entropy_values else None
            ),
            "mean_normalized_nll": fmt(
                sum(normalized_nll_values) / len(normalized_nll_values)
                if normalized_nll_values else None
            ),
            "max_token_entropy": fmt(max(max_entropy_values) if max_entropy_values else None),
        }
    return output


def build_se_eval_rows(
    example_quality_rows: list[dict[str, str]],
    *,
    answer_score_rows: list[dict[str, str]],
    citation_uq_rows: list[dict[str, str]],
    generation_uq_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    answer_by_id = _rows_by_id(answer_score_rows)
    citation_by_id = _rows_by_id(citation_uq_rows)
    generation_by_id = _aggregate_generation_uq(generation_uq_rows)
    rows = []
    for quality_row in example_quality_rows:
        example_id = str(quality_row["example_id"])
        merged = {
            **quality_row,
            **answer_by_id.get(example_id, {}),
            **citation_by_id.get(example_id, {}),
            **generation_by_id.get(example_id, {}),
        }
        rows.append({field: fmt(merged.get(field)) for field in SE_EVAL_EXAMPLE_FIELDS})
    return rows


def summarize_evaluation(
    generation_rows: list[dict[str, str]],
    example_rows: list[dict[str, str]],
    auroc_rows: list[dict[str, str]],
    ece_rows: list[dict[str, str]],
) -> dict[str, Any]:
    quality_values = [
        value for value in (finite_float(row.get("quality_score")) for row in generation_rows)
        if value is not None
    ]
    example_quality_values = [
        value for value in (finite_float(row.get("quality_score")) for row in example_rows)
        if value is not None
    ]
    low_quality_examples = sum(
        1 for row in example_rows if str(row.get("is_low_quality")).lower() == "true"
    )
    examples_with_gold_evidence = sum(
        1 for row in example_rows if str(row.get("has_gold_evidence")).lower() == "true"
    )
    aurocs = {
        row["score_name"]: finite_float(row.get("auroc"))
        for row in auroc_rows
        if row.get("score_name")
    }
    eces = {
        row["score_name"]: finite_float(row.get("ece"))
        for row in ece_rows
        if row.get("score_name")
    }
    return {
        "dataset": "archehr_qa",
        "num_examples": len(example_rows),
        "num_generations": len(generation_rows),
        "examples_with_gold_evidence_labels": examples_with_gold_evidence,
        "examples_reference_only": len(example_rows) - examples_with_gold_evidence,
        "low_quality_examples": low_quality_examples,
        "low_quality_rate": low_quality_examples / len(example_rows) if example_rows else None,
        "mean_generation_quality_score": (
            sum(quality_values) / len(quality_values) if quality_values else None
        ),
        "mean_example_quality_score": (
            sum(example_quality_values) / len(example_quality_values)
            if example_quality_values else None
        ),
        "auroc_low_quality_by_score": aurocs,
        "ece_by_score": eces,
    }


def write_report(summary: dict[str, Any], paths: dict[str, str], path: str | Path, *, overwrite: bool) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fail_if_exists(output_path, overwrite)

    aurocs = summary.get("auroc_low_quality_by_score", {})
    eces = summary.get("ece_by_score", {})
    lines = [
        "# ArchEHR-QA SE Evaluation Report",
        "",
        "## Scope",
        "",
        "This is a lightweight evaluation pass for the SE baseline. Answer quality is estimated from evidence-citation agreement plus answer-content coverage against the clinician answer.",
        "",
        "## Summary",
        "",
        f"- examples: {summary.get('num_examples')}",
        f"- generations: {summary.get('num_generations')}",
        f"- examples_with_gold_evidence_labels: {summary.get('examples_with_gold_evidence_labels')}",
        f"- examples_reference_only: {summary.get('examples_reference_only')}",
        f"- low_quality_examples: {summary.get('low_quality_examples')}",
        f"- low_quality_rate: {fmt(summary.get('low_quality_rate'))}",
        f"- mean_generation_quality_score: {fmt(summary.get('mean_generation_quality_score'))}",
        f"- mean_example_quality_score: {fmt(summary.get('mean_example_quality_score'))}",
        "",
        "## AUROC",
        "",
    ]
    for score_name in DEFAULT_SCORE_NAMES:
        if score_name in aurocs:
            lines.append(f"- {score_name}: {fmt(aurocs[score_name])}")
    lines.extend(["", "## ECE", ""])
    for score_name in DEFAULT_SCORE_NAMES:
        if score_name in eces:
            lines.append(f"- {score_name}: {fmt(eces[score_name])}")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- generation quality: `{paths['generation_quality']}`",
            f"- example quality: `{paths['example_quality']}`",
            f"- SE evaluation examples: `{paths['se_eval_examples']}`",
            f"- AUROC table: `{paths['auroc']}`",
            f"- ECE table: `{paths['ece']}`",
            f"- rejection curve: `{paths['rejection_curve']}`",
            f"- AUROC plot: `{paths['auroc_plot']}`",
            f"- rejection plot: `{paths['rejection_plot']}`",
            f"- reliability plot: `{paths['reliability_plot']}`",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_archehr_se_run(
    run_dir: str | Path,
    key_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    quality_threshold: float = 0.4,
    reference_only_threshold: float = 0.2,
    quality_target: str = "sample0",
    overwrite: bool = False,
) -> dict[str, Any]:
    run_path = Path(run_dir)
    output_path = Path(output_dir) if output_dir else run_path / "eval"
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "generation_quality": output_path / "answer_quality_generations.csv",
        "example_quality": output_path / "answer_quality_examples.csv",
        "se_eval_examples": output_path / "se_eval_examples.csv",
        "auroc": output_path / "se_auroc.csv",
        "ece": output_path / "se_ece.csv",
        "reliability_bins": output_path / "se_reliability_bins.csv",
        "rejection_curve": output_path / "se_rejection_curve.csv",
        "summary": output_path / "evaluation_summary.json",
        "report": output_path / "evaluation_report.md",
        "auroc_plot": output_path / "auroc_bar.svg",
        "rejection_plot": output_path / "rejection_curve.svg",
        "reliability_plot": output_path / "reliability_diagram.svg",
    }

    examples = read_jsonl(run_path / "examples.jsonl")
    generations = read_jsonl(run_path / "cleaned_generations.jsonl")
    references = load_archehr_key(key_path)
    generation_rows, example_rows = build_quality_rows(
        examples,
        generations,
        references,
        quality_threshold=quality_threshold,
        reference_only_threshold=reference_only_threshold,
        quality_target=quality_target,
    )

    se_eval_rows = build_se_eval_rows(
        example_rows,
        answer_score_rows=read_csv_rows(run_path / "answer_se_scores.csv"),
        citation_uq_rows=read_csv_rows(run_path / "citation_uq.csv"),
        generation_uq_rows=read_csv_rows(run_path / "generation_uq.csv"),
    )
    auroc_rows = build_auroc_rows(se_eval_rows, score_names=DEFAULT_SCORE_NAMES)
    ece_rows, reliability_rows = build_ece_rows(se_eval_rows, score_names=DEFAULT_SCORE_NAMES)
    rejection_rows = build_rejection_curve_rows(se_eval_rows, score_names=DEFAULT_SCORE_NAMES)
    summary = summarize_evaluation(generation_rows, se_eval_rows, auroc_rows, ece_rows)

    write_csv(generation_rows, paths["generation_quality"], GENERATION_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(example_rows, paths["example_quality"], EXAMPLE_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(se_eval_rows, paths["se_eval_examples"], SE_EVAL_EXAMPLE_FIELDS, overwrite=overwrite)
    write_csv(auroc_rows, paths["auroc"], AUROC_FIELDS, overwrite=overwrite)
    write_csv(ece_rows, paths["ece"], ECE_FIELDS, overwrite=overwrite)
    write_csv(reliability_rows, paths["reliability_bins"], RELIABILITY_BIN_FIELDS, overwrite=overwrite)
    write_csv(rejection_rows, paths["rejection_curve"], REJECTION_CURVE_FIELDS, overwrite=overwrite)
    write_auroc_bar_svg(auroc_rows, paths["auroc_plot"], overwrite=overwrite)
    write_rejection_curve_svg(rejection_rows, paths["rejection_plot"], overwrite=overwrite)
    write_reliability_svg(reliability_rows, paths["reliability_plot"], overwrite=overwrite)

    summary_path = paths["summary"]
    fail_if_exists(summary_path, overwrite)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    string_paths = {name: str(path) for name, path in paths.items()}
    write_report(summary, string_paths, paths["report"], overwrite=overwrite)
    return {"summary": summary, "paths": string_paths}
