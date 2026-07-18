"""Post-hoc self-report uncertainty baselines for completed Level 4 runs."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Protocol


SELF_REPORT_GENERATION_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "sample_id",
    "verbalized_confidence",
    "verbalized_confidence_uncertainty",
    "p_true",
    "p_true_uncertainty",
    "p_true_blind",
    "p_true_blind_uncertainty",
    "confidence_response",
    "answer_source",
]

SELF_REPORT_EXAMPLE_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "num_generations",
    "mean_verbalized_confidence",
    "verbalized_confidence_uncertainty",
    "mean_p_true",
    "p_true_uncertainty",
    "mean_p_true_blind",
    "p_true_blind_uncertainty",
    "answer_source",
]

_NUMBER_RE = re.compile(r"(?<!\d)(100(?:\.0+)?|\d{1,2}(?:\.\d+)?)(?!\d)")


class SelfReportScorer(Protocol):
    """Model interface required by the two post-hoc uncertainty baselines."""

    def generate_deterministic(self, prompt: str, *, max_new_tokens: int) -> str:
        """Return a short deterministic response."""

    def binary_continuation_probability(self, prompt: str, *, true_text: str, false_text: str) -> float:
        """Return probability of ``true_text`` after normalizing against false."""


def _evidence_text(example: dict[str, Any], *, max_characters: int = 5000) -> str:
    snippets = [str(item).strip() for item in example.get("evidence_sentences") or [] if str(item).strip()]
    return "\n".join(f"[S{index}] {snippet}" for index, snippet in enumerate(snippets, start=1))[:max_characters]


def _uses_evidence(example: dict[str, Any]) -> bool:
    """Use the generation prompt's evidence mode, never the stored metadata alone."""

    return str(example.get("prompt_evidence_mode") or "provided").lower() == "provided"


def build_verbalized_confidence_prompt(example: dict[str, Any], answer: str) -> str:
    if not _uses_evidence(example):
        return (
            "You are checking a biomedical answer. Using the question and your biomedical knowledge, "
            "estimate the probability that the proposed answer is fully correct. "
            "Reply with one number from 0 to 100 and no other text.\n\n"
            f"Question: {example.get('question', '')}\n"
            f"Proposed answer: {answer}\n"
            "Confidence (0-100):"
        )
    return (
        "You are checking a grounded biomedical answer. Using only the question and evidence below, "
        "estimate the probability that the proposed answer is fully correct and supported. "
        "Reply with one number from 0 to 100 and no other text.\n\n"
        f"Question: {example.get('question', '')}\n"
        f"Evidence:\n{_evidence_text(example)}\n\n"
        f"Proposed answer: {answer}\n"
        "Confidence (0-100):"
    )


def build_p_true_prompt(
    example: dict[str, Any],
    answer: str,
) -> str:
    evidence_section = (
        f"Evidence:\n{_evidence_text(example)}\n\n" if _uses_evidence(example) else ""
    )
    basis = "the question and supplied evidence" if _uses_evidence(example) else "the question"
    return (
        f"Using {basis}, is the proposed answer fully correct? Reply True or False.\n\n"
        f"Question: {example.get('question', '')}\n"
        f"{evidence_section}"
        f"Proposed answer: {answer}\n"
        "Verdict:"
    )


def parse_verbalized_confidence(text: str) -> float | None:
    """Parse a 0--100 self-reported confidence without accepting prose as a score."""

    match = _NUMBER_RE.search(str(text or ""))
    if not match:
        return None
    value = float(match.group(1))
    return value / 100.0 if 0.0 <= value <= 100.0 else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def score_self_report_generations(
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    scorer: SelfReportScorer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every generated answer and aggregate the two UQ baselines by question."""

    examples_by_id = {str(example["id"]): example for example in examples}
    generation_rows = []
    for generation in sorted(generations, key=lambda item: (str(item["example_id"]), int(item.get("sample_id", 0)))):
        example = examples_by_id[str(generation["example_id"])]
        answer = str(generation.get("clean_answer") or generation.get("raw_answer") or "").strip()
        confidence_response = scorer.generate_deterministic(
            build_verbalized_confidence_prompt(example, answer),
            max_new_tokens=8,
        )
        confidence = parse_verbalized_confidence(confidence_response)
        p_true = scorer.binary_continuation_probability(
            build_p_true_prompt(example, answer),
            true_text=" True",
            false_text=" False",
        )
        generation_rows.append(
            {
                "example_id": generation["example_id"],
                "dataset": generation.get("dataset") or example.get("dataset"),
                "split": generation.get("split") or example.get("split"),
                "sample_id": generation.get("sample_id"),
                "verbalized_confidence": confidence,
                "verbalized_confidence_uncertainty": 1.0 - confidence if confidence is not None else None,
                "p_true": p_true,
                "p_true_uncertainty": 1.0 - p_true,
                "confidence_response": confidence_response,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in generation_rows:
        grouped[str(row["example_id"])].append(row)
    example_rows = []
    for example_id, rows in sorted(grouped.items()):
        confidence = _mean([float(row["verbalized_confidence"]) for row in rows if row["verbalized_confidence"] is not None])
        p_true = _mean([float(row["p_true"]) for row in rows if row["p_true"] is not None])
        example_rows.append(
            {
                "example_id": example_id,
                "dataset": rows[0].get("dataset"),
                "split": rows[0].get("split"),
                "num_generations": len(rows),
                "mean_verbalized_confidence": confidence,
                "verbalized_confidence_uncertainty": 1.0 - confidence if confidence is not None else None,
                "mean_p_true": p_true,
                "p_true_uncertainty": 1.0 - p_true if p_true is not None else None,
            }
        )
    return generation_rows, example_rows


def score_self_report_best_answers(
    examples: list[dict[str, Any]],
    best_generations: list[dict[str, Any]],
    scorer: SelfReportScorer,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score one low-temperature answer per question with blind P(True)."""

    examples_by_id = {str(example["id"]): example for example in examples}
    best_by_id: dict[str, dict[str, Any]] = {}
    for generation in best_generations:
        example_id = str(generation["example_id"])
        if example_id in best_by_id:
            raise ValueError(f"More than one best generation for {example_id}.")
        best_by_id[example_id] = generation
    if set(best_by_id) != set(examples_by_id):
        raise ValueError("Best-generation examples do not match the source examples.")

    generation_rows = []
    example_rows = []
    for example_id in sorted(best_by_id):
        generation = best_by_id[example_id]
        example = examples_by_id[example_id]
        answer = str(generation.get("clean_answer") or generation.get("raw_answer") or "").strip()
        if not answer:
            raise ValueError(f"Best generation is empty for {example_id}.")
        confidence_response = scorer.generate_deterministic(
            build_verbalized_confidence_prompt(example, answer),
            max_new_tokens=8,
        )
        confidence = parse_verbalized_confidence(confidence_response)
        p_true_blind = scorer.binary_continuation_probability(
            build_p_true_prompt(example, answer),
            true_text=" True",
            false_text=" False",
        )
        generation_row = {
            "example_id": generation["example_id"],
            "dataset": generation.get("dataset") or example.get("dataset"),
            "split": generation.get("split") or example.get("split"),
            "sample_id": generation.get("sample_id", 0),
            "verbalized_confidence": confidence,
            "verbalized_confidence_uncertainty": 1.0 - confidence if confidence is not None else None,
            # Keep the concise names as aliases of the only active P(True)
            # condition for legacy readers.
            "p_true": p_true_blind,
            "p_true_uncertainty": 1.0 - p_true_blind,
            "p_true_blind": p_true_blind,
            "p_true_blind_uncertainty": 1.0 - p_true_blind,
            "confidence_response": confidence_response,
            "answer_source": "best_generation_low_temperature",
        }
        generation_rows.append(generation_row)
        example_rows.append(
            {
                "example_id": generation["example_id"],
                "dataset": generation_row["dataset"],
                "split": generation_row["split"],
                "num_generations": 1,
                "mean_verbalized_confidence": confidence,
                "verbalized_confidence_uncertainty": generation_row["verbalized_confidence_uncertainty"],
                "mean_p_true": p_true_blind,
                "p_true_uncertainty": generation_row["p_true_uncertainty"],
                "mean_p_true_blind": p_true_blind,
                "p_true_blind_uncertainty": generation_row["p_true_blind_uncertainty"],
                "answer_source": generation_row["answer_source"],
            }
        )
    return generation_rows, example_rows
