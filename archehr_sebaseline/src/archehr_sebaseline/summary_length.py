"""Frozen helpers for the paired BioASQ summary-length intervention."""

from __future__ import annotations

import re
from typing import Any


CURRENT_SUMMARY_INSTRUCTION = (
    "Write one concise biomedical paragraph that directly answers the question."
)
SHORT_SUMMARY_INSTRUCTIONS = (
    "Keep the answer to one or two brief but complete sentences.",
    "Include only information needed to answer the question, without extra background.",
)
CURRENT_PROMPT_VERSION = "bioasq_summary_direct_v2"
SHORT_PROMPT_VERSION = "bioasq_summary_direct_one_or_two_sentences_v1"

_WORD_RE = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b", re.UNICODE)
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])(?:[\"')\]]*)\s+(?=[\"'(*\[]*[A-Z0-9])"
)


def build_short_summary_prompt(current_prompt: str) -> str:
    """Insert only the two pre-declared short-answer instructions."""

    if current_prompt.count(CURRENT_SUMMARY_INSTRUCTION) != 1:
        raise ValueError(
            "Current summary prompt must contain the frozen summary instruction exactly once."
        )
    if any(instruction in current_prompt for instruction in SHORT_SUMMARY_INSTRUCTIONS):
        raise ValueError("Current prompt already contains a short-answer instruction.")
    addition = "\n".join((CURRENT_SUMMARY_INSTRUCTION, *SHORT_SUMMARY_INSTRUCTIONS))
    return current_prompt.replace(CURRENT_SUMMARY_INSTRUCTION, addition)


def text_length_stats(text: str) -> dict[str, int | bool]:
    """Return deterministic word/sentence counts and two-sentence compliance."""

    normalized = " ".join(str(text or "").split())
    words = _WORD_RE.findall(normalized)
    if not normalized:
        sentence_count = 0
    else:
        sentences = [
            sentence.strip()
            for sentence in _SENTENCE_BOUNDARY_RE.split(normalized)
            if sentence.strip()
        ]
        sentence_count = len(sentences) or 1
    return {
        "word_count": len(words),
        "sentence_count": sentence_count,
        "one_or_two_sentence_compliant": sentence_count in {1, 2},
    }


def largest_cluster_fraction(cluster_sizes: list[int]) -> float:
    if not cluster_sizes or any(int(size) <= 0 for size in cluster_sizes):
        raise ValueError("cluster_sizes must contain positive integers.")
    return max(int(size) for size in cluster_sizes) / sum(int(size) for size in cluster_sizes)


def prompt_record_for_condition(
    current_prompt_record: dict[str, Any], condition: str
) -> dict[str, Any]:
    """Create a condition prompt without changing any other prompt metadata."""

    if str(current_prompt_record.get("prompt_version")) != CURRENT_PROMPT_VERSION:
        raise ValueError(
            "Summary intervention requires the frozen current prompt version "
            f"{CURRENT_PROMPT_VERSION!r}."
        )
    if str(current_prompt_record.get("evidence_mode")) != "none":
        raise ValueError("Summary intervention requires the no-evidence prompt.")
    if condition == "current":
        return {**current_prompt_record, "condition": condition}
    if condition == "short":
        return {
            **current_prompt_record,
            "prompt": build_short_summary_prompt(str(current_prompt_record["prompt"])),
            "prompt_version": SHORT_PROMPT_VERSION,
            "condition": condition,
        }
    raise ValueError(f"Unknown summary-length condition: {condition!r}.")
