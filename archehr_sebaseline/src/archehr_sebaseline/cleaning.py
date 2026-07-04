"""Answer cleaning for semantic clustering."""

from __future__ import annotations

import re
from typing import Any


CITATION_RE = re.compile(r"\[\s*(?:S\s*)?\d+\s*\]", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_answer(answer: str) -> str:
    without_citations = CITATION_RE.sub(" ", answer)
    normalized = WHITESPACE_RE.sub(" ", without_citations)
    return normalized.strip()


def clean_generation_records(generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for generation in generations:
        if "raw_answer" not in generation:
            raise ValueError(f"Generation is missing raw_answer: {generation}")
        cleaned.append({**generation, "clean_answer": clean_answer(generation["raw_answer"])})
    return cleaned
