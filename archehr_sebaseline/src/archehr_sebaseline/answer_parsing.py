"""Parse structured grounded answers and citations."""

from __future__ import annotations

import json
import re
from typing import Any


CITATION_RE = re.compile(r"\bS?\d+\b|\[[^\]]+\]")


def normalize_citation_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def extract_citation_ids(text: str) -> list[str]:
    citations: list[str] = []
    for match in CITATION_RE.finditer(str(text or "")):
        raw = normalize_citation_id(match.group(0))
        for part in re.split(r"[,;\s]+", raw):
            part = normalize_citation_id(part)
            if part and re.fullmatch(r"S?\d+", part):
                citations.append(part)
    return sorted(set(citations), key=lambda item: (len(item), item))


def _coerce_statement_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    statement = item.get("statement") or item.get("answer") or item.get("text")
    citation = item.get("citation") or item.get("citations") or item.get("source")
    if statement is None:
        return None
    if isinstance(citation, list):
        citations = [normalize_citation_id(value) for value in citation if str(value).strip()]
    else:
        citations = [normalize_citation_id(citation)] if citation is not None else []
    if not citations:
        citations = extract_citation_ids(str(statement))
    return {
        "statement": str(statement).strip(),
        "citations": sorted(set(citations), key=lambda value: (len(value), value)),
    }


def parse_grounded_answer(raw_answer: str) -> dict[str, Any]:
    """Parse the expected JSON answer format with a citation-regex fallback."""

    text = str(raw_answer or "").strip()
    parsed_items: list[dict[str, Any]] = []
    parse_status = "fallback"
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, dict):
        loaded = [loaded]
    if isinstance(loaded, list):
        for item in loaded:
            parsed_item = _coerce_statement_item(item)
            if parsed_item is not None and parsed_item["statement"]:
                parsed_items.append(parsed_item)
        if parsed_items:
            parse_status = "json"

    if not parsed_items:
        citations = extract_citation_ids(text)
        parsed_items = [{"statement": text, "citations": citations}]

    citation_ids = sorted(
        {citation for item in parsed_items for citation in item["citations"] if citation},
        key=lambda value: (len(value), value),
    )
    statement_text = " ".join(item["statement"] for item in parsed_items).strip()
    return {
        "parse_status": parse_status,
        "statements": parsed_items,
        "citation_ids": citation_ids,
        "answer_text": statement_text,
    }


def parse_generation_records(generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for generation in generations:
        parsed = parse_grounded_answer(str(generation.get("raw_answer") or ""))
        records.append({**generation, **parsed})
    return records

