"""Parse structured grounded answers and citations."""

from __future__ import annotations

import json
import re
from typing import Any


BRACKETED_CITATION_RE = re.compile(r"\[([^\]]+)\]")
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_citation_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    return text


def _split_citation_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        citations = []
        for item in value:
            citations.extend(_split_citation_values(item))
        return citations

    text = normalize_citation_id(value)
    citations = []
    for part in re.split(r"[,;\s]+", text):
        part = normalize_citation_id(part)
        if re.fullmatch(r"S?\d+", part):
            citations.append(part)
    return citations


def extract_citation_ids(text: str) -> list[str]:
    citations: list[str] = []
    for match in BRACKETED_CITATION_RE.finditer(str(text or "")):
        citations.extend(_split_citation_values(match.group(1)))
    return sorted(set(citations), key=lambda item: (len(item), item))


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    candidates.extend(match.group(1).strip() for match in JSON_FENCE_RE.finditer(text))

    list_match = re.search(r"\[\s*\{.*?\}\s*\]", text, flags=re.DOTALL)
    if list_match:
        candidates.append(list_match.group(0))

    dict_match = re.search(r"\{\s*\"(?:statement|answer|text)\".*?\}", text, flags=re.DOTALL)
    if dict_match:
        candidates.append(dict_match.group(0))

    unique_candidates = []
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate and candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _coerce_statement_item(item: Any) -> dict[str, Any] | None:
    if not isinstance(item, dict):
        return None
    statement = item.get("statement") or item.get("answer") or item.get("text")
    citation = item.get("citation") or item.get("citations") or item.get("source")
    if statement is None:
        return None
    citations = _split_citation_values(citation)
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
    for candidate in _json_candidates(text):
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            loaded = [loaded]
        if isinstance(loaded, list):
            for item in loaded:
                parsed_item = _coerce_statement_item(item)
                if parsed_item is not None and parsed_item["statement"]:
                    parsed_items.append(parsed_item)
            if parsed_items:
                parse_status = "json"
                break

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
