"""Baseline uncertainty summaries from token-level generation scores."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


GENERATION_UQ_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "sample_id",
    "model_name",
    "generation_level",
    "num_generated_tokens",
    "sequence_logprob",
    "mean_token_logprob",
    "sequence_nll",
    "normalized_nll",
    "mean_token_entropy",
    "max_token_entropy",
]


EXAMPLE_UQ_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "num_generations",
    "mean_num_generated_tokens",
    "mean_sequence_logprob",
    "mean_token_logprob",
    "mean_sequence_nll",
    "mean_normalized_nll",
    "mean_token_entropy",
    "max_token_entropy",
    "sample_consistency_exact",
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def generation_uq_rows(generations: list[dict[str, Any]]) -> list[dict[str, str]]:
    rows = []
    for generation in generations:
        rows.append({field: _fmt(generation.get(field)) for field in GENERATION_UQ_FIELDS})
    return rows


def example_uq_rows(generations: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)

    rows = []
    for example_id in sorted(grouped):
        records = sorted(grouped[example_id], key=lambda item: item["sample_id"])
        clean_answers = [str(record.get("clean_answer") or "") for record in records]
        answer_counts = Counter(clean_answers)
        sample_consistency = max(answer_counts.values()) / len(records) if records else None

        row = {
            "example_id": example_id,
            "dataset": records[0].get("dataset"),
            "split": records[0].get("split"),
            "num_generations": len(records),
            "mean_num_generated_tokens": _mean(_numeric_values(records, "num_generated_tokens")),
            "mean_sequence_logprob": _mean(_numeric_values(records, "sequence_logprob")),
            "mean_token_logprob": _mean(_numeric_values(records, "mean_token_logprob")),
            "mean_sequence_nll": _mean(_numeric_values(records, "sequence_nll")),
            "mean_normalized_nll": _mean(_numeric_values(records, "normalized_nll")),
            "mean_token_entropy": _mean(_numeric_values(records, "mean_token_entropy")),
            "max_token_entropy": max(_numeric_values(records, "max_token_entropy"), default=None),
            "sample_consistency_exact": sample_consistency,
        }
        rows.append({field: _fmt(row.get(field)) for field in EXAMPLE_UQ_FIELDS})
    return rows
