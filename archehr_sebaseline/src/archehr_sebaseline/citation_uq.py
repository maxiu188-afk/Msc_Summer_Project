"""Citation-set uncertainty metrics for grounded QA generations."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .entropy import normalized_semantic_entropy, semantic_entropy_from_cluster_sizes


CITATION_UQ_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "num_samples",
    "num_unique_citation_sets",
    "citation_set_counts",
    "citation_set_entropy",
    "normalized_citation_set_entropy",
    "mean_pairwise_citation_jaccard",
    "citation_vote_distribution",
]


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def citation_set_key(citation_ids: list[str] | None) -> tuple[str, ...]:
    return tuple(sorted({str(citation) for citation in citation_ids or []}, key=lambda item: (len(item), item)))


def jaccard_similarity(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def mean_pairwise_jaccard(citation_sets: list[tuple[str, ...]]) -> float | None:
    if len(citation_sets) < 2:
        return None
    values = []
    for i, left in enumerate(citation_sets):
        for right in citation_sets[i + 1:]:
            values.append(jaccard_similarity(left, right))
    return sum(values) / len(values) if values else None


def citation_uq_rows(parsed_generations: list[dict[str, Any]]) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in parsed_generations:
        grouped[str(record["example_id"])].append(record)

    rows: list[dict[str, str]] = []
    for example_id in sorted(grouped):
        records = sorted(grouped[example_id], key=lambda item: int(item.get("sample_id", 0)))
        citation_sets = [citation_set_key(record.get("citation_ids")) for record in records]
        set_counts = Counter(citation_sets)
        cluster_sizes = list(set_counts.values())
        entropy = semantic_entropy_from_cluster_sizes(cluster_sizes)
        normalized_entropy = normalized_semantic_entropy(cluster_sizes, num_samples=len(records))
        citation_votes = Counter(citation for citation_set in citation_sets for citation in citation_set)
        row = {
            "example_id": example_id,
            "dataset": records[0].get("dataset"),
            "split": records[0].get("split"),
            "num_samples": len(records),
            "num_unique_citation_sets": len(set_counts),
            "citation_set_counts": {
                ",".join(citation_set) if citation_set else "[none]": count
                for citation_set, count in sorted(set_counts.items())
            },
            "citation_set_entropy": entropy,
            "normalized_citation_set_entropy": normalized_entropy,
            "mean_pairwise_citation_jaccard": mean_pairwise_jaccard(citation_sets),
            "citation_vote_distribution": dict(sorted(citation_votes.items())),
        }
        rows.append({field: _fmt(row.get(field)) for field in CITATION_UQ_FIELDS})
    return rows

