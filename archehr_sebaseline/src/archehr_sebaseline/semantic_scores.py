"""Shared Semantic Entropy score table helpers."""

from __future__ import annotations

from typing import Any

from .entropy import normalized_semantic_entropy, semantic_entropy_from_cluster_sizes


SCORE_FIELDS = [
    "example_id",
    "num_samples",
    "num_clusters",
    "cluster_sizes",
    "semantic_entropy",
    "normalized_semantic_entropy",
    "clustering_method",
]


def score_clusters(cluster_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    score_rows = []
    for cluster_record in cluster_records:
        cluster_sizes = cluster_record["cluster_sizes"]
        entropy = semantic_entropy_from_cluster_sizes(cluster_sizes)
        normalized_entropy = normalized_semantic_entropy(
            cluster_sizes, num_samples=cluster_record["num_samples"]
        )
        score_rows.append(
            {
                "example_id": cluster_record["example_id"],
                "num_samples": cluster_record["num_samples"],
                "num_clusters": len(cluster_sizes),
                "cluster_sizes": str(cluster_sizes),
                "semantic_entropy": f"{entropy:.10f}",
                "normalized_semantic_entropy": f"{normalized_entropy:.10f}",
                "clustering_method": cluster_record["clustering_method"],
            }
        )
    return score_rows
