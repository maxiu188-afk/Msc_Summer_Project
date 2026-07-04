"""Simple deterministic clustering fallbacks."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def cluster_by_clean_answer_exact(generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster exact cleaned answers.

    This is not a semantic-equivalence method. It is a fast smoke-test
    fallback used until an NLI or LLM entailment clusterer is introduced.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)

    cluster_records = []
    for example_id in sorted(grouped):
        example_generations = sorted(grouped[example_id], key=lambda item: item["sample_id"])
        answer_to_cluster_id: dict[str, int] = {}
        clusters: list[dict[str, Any]] = []
        semantic_ids: list[int] = []

        for generation in example_generations:
            clean_answer = generation.get("clean_answer")
            if clean_answer is None:
                raise ValueError("Exact clustering requires clean_answer on each generation.")
            if clean_answer not in answer_to_cluster_id:
                cluster_id = len(answer_to_cluster_id)
                answer_to_cluster_id[clean_answer] = cluster_id
                clusters.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster_key": clean_answer,
                        "sample_ids": [],
                    }
                )
            cluster_id = answer_to_cluster_id[clean_answer]
            semantic_ids.append(cluster_id)
            clusters[cluster_id]["sample_ids"].append(generation["sample_id"])

        for cluster in clusters:
            cluster["cluster_size"] = len(cluster["sample_ids"])

        cluster_records.append(
            {
                "example_id": example_id,
                "num_samples": len(example_generations),
                "clustering_method": "exact_clean_answer",
                "semantic_ids": semantic_ids,
                "cluster_sizes": [cluster["cluster_size"] for cluster in clusters],
                "clusters": clusters,
            }
        )

    return cluster_records
