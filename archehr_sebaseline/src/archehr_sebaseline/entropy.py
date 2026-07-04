"""Discrete Semantic Entropy utilities."""

from __future__ import annotations

import math


def semantic_entropy_from_cluster_sizes(cluster_sizes: list[int]) -> float:
    total = sum(cluster_sizes)
    if total <= 0:
        raise ValueError("At least one sample is required to compute entropy.")

    entropy = 0.0
    for cluster_size in cluster_sizes:
        if cluster_size < 0:
            raise ValueError("Cluster sizes must be non-negative.")
        if cluster_size == 0:
            continue
        probability = cluster_size / total
        entropy -= probability * math.log(probability)
    return entropy


def normalized_semantic_entropy(
    cluster_sizes: list[int],
    *,
    num_samples: int | None = None,
) -> float:
    total = sum(cluster_sizes) if num_samples is None else num_samples
    if total <= 1:
        return 0.0
    entropy = semantic_entropy_from_cluster_sizes(cluster_sizes)
    return entropy / math.log(total)


def logsumexp(values: list[float]) -> float:
    if not values:
        raise ValueError("logsumexp requires at least one value.")
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def predictive_entropy_from_logprobs(log_probs: list[float]) -> float:
    """Monte Carlo estimate of predictive entropy from sequence log-probs."""

    if not log_probs:
        raise ValueError("At least one log-probability is required.")
    return -sum(log_probs) / len(log_probs)


def semantic_entropy_from_ids_and_logprobs(
    semantic_ids: list[int],
    log_likelihoods: list[float],
) -> float:
    """Compute likelihood-weighted Semantic Entropy.

    This follows the reference implementation's idea: normalize sampled
    sequence likelihoods, sum probability mass within each semantic cluster,
    then compute entropy over semantic-cluster probabilities.
    """

    if len(semantic_ids) != len(log_likelihoods):
        raise ValueError("semantic_ids and log_likelihoods must have the same length.")
    if not semantic_ids:
        raise ValueError("At least one semantic id is required.")

    total_logprob = logsumexp(log_likelihoods)
    normalized_logprobs = [value - total_logprob for value in log_likelihoods]
    cluster_logprobs = []
    for cluster_id in sorted(set(semantic_ids)):
        cluster_values = [
            logprob for sem_id, logprob in zip(semantic_ids, normalized_logprobs)
            if sem_id == cluster_id
        ]
        cluster_logprobs.append(logsumexp(cluster_values))

    entropy = 0.0
    for cluster_logprob in cluster_logprobs:
        probability = math.exp(cluster_logprob)
        if probability:
            entropy -= probability * cluster_logprob
    return entropy


def normalize_entropy_value(entropy: float, *, num_samples: int) -> float:
    if num_samples <= 1:
        return 0.0
    return entropy / math.log(num_samples)
