#!/usr/bin/env python3
"""Analyze the bounded Claude-versus-PubMedBERT summary clustering diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.claude_semantic_clustering import (
    transitivity_violation_count,
)
from archehr_sebaseline.entropy import semantic_entropy_from_cluster_sizes
from archehr_sebaseline.phase2_probe import binary_ranking_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260726)
    parser.add_argument("--manual-review-size", type=int, default=24)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Analyze the validated complete-case subset when Claude results are incomplete.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjusted_rand_index(labels_a: list[int], labels_b: list[int]) -> float:
    """Calculate adjusted Rand index without a scikit-learn dependency."""

    if len(labels_a) != len(labels_b) or len(labels_a) < 2:
        raise ValueError("ARI requires equal label vectors with at least two items.")
    contingency = Counter(zip(labels_a, labels_b))
    row_sums = Counter(labels_a)
    column_sums = Counter(labels_b)
    choose2 = lambda value: value * (value - 1) / 2
    index = sum(choose2(value) for value in contingency.values())
    row_index = sum(choose2(value) for value in row_sums.values())
    column_index = sum(choose2(value) for value in column_sums.values())
    total_pairs = choose2(len(labels_a))
    expected = row_index * column_index / total_pairs
    maximum = 0.5 * (row_index + column_index)
    if math.isclose(maximum, expected):
        return 1.0
    return float((index - expected) / (maximum - expected))


def pair_cell(nli_same: bool, claude_same: bool) -> str:
    if nli_same and claude_same:
        return "agree_same"
    if not nli_same and not claude_same:
        return "agree_different"
    if nli_same:
        return "nli_same_claude_different"
    return "nli_different_claude_same"


def finite_float(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return number


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    result = binary_ranking_metrics(labels, scores)
    return {
        "auroc": float(result["auroc"]),
        "average_precision": float(result["average_precision"]),
    }


def interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def select_manual_pairs(
    pair_rows: list[dict[str, Any]], *, size: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if size <= 0:
        raise ValueError("manual review size must be positive.")
    targets = {
        "nli_same_claude_different": size // 3,
        "nli_different_claude_same": size // 3,
        "agree_same": size // 6,
        "agree_different": size - 2 * (size // 3) - size // 6,
    }
    ranked_by_cell: dict[str, list[dict[str, Any]]] = {}
    for cell in targets:
        ranked_by_cell[cell] = sorted(
            (row for row in pair_rows if row["comparison_cell"] == cell),
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['pair_id']}".encode("utf-8")
            ).hexdigest(),
        )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for cell, target in targets.items():
        for row in ranked_by_cell[cell][:target]:
            selected.append(row)
            selected_ids.add(str(row["pair_id"]))
    if len(selected) < size:
        remainder = sorted(
            (row for row in pair_rows if str(row["pair_id"]) not in selected_ids),
            key=lambda row: hashlib.sha256(
                f"{seed}:fill:{row['pair_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(remainder[: size - len(selected)])
    selected.sort(key=lambda row: str(row["pair_id"]))
    blinded = [
        {
            "pair_id": row["pair_id"],
            "question": row["question"],
            "answer_a": row["answer_a"],
            "answer_b": row["answer_b"],
            "human_equivalent_1_or_0": "",
            "human_notes": "",
        }
        for row in selected
    ]
    key = [
        {
            "pair_id": row["pair_id"],
            "comparison_cell": row["comparison_cell"],
            "nli_same_cluster": row["nli_same_cluster"],
            "claude_same_cluster": row["claude_same_cluster"],
            "claude_direct_equivalent": row["claude_direct_equivalent"],
        }
        for row in selected
    ]
    return blinded, key


def main() -> int:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cohort = read_jsonl(args.diagnostic_dir / "cohort_manifest.jsonl")
    cohort_by_id = {str(row["example_id"]): row for row in cohort}
    frozen_cohort_ids = set(cohort_by_id)
    if len(frozen_cohort_ids) < 2 or len(frozen_cohort_ids) % 2:
        raise ValueError("Frozen cohort must have a positive even number of questions.")
    frozen_labels = Counter(int(row["incorrect"]) for row in cohort)
    expected_per_class = len(frozen_cohort_ids) // 2
    if frozen_labels != Counter({0: expected_per_class, 1: expected_per_class}):
        raise ValueError("Expected an exactly label-balanced frozen cohort.")

    nli_all = {
        str(row["example_id"]): row
        for row in read_jsonl(args.collection_dir / "current" / "clusters.jsonl")
    }
    claude_all = {
        str(row["example_id"]): row
        for row in read_jsonl(args.diagnostic_dir / "claude_clusters.jsonl")
    }
    pairwise_all = {
        str(row["example_id"]): row
        for row in read_jsonl(args.diagnostic_dir / "claude_pairwise_decisions.jsonl")
    }
    claude_ids = set(claude_all)
    pairwise_ids = set(pairwise_all)
    if claude_ids != pairwise_ids:
        raise ValueError("Claude cluster and pairwise result IDs differ.")
    if not claude_ids.issubset(frozen_cohort_ids):
        raise ValueError("Claude results contain IDs outside the frozen cohort.")
    if claude_ids != frozen_cohort_ids and not args.allow_incomplete:
        raise ValueError("Claude clustering results must cover the full frozen cohort.")
    cohort_ids = sorted(claude_ids)
    labels = np.asarray(
        [int(cohort_by_id[example_id]["incorrect"]) for example_id in cohort_ids],
        dtype=np.int64,
    )
    label_counts = Counter(labels.tolist())
    if set(label_counts) != {0, 1}:
        raise ValueError("Complete-case results must retain both correctness classes.")
    missing_ids = sorted(frozen_cohort_ids.difference(claude_ids))
    examples = {
        str(row["id"]): row
        for row in read_jsonl(args.collection_dir / "current" / "examples.jsonl")
    }
    generations: dict[str, dict[int, str]] = {}
    for row in read_jsonl(
        args.collection_dir / "current" / "cleaned_generations.jsonl"
    ):
        generations.setdefault(str(row["example_id"]), {})[
            int(row["sample_id"])
        ] = str(row["clean_answer"])
    scores = {
        str(row["example_id"]): row
        for row in read_csv(args.collection_dir / "current" / "condition_scores.csv")
    }

    per_question: list[dict[str, Any]] = []
    all_pairs: list[dict[str, Any]] = []
    score_arrays: dict[str, list[float]] = {
        "blind_p_true": [],
        "nli_semantic_entropy": [],
        "claude_semantic_entropy": [],
        "nli_num_clusters": [],
        "claude_num_clusters": [],
    }
    for example_id in cohort_ids:
        nli = nli_all[example_id]
        claude = claude_all[example_id]
        pairwise = pairwise_all[example_id]
        nli_ids = [int(value) for value in nli["semantic_ids"]]
        claude_ids = [int(value) for value in claude["semantic_ids"]]
        if len(nli_ids) != 10 or len(claude_ids) != 10:
            raise ValueError(f"{example_id} does not have ten cluster assignments.")
        nli_sizes = [int(value) for value in nli["cluster_sizes"]]
        claude_sizes = [int(value) for value in claude["cluster_sizes"]]
        if sum(nli_sizes) != 10 or sum(claude_sizes) != 10:
            raise ValueError(f"{example_id} cluster sizes do not sum to ten.")
        nli_se = semantic_entropy_from_cluster_sizes(nli_sizes)
        claude_se = semantic_entropy_from_cluster_sizes(claude_sizes)
        score_arrays["blind_p_true"].append(
            finite_float(scores[example_id]["blind_p_true_uncertainty"], "blind PTrue")
        )
        score_arrays["nli_semantic_entropy"].append(nli_se)
        score_arrays["claude_semantic_entropy"].append(claude_se)
        score_arrays["nli_num_clusters"].append(float(len(nli_sizes)))
        score_arrays["claude_num_clusters"].append(float(len(claude_sizes)))

        pair_counts: Counter[str] = Counter()
        direct_by_pair = {
            (int(row["left_sample_id"]), int(row["right_sample_id"])): bool(
                row["equivalent"]
            )
            for row in pairwise["decisions"]
        }
        for left, right in combinations(range(10), 2):
            nli_same = nli_ids[left] == nli_ids[right]
            claude_same = claude_ids[left] == claude_ids[right]
            cell = pair_cell(nli_same, claude_same)
            pair_counts[cell] += 1
            all_pairs.append(
                {
                    "pair_id": f"{example_id}-s{left}-s{right}",
                    "example_id": example_id,
                    "left_sample_id": left,
                    "right_sample_id": right,
                    "question": str(examples[example_id]["question"]),
                    "answer_a": generations[example_id][left],
                    "answer_b": generations[example_id][right],
                    "comparison_cell": cell,
                    "nli_same_cluster": int(nli_same),
                    "claude_same_cluster": int(claude_same),
                    "claude_direct_equivalent": int(direct_by_pair[(left, right)]),
                }
            )
        per_question.append(
            {
                "example_id": example_id,
                "incorrect": int(cohort_by_id[example_id]["incorrect"]),
                "blind_p_true_uncertainty": score_arrays["blind_p_true"][-1],
                "nli_semantic_entropy": nli_se,
                "claude_semantic_entropy": claude_se,
                "nli_num_clusters": len(nli_sizes),
                "claude_num_clusters": len(claude_sizes),
                "nli_largest_cluster_fraction": max(nli_sizes) / 10,
                "claude_largest_cluster_fraction": max(claude_sizes) / 10,
                "adjusted_rand_index": adjusted_rand_index(nli_ids, claude_ids),
                "pair_agreement_rate": (
                    pair_counts["agree_same"] + pair_counts["agree_different"]
                )
                / 45,
                "nli_same_claude_different_pairs": pair_counts[
                    "nli_same_claude_different"
                ],
                "nli_different_claude_same_pairs": pair_counts[
                    "nli_different_claude_same"
                ],
                "claude_direct_transitivity_violations": transitivity_violation_count(
                    str(pairwise["bitstring"])
                ),
            }
        )

    arrays = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in score_arrays.items()
    }
    metric_rows: list[dict[str, Any]] = []
    observed_metrics: dict[str, dict[str, float]] = {}
    for name, values in arrays.items():
        observed_metrics[name] = metric(labels, values)
        metric_rows.append(
            {
                "method": name,
                "examples": len(labels),
                "incorrect": int(labels.sum()),
                **observed_metrics[name],
            }
        )

    observed = {
        "se_auroc_claude_minus_nli": (
            observed_metrics["claude_semantic_entropy"]["auroc"]
            - observed_metrics["nli_semantic_entropy"]["auroc"]
        ),
        "cluster_count_auroc_claude_minus_nli": (
            observed_metrics["claude_num_clusters"]["auroc"]
            - observed_metrics["nli_num_clusters"]["auroc"]
        ),
        "ptrue_minus_nli_se_auroc": (
            observed_metrics["blind_p_true"]["auroc"]
            - observed_metrics["nli_semantic_entropy"]["auroc"]
        ),
        "ptrue_minus_claude_se_auroc": (
            observed_metrics["blind_p_true"]["auroc"]
            - observed_metrics["claude_semantic_entropy"]["auroc"]
        ),
    }
    rng = np.random.default_rng(args.bootstrap_seed)
    correct_indices = np.flatnonzero(labels == 0)
    incorrect_indices = np.flatnonzero(labels == 1)
    bootstrap = {name: [] for name in observed}
    for _ in range(args.bootstrap_resamples):
        sampled = np.concatenate(
            [
                rng.choice(correct_indices, size=len(correct_indices), replace=True),
                rng.choice(incorrect_indices, size=len(incorrect_indices), replace=True),
            ]
        )
        sampled_labels = labels[sampled]
        sampled_metrics = {
            name: metric(sampled_labels, values[sampled])["auroc"]
            for name, values in arrays.items()
        }
        bootstrap["se_auroc_claude_minus_nli"].append(
            sampled_metrics["claude_semantic_entropy"]
            - sampled_metrics["nli_semantic_entropy"]
        )
        bootstrap["cluster_count_auroc_claude_minus_nli"].append(
            sampled_metrics["claude_num_clusters"]
            - sampled_metrics["nli_num_clusters"]
        )
        bootstrap["ptrue_minus_nli_se_auroc"].append(
            sampled_metrics["blind_p_true"]
            - sampled_metrics["nli_semantic_entropy"]
        )
        bootstrap["ptrue_minus_claude_se_auroc"].append(
            sampled_metrics["blind_p_true"]
            - sampled_metrics["claude_semantic_entropy"]
        )
    bootstrap_rows = []
    for name, value in observed.items():
        lower, upper = interval(bootstrap[name])
        bootstrap_rows.append(
            {
                "metric": name,
                "observed": value,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "label_stratified": True,
            }
        )

    blinded, review_key = select_manual_pairs(
        all_pairs, size=args.manual_review_size, seed=args.bootstrap_seed
    )
    write_csv(args.output_dir / "per_question.csv", per_question)
    write_csv(args.output_dir / "uq_metrics.csv", metric_rows)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / "manual_review_blinded.csv", blinded)
    write_csv(args.output_dir / "manual_review_key.csv", review_key)

    aggregate_pair_counts = Counter(
        str(row["comparison_cell"]) for row in all_pairs
    )
    merged_results_path = args.diagnostic_dir / "claude_pair_results_merged.jsonl"
    if not merged_results_path.exists():
        merged_results_path = args.diagnostic_dir / "claude_pair_results.jsonl"
    summary = {
        "schema_version": "bioasq_summary_clustering_diagnostic_complete_case_v2",
        "scope": (
            f"{len(cohort_ids)} validated complete-case questions from the frozen "
            "48-question current long-summary cohort"
        ),
        "frozen_questions": len(frozen_cohort_ids),
        "analyzed_questions": len(cohort_ids),
        "analyzed_label_counts": {
            "correct": int(label_counts[0]),
            "incorrect": int(label_counts[1]),
        },
        "missing_example_ids": missing_ids,
        "complete_case_limit": (
            "One or more frozen questions lacked a valid Claude decision set; "
            "complete-case results are descriptive and may be affected by "
            "non-random output truncation."
            if missing_ids
            else ""
        ),
        "positive_class": "Claude correctness judge incorrect",
        "metric_rows": observed_metrics,
        "observed_differences": observed,
        "cluster_agreement": {
            "mean_adjusted_rand_index": float(
                np.mean([row["adjusted_rand_index"] for row in per_question])
            ),
            "mean_pair_agreement_rate": float(
                np.mean([row["pair_agreement_rate"] for row in per_question])
            ),
            "pair_cells": dict(aggregate_pair_counts),
            "claude_direct_transitivity_violations": int(
                sum(
                    int(row["claude_direct_transitivity_violations"])
                    for row in per_question
                )
            ),
        },
        "cluster_distribution": {
            "nli_mean_num_clusters": float(np.mean(arrays["nli_num_clusters"])),
            "claude_mean_num_clusters": float(
                np.mean(arrays["claude_num_clusters"])
            ),
            "nli_mean_largest_cluster_fraction": float(
                np.mean(
                    [row["nli_largest_cluster_fraction"] for row in per_question]
                )
            ),
            "claude_mean_largest_cluster_fraction": float(
                np.mean(
                    [row["claude_largest_cluster_fraction"] for row in per_question]
                )
            ),
        },
        "manual_review": {
            "status": "pending_human_review",
            "pairs": len(blinded),
            "blinded_worksheet": "manual_review_blinded.csv",
            "sealed_comparison_key": "manual_review_key.csv",
            "interpretation_boundary": (
                "Do not attribute any UQ change to better semantic equivalence until "
                "the blinded worksheet has been reviewed by a human."
            ),
        },
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "label_stratified_paired_question_resampling": True,
        },
        "input_sha256": {
            "protocol": sha256_file(args.diagnostic_dir / "protocol.json"),
            "cohort": sha256_file(args.diagnostic_dir / "cohort_manifest.jsonl"),
            "claude_results": sha256_file(merged_results_path),
            "claude_clusters": sha256_file(
                args.diagnostic_dir / "claude_clusters.jsonl"
            ),
            "claude_pairwise": sha256_file(
                args.diagnostic_dir / "claude_pairwise_decisions.jsonl"
            ),
            "nli_clusters": sha256_file(
                args.collection_dir / "current" / "clusters.jsonl"
            ),
            "condition_scores": sha256_file(
                args.collection_dir / "current" / "condition_scores.csv"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"completed bounded clustering diagnostic: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
