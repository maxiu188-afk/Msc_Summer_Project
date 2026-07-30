#!/usr/bin/env python3
"""Evaluate the staged Gemma 3 1B Phase-1-aligned scale extension.

The 200 summary questions form the formal three-model comparison.  The
50-question factoid and list subsets are accuracy-only feasibility gates and
must not be interpreted as complete model-scale UQ experiments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_probe import binary_ranking_metrics


SCORES = {
    "blind_p_true": ("self_report", "p_true_blind_uncertainty"),
    "discrete_semantic_entropy": ("se", "discrete_semantic_entropy"),
    "num_clusters": ("se", "num_clusters"),
    "normalized_nll": ("example_uq", "mean_normalized_nll"),
}
MODELS = ("gemma3_1b", "gemma3_4b", "gemma3_12b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-1b", type=Path, required=True)
    parser.add_argument("--run-4b", type=Path, required=True)
    parser.add_argument("--run-12b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv_by_id(
    path: Path, *, id_field: str = "example_id"
) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))
    result = {str(row[id_field]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate {id_field} in {path}.")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(ids: list[str]) -> str:
    return hashlib.sha256(
        json.dumps(ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return result


def main_word_count(answer: str) -> int:
    return len(re.findall(r"\S+", answer.strip()))


def load_run(run_dir: Path, *, model: str) -> dict[str, Any]:
    examples_rows = read_jsonl(run_dir / "examples.jsonl")
    examples = {str(row["id"]): row for row in examples_rows}
    if len(examples) != len(examples_rows):
        raise ValueError(f"Duplicate examples in {run_dir}.")
    best_rows = read_jsonl(run_dir / "best_generations.jsonl")
    best = {str(row["example_id"]): row for row in best_rows}
    if set(best) != set(examples):
        raise ValueError(f"Best-generation IDs differ from examples in {run_dir}.")
    sources = {
        "se": read_csv_by_id(run_dir / "se_scores.csv"),
        "example_uq": read_csv_by_id(run_dir / "example_uq.csv"),
        "self_report": read_csv_by_id(
            run_dir / "uq_baselines" / "self_report_examples.csv"
        ),
    }
    for name, rows in sources.items():
        if set(rows) != set(examples):
            raise ValueError(f"{name} IDs differ from examples in {run_dir}.")
    labels_path = (
        run_dir
        / "claude_binary_main_answer_judge"
        / "claude_generation_labels.csv"
    )
    labels_rows = read_csv_by_id(labels_path)
    if set(labels_rows) != set(examples):
        raise ValueError(f"Label IDs differ from examples in {run_dir}.")
    valid_labels: dict[str, int] = {}
    for example_id, row in labels_rows.items():
        if (
            str(row.get("label_valid") or "").lower() != "true"
            or str(row.get("label") or "") not in {"correct", "incorrect"}
        ):
            continue
        valid_labels[example_id] = int(row["label"] == "incorrect")
    records: dict[str, dict[str, Any]] = {}
    for example_id, example in examples.items():
        record: dict[str, Any] = {
            "example_id": example_id,
            "bioasq_type": str(example["bioasq_type"]).lower(),
            "label": valid_labels.get(example_id),
            "main_answer_words": main_word_count(
                str(best[example_id]["clean_answer"])
            ),
            "mean_sample_tokens": finite_float(
                sources["example_uq"][example_id]["mean_num_generated_tokens"],
                name=f"{model} mean sample tokens {example_id}",
            ),
        }
        for score_name, (source_name, field) in SCORES.items():
            record[score_name] = finite_float(
                sources[source_name][example_id][field],
                name=f"{model} {score_name} {example_id}",
            )
        records[example_id] = record
    return {
        "model": model,
        "run_dir": run_dir,
        "examples": examples,
        "example_rows": examples_rows,
        "records": records,
        "valid_labels": valid_labels,
        "labels_path": labels_path,
    }


def validate_alignment(
    run1: dict[str, Any], run4: dict[str, Any], run12: dict[str, Any]
) -> None:
    if run4["example_rows"] != run12["example_rows"]:
        raise ValueError("4B and 12B examples.jsonl records are not identical.")
    missing = sorted(set(run1["examples"]) - set(run4["examples"]))
    if missing:
        raise ValueError(f"1B IDs are absent from the accepted cohort: {missing[:5]}")
    mismatched = [
        example_id
        for example_id, record in run1["examples"].items()
        if record != run4["examples"][example_id]
    ]
    if mismatched:
        raise ValueError(
            "1B example records differ from the accepted cohort: "
            f"{mismatched[:5]}"
        )
    counts = Counter(
        str(row["bioasq_type"]).lower() for row in run1["example_rows"]
    )
    expected = {"factoid": 50, "list": 50, "summary": 200}
    if counts != expected:
        raise ValueError(f"Unexpected staged 1B type counts: {dict(counts)}.")


def ranking_metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    values = binary_ranking_metrics(labels, scores)
    if values["auroc"] is None or values["average_precision"] is None:
        raise ValueError("Ranking metrics require both correctness classes.")
    return {
        "auroc": float(values["auroc"]),
        "average_precision": float(values["average_precision"]),
    }


def percentile_interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation.")
    if not 0 <= successes <= total:
        raise ValueError("Successes must be between zero and total.")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return center - margin, center + margin


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}.")
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.bootstrap_resamples <= 0:
        raise ValueError("--bootstrap-resamples must be positive.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to use non-empty output directory: {args.output_dir}"
        )
    runs = {
        "gemma3_1b": load_run(args.run_1b, model="gemma3_1b"),
        "gemma3_4b": load_run(args.run_4b, model="gemma3_4b"),
        "gemma3_12b": load_run(args.run_12b, model="gemma3_12b"),
    }
    validate_alignment(
        runs["gemma3_1b"], runs["gemma3_4b"], runs["gemma3_12b"]
    )

    feasibility_rows: list[dict[str, Any]] = []
    for question_type in ("factoid", "list"):
        submitted_ids = sorted(
            example_id
            for example_id, record in runs["gemma3_1b"]["records"].items()
            if record["bioasq_type"] == question_type
        )
        valid_ids = [
            example_id
            for example_id in submitted_ids
            if example_id in runs["gemma3_1b"]["valid_labels"]
        ]
        correct = sum(
            1 - runs["gemma3_1b"]["valid_labels"][example_id]
            for example_id in valid_ids
        )
        lower, upper = wilson_interval(correct, len(valid_ids))
        feasibility_rows.append(
            {
                "bioasq_type": question_type,
                "submitted": len(submitted_ids),
                "valid_labels": len(valid_ids),
                "invalid_labels": len(submitted_ids) - len(valid_ids),
                "correct": correct,
                "incorrect": len(valid_ids) - correct,
                "accuracy": correct / len(valid_ids),
                "accuracy_ci_lower_95_wilson": lower,
                "accuracy_ci_upper_95_wilson": upper,
                "full_expansion_decision": "not_automated",
            }
        )

    summary_submitted_ids = sorted(
        example_id
        for example_id, record in runs["gemma3_1b"]["records"].items()
        if record["bioasq_type"] == "summary"
    )
    common_ids = [
        example_id
        for example_id in summary_submitted_ids
        if all(example_id in runs[model]["valid_labels"] for model in MODELS)
    ]
    if len(common_ids) < 2:
        raise ValueError("No usable three-model common-label summary cohort.")
    labels = {
        model: np.asarray(
            [runs[model]["valid_labels"][example_id] for example_id in common_ids],
            dtype=np.int64,
        )
        for model in MODELS
    }
    for model, values in labels.items():
        if len(np.unique(values)) != 2:
            raise ValueError(f"{model} summary labels contain only one class.")
    score_arrays = {
        model: {
            score_name: np.asarray(
                [
                    runs[model]["records"][example_id][score_name]
                    for example_id in common_ids
                ],
                dtype=np.float64,
            )
            for score_name in SCORES
        }
        for model in MODELS
    }

    metric_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    observed_aurocs: dict[str, dict[str, float]] = {}
    for model in MODELS:
        observed_aurocs[model] = {}
        for score_name, scores in score_arrays[model].items():
            values = ranking_metric(labels[model], scores)
            observed_aurocs[model][score_name] = values["auroc"]
            metric_rows.append(
                {
                    "scope": "summary",
                    "model": model,
                    "examples": len(common_ids),
                    "incorrect": int(labels[model].sum()),
                    "accuracy": float(1.0 - labels[model].mean()),
                    "score": score_name,
                    **values,
                }
            )
        length_rows.append(
            {
                "scope": "summary",
                "model": model,
                "examples": len(common_ids),
                "mean_main_answer_words": float(
                    np.mean(
                        [
                            runs[model]["records"][example_id][
                                "main_answer_words"
                            ]
                            for example_id in common_ids
                        ]
                    )
                ),
                "mean_sample_answer_tokens": float(
                    np.mean(
                        [
                            runs[model]["records"][example_id][
                                "mean_sample_tokens"
                            ]
                            for example_id in common_ids
                        ]
                    )
                ),
            }
        )

    observed_gaps = {
        model: (
            observed_aurocs[model]["blind_p_true"]
            - observed_aurocs[model]["discrete_semantic_entropy"]
        )
        for model in MODELS
    }
    observed = {
        **{f"ptrue_minus_se_{model}": observed_gaps[model] for model in MODELS},
        "gap_change_1b_minus_4b": (
            observed_gaps["gemma3_1b"] - observed_gaps["gemma3_4b"]
        ),
        "gap_change_4b_minus_12b": (
            observed_gaps["gemma3_4b"] - observed_gaps["gemma3_12b"]
        ),
        "gap_change_1b_minus_12b": (
            observed_gaps["gemma3_1b"] - observed_gaps["gemma3_12b"]
        ),
    }
    strata: dict[tuple[int, int, int], np.ndarray] = {}
    label_tuples = zip(*(labels[model] for model in MODELS))
    for label_tuple in sorted(set(label_tuples)):
        mask = np.flatnonzero(
            np.logical_and.reduce(
                [
                    labels[model] == label_tuple[index]
                    for index, model in enumerate(MODELS)
                ]
            )
        )
        strata[tuple(int(value) for value in label_tuple)] = mask
    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap = {name: [] for name in observed}
    for _ in range(args.bootstrap_resamples):
        sampled = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in strata.values()
            ]
        )
        sampled_gaps: dict[str, float] = {}
        for model in MODELS:
            ptrue_auc = ranking_metric(
                labels[model][sampled],
                score_arrays[model]["blind_p_true"][sampled],
            )["auroc"]
            se_auc = ranking_metric(
                labels[model][sampled],
                score_arrays[model]["discrete_semantic_entropy"][sampled],
            )["auroc"]
            sampled_gaps[model] = ptrue_auc - se_auc
            bootstrap[f"ptrue_minus_se_{model}"].append(sampled_gaps[model])
        bootstrap["gap_change_1b_minus_4b"].append(
            sampled_gaps["gemma3_1b"] - sampled_gaps["gemma3_4b"]
        )
        bootstrap["gap_change_4b_minus_12b"].append(
            sampled_gaps["gemma3_4b"] - sampled_gaps["gemma3_12b"]
        )
        bootstrap["gap_change_1b_minus_12b"].append(
            sampled_gaps["gemma3_1b"] - sampled_gaps["gemma3_12b"]
        )
    bootstrap_rows: list[dict[str, Any]] = []
    for comparison, value in observed.items():
        lower, upper = percentile_interval(bootstrap[comparison])
        bootstrap_rows.append(
            {
                "scope": "summary",
                "comparison": comparison,
                "observed": value,
                "ci_lower_95": lower,
                "ci_upper_95": upper,
                "resamples": args.bootstrap_resamples,
                "seed": args.bootstrap_seed,
                "joint_label_stratified": True,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary_uq_metrics.csv", metric_rows)
    write_csv(args.output_dir / "summary_answer_lengths.csv", length_rows)
    write_csv(args.output_dir / "summary_paired_bootstrap.csv", bootstrap_rows)
    write_csv(args.output_dir / "factoid_list_feasibility.csv", feasibility_rows)
    summary = {
        "schema_version": "gemma3_1b_staged_phase1_model_scale_v1",
        "purpose": (
            "Formal aligned 1B/4B/12B summary comparison plus accuracy-only "
            "1B factoid/list feasibility gates; not a Probe experiment."
        ),
        "positive_class": "Claude incorrect",
        "summary": {
            "submitted_1b_questions": len(summary_submitted_ids),
            "paired_common_valid_questions": len(common_ids),
            "paired_common_id_sha256": id_sha256(common_ids),
            "valid_labels_in_submitted_summary": {
                model: sum(
                    example_id in runs[model]["valid_labels"]
                    for example_id in summary_submitted_ids
                )
                for model in MODELS
            },
            "label_tuple_counts": {
                "_".join(
                    f"{model.replace('gemma3_', '')}_{label}"
                    for model, label in zip(MODELS, label_tuple)
                ): int(len(indices))
                for label_tuple, indices in strata.items()
            },
            "observed_auroc": observed_aurocs,
            "observed_gaps": observed,
        },
        "factoid_list": feasibility_rows,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "joint_correctness_label_stratified": True,
        },
        "input_sha256": {
            f"{model}_{artifact}": sha256_file(path)
            for model, run_dir in (
                ("1b", args.run_1b),
                ("4b", args.run_4b),
                ("12b", args.run_12b),
            )
            for artifact, path in (
                ("examples", run_dir / "examples.jsonl"),
                (
                    "labels",
                    run_dir
                    / "claude_binary_main_answer_judge"
                    / "claude_generation_labels.csv",
                ),
                ("se", run_dir / "se_scores.csv"),
                (
                    "self_report",
                    run_dir / "uq_baselines" / "self_report_examples.csv",
                ),
            )
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        "completed staged 1B analysis on "
        f"{len(common_ids)} paired summary questions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
