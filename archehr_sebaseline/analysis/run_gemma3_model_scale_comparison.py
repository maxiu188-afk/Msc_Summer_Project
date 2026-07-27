#!/usr/bin/env python3
"""Compare aligned Gemma 3 4B and 12B Phase-1 UQ ranking regimes."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-4b", type=Path, required=True)
    parser.add_argument("--run-12b", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260727)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv_by_id(path: Path, *, id_field: str = "example_id") -> dict[str, dict[str, str]]:
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
            "main_answer_words": main_word_count(str(best[example_id]["clean_answer"])),
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
        "examples": examples_rows,
        "records": records,
        "valid_labels": valid_labels,
        "labels_path": labels_path,
    }


def metric(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    values = binary_ranking_metrics(labels, scores)
    if values["auroc"] is None or values["average_precision"] is None:
        raise ValueError("Ranking metrics require both correctness classes.")
    return {
        "auroc": float(values["auroc"]),
        "average_precision": float(values["average_precision"]),
    }


def interval(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))


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
        raise FileExistsError(f"Refusing to use non-empty output directory: {args.output_dir}")
    run4 = load_run(args.run_4b, model="gemma3_4b")
    run12 = load_run(args.run_12b, model="gemma3_12b")
    if run4["examples"] != run12["examples"]:
        raise ValueError("4B and 12B examples.jsonl records are not identical.")
    common_ids = sorted(set(run4["valid_labels"]).intersection(run12["valid_labels"]))
    if len(common_ids) < 2:
        raise ValueError("No usable paired common-label cohort.")
    types = sorted(
        {str(run4["records"][example_id]["bioasq_type"]) for example_id in common_ids}
    )
    scopes = ["overall", *types]

    metric_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    scope_summaries: dict[str, Any] = {}
    rng = np.random.default_rng(args.bootstrap_seed)
    for scope in scopes:
        ids = [
            example_id
            for example_id in common_ids
            if scope == "overall"
            or run4["records"][example_id]["bioasq_type"] == scope
        ]
        labels = {
            "gemma3_4b": np.asarray(
                [run4["valid_labels"][example_id] for example_id in ids],
                dtype=np.int64,
            ),
            "gemma3_12b": np.asarray(
                [run12["valid_labels"][example_id] for example_id in ids],
                dtype=np.int64,
            ),
        }
        arrays: dict[str, dict[str, np.ndarray]] = {}
        for run in (run4, run12):
            model = str(run["model"])
            arrays[model] = {
                score_name: np.asarray(
                    [run["records"][example_id][score_name] for example_id in ids],
                    dtype=np.float64,
                )
                for score_name in SCORES
            }
            for score_name, scores in arrays[model].items():
                values = metric(labels[model], scores)
                metric_rows.append(
                    {
                        "scope": scope,
                        "model": model,
                        "examples": len(ids),
                        "incorrect": int(labels[model].sum()),
                        "accuracy": float(1.0 - labels[model].mean()),
                        "score": score_name,
                        **values,
                    }
                )
            length_rows.append(
                {
                    "scope": scope,
                    "model": model,
                    "examples": len(ids),
                    "mean_main_answer_words": float(
                        np.mean(
                            [
                                run["records"][example_id]["main_answer_words"]
                                for example_id in ids
                            ]
                        )
                    ),
                    "mean_sample_answer_tokens": float(
                        np.mean(
                            [
                                run["records"][example_id]["mean_sample_tokens"]
                                for example_id in ids
                            ]
                        )
                    ),
                }
            )
        observed_metrics = {
            model: {
                score_name: metric(labels[model], scores)["auroc"]
                for score_name, scores in model_arrays.items()
            }
            for model, model_arrays in arrays.items()
        }
        observed_gaps = {
            model: (
                observed_metrics[model]["blind_p_true"]
                - observed_metrics[model]["discrete_semantic_entropy"]
            )
            for model in ("gemma3_4b", "gemma3_12b")
        }
        observed = {
            "ptrue_minus_se_gemma3_4b": observed_gaps["gemma3_4b"],
            "ptrue_minus_se_gemma3_12b": observed_gaps["gemma3_12b"],
            "gap_change_4b_minus_12b": (
                observed_gaps["gemma3_4b"] - observed_gaps["gemma3_12b"]
            ),
        }
        strata: dict[tuple[int, int], np.ndarray] = {}
        for label_pair in sorted(set(zip(labels["gemma3_4b"], labels["gemma3_12b"]))):
            mask = np.flatnonzero(
                (labels["gemma3_4b"] == label_pair[0])
                & (labels["gemma3_12b"] == label_pair[1])
            )
            strata[(int(label_pair[0]), int(label_pair[1]))] = mask
        bootstrap = {name: [] for name in observed}
        for _ in range(args.bootstrap_resamples):
            sampled = np.concatenate(
                [
                    rng.choice(indices, size=len(indices), replace=True)
                    for indices in strata.values()
                ]
            )
            sampled_gaps: dict[str, float] = {}
            for model in ("gemma3_4b", "gemma3_12b"):
                ptrue_auc = metric(
                    labels[model][sampled],
                    arrays[model]["blind_p_true"][sampled],
                )["auroc"]
                se_auc = metric(
                    labels[model][sampled],
                    arrays[model]["discrete_semantic_entropy"][sampled],
                )["auroc"]
                sampled_gaps[model] = ptrue_auc - se_auc
            bootstrap["ptrue_minus_se_gemma3_4b"].append(
                sampled_gaps["gemma3_4b"]
            )
            bootstrap["ptrue_minus_se_gemma3_12b"].append(
                sampled_gaps["gemma3_12b"]
            )
            bootstrap["gap_change_4b_minus_12b"].append(
                sampled_gaps["gemma3_4b"] - sampled_gaps["gemma3_12b"]
            )
        for comparison, value in observed.items():
            lower, upper = interval(bootstrap[comparison])
            bootstrap_rows.append(
                {
                    "scope": scope,
                    "comparison": comparison,
                    "observed": value,
                    "ci_lower_95": lower,
                    "ci_upper_95": upper,
                    "resamples": args.bootstrap_resamples,
                    "seed": args.bootstrap_seed,
                    "joint_label_stratified": True,
                }
            )
        agreement = float(
            np.mean(labels["gemma3_4b"] == labels["gemma3_12b"])
        )
        scope_summaries[scope] = {
            "examples": len(ids),
            "label_pair_counts": {
                f"4b_{pair[0]}_12b_{pair[1]}": int(len(indices))
                for pair, indices in strata.items()
            },
            "correctness_label_agreement": agreement,
            "observed_auroc": observed_metrics,
            "observed_gaps": observed,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "uq_metrics.csv", metric_rows)
    write_csv(args.output_dir / "answer_lengths.csv", length_rows)
    write_csv(args.output_dir / "paired_bootstrap.csv", bootstrap_rows)
    summary = {
        "schema_version": "gemma3_4b_12b_phase1_model_scale_v1",
        "purpose": (
            "Compare the blind-P(True)-versus-SE ranking regime across aligned "
            "Gemma 3 sizes; not a Probe experiment."
        ),
        "positive_class": "Claude incorrect",
        "paired_common_valid_questions": len(common_ids),
        "valid_labels": {
            "gemma3_4b": len(run4["valid_labels"]),
            "gemma3_12b": len(run12["valid_labels"]),
        },
        "scope_summaries": scope_summaries,
        "bootstrap": {
            "resamples": args.bootstrap_resamples,
            "seed": args.bootstrap_seed,
            "joint_correctness_label_stratified": True,
        },
        "input_sha256": {
            "4b_examples": sha256_file(args.run_4b / "examples.jsonl"),
            "12b_examples": sha256_file(args.run_12b / "examples.jsonl"),
            "4b_labels": sha256_file(run4["labels_path"]),
            "12b_labels": sha256_file(run12["labels_path"]),
            "4b_se": sha256_file(args.run_4b / "se_scores.csv"),
            "12b_se": sha256_file(args.run_12b / "se_scores.csv"),
            "4b_self_report": sha256_file(
                args.run_4b / "uq_baselines" / "self_report_examples.csv"
            ),
            "12b_self_report": sha256_file(
                args.run_12b / "uq_baselines" / "self_report_examples.csv"
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"completed 4B/12B model-scale comparison on {len(common_ids)} paired questions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
