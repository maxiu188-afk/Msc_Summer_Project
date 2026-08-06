#!/usr/bin/env python3
"""Prepare and analyze the bounded PubMedQA-v2 SE temperature study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_artifacts import sha256_file
from archehr_sebaseline.phase2_probe import binary_ranking_metrics


TEMPERATURES = {"low": 0.7, "baseline": 1.0, "high": 1.3}
DEFAULT_SELECTION_SEED = 20260806
DEFAULT_BOOTSTRAP_SEED = 20260806
DEFAULT_BOOTSTRAP_SAMPLES = 20_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create the fixed stratified cohort.")
    prepare.add_argument("--source-analysis-dir", type=Path, required=True)
    prepare.add_argument("--output-manifest", type=Path, required=True)
    prepare.add_argument("--output-summary", type=Path, required=True)
    prepare.add_argument("--sample-size", type=int, default=200)
    prepare.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)

    analyze = subparsers.add_parser("analyze", help="Compare T=0.7/1.0/1.3 arms.")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--baseline-analysis-dir", type=Path, required=True)
    analyze.add_argument("--low-analysis-dir", type=Path, required=True)
    analyze.add_argument("--high-analysis-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--bootstrap-samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES)
    analyze.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as infile:
        return list(csv.DictReader(infile))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _indexed(rows: list[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get("example_id") or ""): row for row in rows}
    if not indexed or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{name} contains an empty or duplicate example_id.")
    return indexed


def _stratum(row: dict[str, str]) -> tuple[int, str]:
    incorrect_text = str(row.get("incorrect") or "")
    gold_label = str(row.get("gold_label") or "").strip().lower()
    if incorrect_text not in {"0", "1"} or gold_label not in {"yes", "no", "maybe"}:
        raise ValueError(
            f"Invalid frozen stratum for {row.get('example_id')}: "
            f"incorrect={incorrect_text!r}, gold_label={gold_label!r}."
        )
    return int(incorrect_text), gold_label


def _allocate_strata(
    full_counts: dict[tuple[int, str], int], sample_size: int
) -> dict[tuple[int, str], int]:
    if sample_size <= 0 or sample_size > sum(full_counts.values()):
        raise ValueError("sample_size must be between 1 and the source cohort size.")
    nonempty = sorted(key for key, count in full_counts.items() if count)
    allocation = {key: 0 for key in nonempty}
    remaining = sample_size
    if sample_size >= len(nonempty):
        for key in nonempty:
            allocation[key] = 1
        remaining -= len(nonempty)
    capacities = {key: full_counts[key] - allocation[key] for key in nonempty}
    capacity_total = sum(capacities.values())
    if remaining and capacity_total <= 0:
        raise ValueError("No capacity remains for the requested sample.")
    ideals = {
        key: (remaining * capacities[key] / capacity_total if capacity_total else 0.0)
        for key in nonempty
    }
    for key in nonempty:
        add = min(capacities[key], int(math.floor(ideals[key])))
        allocation[key] += add
    unassigned = sample_size - sum(allocation.values())
    ranked = sorted(
        nonempty,
        key=lambda key: (-(ideals[key] - math.floor(ideals[key])), key),
    )
    while unassigned:
        progressed = False
        for key in ranked:
            if allocation[key] < full_counts[key]:
                allocation[key] += 1
                unassigned -= 1
                progressed = True
                if not unassigned:
                    break
        if not progressed:
            raise RuntimeError("Could not complete the stratified allocation.")
    return allocation


def _selection_rank(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode("utf-8")).hexdigest()


def prepare_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.source_analysis_dir / "transfer_predictions.csv"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing frozen transfer predictions: {source_path}")
    source_rows = _read_csv(source_path)
    _indexed(source_rows, name="source transfer predictions")
    by_stratum: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        by_stratum[_stratum(row)].append(row)
    full_counts = {key: len(rows) for key, rows in by_stratum.items()}
    allocation = _allocate_strata(full_counts, args.sample_size)
    selected: set[str] = set()
    for key, rows in by_stratum.items():
        ranked = sorted(
            rows,
            key=lambda row: (_selection_rank(str(row["example_id"]), args.selection_seed), str(row["example_id"])),
        )
        selected.update(str(row["example_id"]) for row in ranked[: allocation[key]])
    manifest_rows = []
    for row in source_rows:
        example_id = str(row["example_id"])
        if example_id not in selected:
            continue
        key = _stratum(row)
        manifest_rows.append(
            {
                "example_id": example_id,
                "incorrect": key[0],
                "gold_label": key[1],
                "stratum": f"incorrect_{key[0]}__gold_{key[1]}",
                "selection_seed": args.selection_seed,
                "full_stratum_size": full_counts[key],
                "selected_stratum_size": allocation[key],
                "inclusion_probability": allocation[key] / full_counts[key],
            }
        )
    if len(manifest_rows) != args.sample_size:
        raise RuntimeError(
            f"Expected {args.sample_size} selected rows, got {len(manifest_rows)}."
        )
    _write_csv(args.output_manifest, manifest_rows)
    selected_counts = Counter(row["stratum"] for row in manifest_rows)
    summary = {
        "schema_version": "pubmedqa_v2_se_temperature_manifest_v1",
        "status": "complete",
        "design": "deterministic stratification by frozen incorrect x official gold label",
        "selection_uses_existing_se": False,
        "sample_size": args.sample_size,
        "selection_seed": args.selection_seed,
        "source_rows": len(source_rows),
        "source_predictions_sha256": sha256_file(source_path),
        "manifest_sha256": sha256_file(args.output_manifest),
        "selected_strata": dict(sorted(selected_counts.items())),
    }
    _write_json(args.output_summary, summary)
    return summary


def _finite_float(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return number


def _load_condition(
    *,
    name: str,
    expected_temperature: float,
    analysis_dir: Path,
    manifest_ids: list[str],
    require_exact_ids: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    predictions_path = analysis_dir / "predictions.csv"
    summary_path = analysis_dir / "summary.json"
    for path in (predictions_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} artifact: {path}")
    predictions = _indexed(_read_csv(predictions_path), name=f"{name} predictions")
    manifest_set = set(manifest_ids)
    if require_exact_ids and set(predictions) != manifest_set:
        raise ValueError(f"{name} predictions do not exactly match the manifest IDs.")
    if not manifest_set.issubset(predictions):
        raise ValueError(f"{name} predictions are missing manifest IDs.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    observed_temperature = _finite_float(
        (summary.get("generation") or {}).get("temperature"),
        name=f"{name} temperature",
    )
    if not math.isclose(observed_temperature, expected_temperature, abs_tol=1e-12):
        raise ValueError(
            f"{name} temperature is {observed_temperature}, expected {expected_temperature}."
        )
    if summary.get("prompt_version") != "pubmedqa_context_explanation_v2":
        raise ValueError(f"{name} prompt version drifted.")
    if summary.get("model_name") != "google/gemma-3-12b-it":
        raise ValueError(f"{name} model drifted.")
    generation = summary.get("generation") or {}
    for field, expected in (
        ("num_samples", 10),
        ("seed", 31),
        ("top_p", 0.9),
        ("top_k", 50),
        ("max_new_tokens", 192),
    ):
        if not math.isclose(float(generation.get(field, math.nan)), float(expected), abs_tol=1e-12):
            raise ValueError(f"{name} generation field {field} drifted.")
    clustering = summary.get("clustering") or {}
    if (
        clustering.get("model_name") != "pritamdeka/PubMedBERT-MNLI-MedNLI"
        or clustering.get("method") != "nli_bidirectional_entailment"
        or clustering.get("condition_on_question") is not True
        or clustering.get("strict_entailment") is not False
    ):
        raise ValueError(f"{name} clustering contract drifted.")
    return predictions, summary


def _condition_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [int(row["incorrect"]) for row in rows]
    scores = [float(row["discrete_semantic_entropy"]) for row in rows]
    clusters = [int(row["num_clusters"]) for row in rows]
    ranking = binary_ranking_metrics(labels, scores)
    return {
        "num_examples": len(rows),
        "num_positive": sum(labels),
        "positive_prevalence": sum(labels) / len(labels),
        "auroc": ranking["auroc"],
        "average_precision": ranking["average_precision"],
        "mean_discrete_semantic_entropy": statistics.mean(scores),
        "median_discrete_semantic_entropy": statistics.median(scores),
        "single_cluster_fraction": sum(value == 1 for value in clusters) / len(clusters),
        "mean_num_clusters": statistics.mean(clusters),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a percentile of an empty list.")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_bootstrap(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples <= 0:
        raise ValueError("bootstrap_samples must be positive.")
    if len(baseline_rows) != len(candidate_rows):
        raise ValueError("Paired conditions have different row counts.")
    labels = [int(row["incorrect"]) for row in baseline_rows]
    baseline_scores = [float(row["discrete_semantic_entropy"]) for row in baseline_rows]
    candidate_scores = [float(row["discrete_semantic_entropy"]) for row in candidate_rows]
    baseline_clusters = [int(row["num_clusters"]) for row in baseline_rows]
    candidate_clusters = [int(row["num_clusters"]) for row in candidate_rows]
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        indices = [rng.randrange(len(labels)) for _ in labels]
        sampled_labels = [labels[index] for index in indices]
        sampled_baseline_scores = [baseline_scores[index] for index in indices]
        sampled_candidate_scores = [candidate_scores[index] for index in indices]
        if len(set(sampled_labels)) == 2:
            baseline_ranking = binary_ranking_metrics(sampled_labels, sampled_baseline_scores)
            candidate_ranking = binary_ranking_metrics(sampled_labels, sampled_candidate_scores)
            distributions["auroc"].append(
                float(candidate_ranking["auroc"]) - float(baseline_ranking["auroc"])
            )
            distributions["average_precision"].append(
                float(candidate_ranking["average_precision"])
                - float(baseline_ranking["average_precision"])
            )
        distributions["mean_discrete_semantic_entropy"].append(
            statistics.mean(sampled_candidate_scores)
            - statistics.mean(sampled_baseline_scores)
        )
        distributions["single_cluster_fraction"].append(
            statistics.mean(candidate_clusters[index] == 1 for index in indices)
            - statistics.mean(baseline_clusters[index] == 1 for index in indices)
        )
        distributions["mean_num_clusters"].append(
            statistics.mean(candidate_clusters[index] for index in indices)
            - statistics.mean(baseline_clusters[index] for index in indices)
        )
    baseline_metrics = _condition_metrics(baseline_rows)
    candidate_metrics = _condition_metrics(candidate_rows)
    output = []
    for metric in (
        "auroc",
        "average_precision",
        "mean_discrete_semantic_entropy",
        "single_cluster_fraction",
        "mean_num_clusters",
    ):
        values = distributions[metric]
        output.append(
            {
                "metric": metric,
                "candidate_minus_baseline": candidate_metrics[metric] - baseline_metrics[metric],
                "ci_lower": _percentile(values, 0.025),
                "ci_upper": _percentile(values, 0.975),
                "valid_resamples": len(values),
            }
        )
    return output


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}")
    manifest_rows = _read_csv(args.manifest)
    manifest = _indexed(manifest_rows, name="temperature manifest")
    manifest_ids = [str(row["example_id"]) for row in manifest_rows]
    conditions = {}
    summaries = {}
    for name, directory, require_exact in (
        ("baseline", args.baseline_analysis_dir, False),
        ("low", args.low_analysis_dir, True),
        ("high", args.high_analysis_dir, True),
    ):
        conditions[name], summaries[name] = _load_condition(
            name=name,
            expected_temperature=TEMPERATURES[name],
            analysis_dir=directory,
            manifest_ids=manifest_ids,
            require_exact_ids=require_exact,
        )
    manifest_sha256 = sha256_file(args.manifest)
    for name in ("low", "high"):
        if (summaries[name].get("source_sha256") or {}).get("selected_ids_manifest") != manifest_sha256:
            raise ValueError(f"{name} did not record the exact selected-ID manifest hash.")
    paired_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in conditions}
    per_question = []
    for example_id in manifest_ids:
        manifest_row = manifest[example_id]
        output_row: dict[str, Any] = {
            "example_id": example_id,
            "incorrect": int(manifest_row["incorrect"]),
            "gold_label": manifest_row["gold_label"],
            "stratum": manifest_row["stratum"],
        }
        for name in ("low", "baseline", "high"):
            row = conditions[name][example_id]
            if int(row["incorrect"]) != int(manifest_row["incorrect"]):
                raise ValueError(f"{name} changed the frozen label for {example_id}.")
            if str(row.get("source_v2_gold_label") or "").lower() != manifest_row["gold_label"]:
                raise ValueError(f"{name} changed the official gold label for {example_id}.")
            condition_row = {
                "example_id": example_id,
                "incorrect": int(row["incorrect"]),
                "discrete_semantic_entropy": _finite_float(
                    row["discrete_semantic_entropy"],
                    name=f"{name} SE {example_id}",
                ),
                "num_clusters": int(row["num_clusters"]),
            }
            paired_rows[name].append(condition_row)
            output_row[f"{name}_temperature"] = TEMPERATURES[name]
            output_row[f"{name}_discrete_semantic_entropy"] = condition_row[
                "discrete_semantic_entropy"
            ]
            output_row[f"{name}_num_clusters"] = condition_row["num_clusters"]
        per_question.append(output_row)
    condition_metrics = []
    condition_metric_map = {}
    for name in ("low", "baseline", "high"):
        metrics = _condition_metrics(paired_rows[name])
        condition_metric_map[name] = metrics
        condition_metrics.append({"condition": name, "temperature": TEMPERATURES[name], **metrics})
    differences = []
    for offset, name in enumerate(("low", "high")):
        comparison_rows = _paired_bootstrap(
            baseline_rows=paired_rows["baseline"],
            candidate_rows=paired_rows[name],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + offset,
        )
        for row in comparison_rows:
            differences.append(
                {
                    "comparison": f"{name}_minus_baseline",
                    "candidate_temperature": TEMPERATURES[name],
                    "baseline_temperature": TEMPERATURES["baseline"],
                    **row,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "temperature_metrics.csv", condition_metrics)
    _write_csv(args.output_dir / "paired_differences.csv", differences)
    _write_csv(args.output_dir / "per_question.csv", per_question)
    summary = {
        "schema_version": "pubmedqa_v2_se_temperature_sensitivity_v1",
        "status": "complete",
        "scope": "paired temperature diagnostic on a frozen stratified PubMedQA-v2 subset",
        "temperatures": TEMPERATURES,
        "num_examples": len(manifest_rows),
        "num_samples_per_question": 10,
        "selection_uses_existing_se": False,
        "frozen_target": "accepted context-v2 official-decision incorrect label",
        "condition_metrics": condition_metric_map,
        "paired_differences": differences,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed},
        "source_sha256": {
            "manifest": sha256_file(args.manifest),
            "baseline_predictions": sha256_file(args.baseline_analysis_dir / "predictions.csv"),
            "low_predictions": sha256_file(args.low_analysis_dir / "predictions.csv"),
            "high_predictions": sha256_file(args.high_analysis_dir / "predictions.csv"),
            "baseline_summary": sha256_file(args.baseline_analysis_dir / "summary.json"),
            "low_summary": sha256_file(args.low_analysis_dir / "summary.json"),
            "high_summary": sha256_file(args.high_analysis_dir / "summary.json"),
        },
        "excluded_work": [
            "new_low_temperature_main_answers",
            "new_correctness_labels",
            "prompt_tuning",
            "NLI_change",
            "model_change",
            "P_True_or_Probe_rescoring",
        ],
    }
    _write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    try:
        if args.command == "prepare":
            result = prepare_manifest(args)
            print(
                f"temperature_manifest=PASS examples={result['sample_size']} "
                f"seed={result['selection_seed']}",
                flush=True,
            )
        else:
            result = analyze(args)
            print(
                f"temperature_analysis=PASS examples={result['num_examples']} ",
                f"conditions={','.join(result['condition_metrics'])}",
                flush=True,
            )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
