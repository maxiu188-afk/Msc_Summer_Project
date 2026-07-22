#!/usr/bin/env python3
"""Evaluate frozen BioASQ Probes and single-answer UQ on PubMedQA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.frozen_probe import load_frozen_probe_bundle, score_frozen_probe, sha256_file
from archehr_sebaseline.phase2_probe import binary_metrics, binary_ranking_metrics, continuous_metrics
from archehr_sebaseline.pubmedqa_transfer import PUBMEDQA_LABELS, extract_leading_pubmedqa_label


UQ_FIELDS = {
    "verbalized_confidence_uncertainty": ("self_report", "verbalized_confidence_uncertainty", True),
    "sequence_nll": ("single_answer", "mean_sequence_nll", False),
    "normalized_nll": ("single_answer", "mean_normalized_nll", False),
    "mean_token_entropy": ("single_answer", "mean_token_entropy", False),
    "max_token_entropy": ("single_answer", "max_token_entropy", False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--frozen-probe-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))
    result = {str(row.get("example_id") or ""): row for row in rows}
    if not result or "" in result or len(result) != len(rows):
        raise ValueError(f"{path} contains an empty or duplicate example ID.")
    return result


def finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return result


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_hidden_test(run_dir: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[str], list[str]]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to load PubMedQA hidden-state artifacts.") from exc
    artifact = torch.load(
        run_dir / "hidden_states" / "phase2_hidden_states_test.pt",
        map_location="cpu",
        weights_only=True,
    )
    vectors = artifact.get("vectors")
    valid = artifact.get("position_valid")
    if vectors is None or valid is None or len(vectors.shape) != 4:
        raise ValueError("PubMedQA hidden-state artifact must contain [N,block,position,hidden].")
    index_rows = sorted(
        read_jsonl(run_dir / "hidden_states" / "phase2_hidden_states_test_index.jsonl"),
        key=lambda row: int(row.get("tensor_row", -1)),
    )
    if len(index_rows) != int(vectors.shape[0]):
        raise ValueError("PubMedQA hidden-state tensor and index row counts differ.")
    if [int(row.get("tensor_row", -1)) for row in index_rows] != list(range(len(index_rows))):
        raise ValueError("PubMedQA hidden-state tensor_row values are not ordered and complete.")
    return (
        vectors.float().numpy(),
        valid.numpy().astype(bool),
        [int(value) for value in artifact.get("transformer_blocks") or []],
        [str(value) for value in artifact.get("token_positions") or []],
        [str(row["example_id"]) for row in index_rows],
    )


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}; pass --overwrite.")
    probes = load_frozen_probe_bundle(args.frozen_probe_bundle)
    if set(probes) != {"p_true_probe", "accuracy_probe"}:
        raise ValueError("Transfer evaluation requires exactly the frozen P(True)-Probe and Accuracy-Probe.")
    p_true_probe = probes["p_true_probe"]
    if p_true_probe.target_threshold is None:
        raise ValueError("Frozen P(True)-Probe is missing its BioASQ-train threshold.")
    run_metadata = json.loads(
        (args.run_dir / "pubmedqa_transfer_run_metadata.json").read_text(encoding="utf-8")
    )
    evidence_mode = str(run_metadata.get("generation", {}).get("evidence_mode") or "none")
    context_conditioned = evidence_mode != "none"
    p_true_score_name = (
        "context_conditioned_p_true_uncertainty"
        if context_conditioned
        else "direct_blind_p_true_uncertainty"
    )

    examples = read_jsonl(args.run_dir / "examples.jsonl")
    generations = read_jsonl(args.run_dir / "best_generations.jsonl")
    self_report = read_csv_by_id(args.run_dir / "uq_baselines" / "self_report_examples.csv")
    single_answer = read_csv_by_id(args.run_dir / "uq_baselines" / "single_answer_example_uq.csv")
    examples_by_id = {str(row.get("id") or ""): row for row in examples}
    generations_by_id = {str(row.get("example_id") or ""): row for row in generations}
    if not examples_by_id or len(examples_by_id) != len(examples):
        raise ValueError("PubMedQA examples contain an empty or duplicate ID.")
    if not (set(examples_by_id) == set(generations_by_id) == set(self_report) == set(single_answer)):
        raise ValueError("PubMedQA examples, generations, P(True), and single-answer UQ IDs differ.")

    vectors, position_valid, blocks, positions, hidden_ids = load_hidden_test(args.run_dir)
    if set(hidden_ids) != set(examples_by_id):
        raise ValueError("PubMedQA hidden-state IDs differ from the other artifacts.")
    probe_scores: dict[str, dict[str, float]] = {name: {} for name in probes}
    for name, probe in probes.items():
        if probe.transformer_block not in blocks or probe.token_position not in positions:
            raise ValueError(f"Target artifact lacks frozen feature {probe.transformer_block}/{probe.token_position}.")
        block_index = blocks.index(probe.transformer_block)
        position_index = positions.index(probe.token_position)
        valid = position_valid[:, position_index]
        features = vectors[valid, block_index, position_index, :].astype(np.float64, copy=False)
        scores = score_frozen_probe(probe, features)
        for example_id, score in zip((value for value, keep in zip(hidden_ids, valid) if keep), scores):
            probe_scores[name][example_id] = float(score)

    prediction_rows: list[dict[str, Any]] = []
    for example_id in hidden_ids:
        example = examples_by_id[example_id]
        generation = generations_by_id[example_id]
        answer = str(generation.get("clean_answer") or generation.get("raw_answer") or "").strip()
        predicted_label = extract_leading_pubmedqa_label(answer)
        gold_label = str(example.get("label") or "").lower()
        if gold_label not in PUBMEDQA_LABELS:
            raise ValueError(f"Invalid official PubMedQA label for {example_id}.")
        uncertainty = finite_float(self_report[example_id].get("p_true_blind_uncertainty"), name=f"P(True) {example_id}")
        row: dict[str, Any] = {
            "example_id": example_id,
            "gold_label": gold_label,
            "predicted_label": predicted_label or "unknown",
            "label_parse_valid": str(predicted_label is not None).lower(),
            "incorrect": int(predicted_label != gold_label),
            "p_true_blind_uncertainty": uncertainty,
            p_true_score_name: uncertainty,
            "p_true_high_frozen_threshold": int(uncertainty >= p_true_probe.target_threshold),
            "p_true_probe_score": probe_scores["p_true_probe"].get(example_id, ""),
            "accuracy_probe_score": probe_scores["accuracy_probe"].get(example_id, ""),
        }
        for score_name, (source_name, field, _) in UQ_FIELDS.items():
            source = self_report if source_name == "self_report" else single_answer
            row[score_name] = finite_float(source[example_id].get(field), name=f"{score_name} {example_id}")
        row.update(
            {
                "answer_with_explanation": answer,
                "reference_long_answer": str(example.get("gold_answer") or ""),
                "reference_long_answer_role": "future_Claude_explanation_alignment_only",
            }
        )
        prediction_rows.append(row)

    metric_rows: list[dict[str, Any]] = []
    target_fields = {
        "p_true_high_frozen_BioASQ_threshold": "p_true_high_frozen_threshold",
        "official_label_incorrect": "incorrect",
    }
    score_fields = {
        "p_true_probe": ("p_true_probe_score", True),
        "accuracy_probe": ("accuracy_probe_score", True),
        p_true_score_name: (p_true_score_name, True),
        **{name: (name, native_probability) for name, (_, _, native_probability) in UQ_FIELDS.items()},
    }
    for target_name, target_field in target_fields.items():
        for score_name, (score_field, native_probability) in score_fields.items():
            valid_rows = [row for row in prediction_rows if row.get(score_field) not in (None, "")]
            labels = np.asarray([int(row[target_field]) for row in valid_rows], dtype=np.int64)
            scores = np.asarray([float(row[score_field]) for row in valid_rows], dtype=np.float64)
            if native_probability:
                metrics = binary_metrics(labels, scores)
            else:
                metrics = {**binary_ranking_metrics(labels, scores), "brier": None}
            p_true_relationship = (
                "frozen_threshold_teacher_target_under_context_shift"
                if context_conditioned
                else "original_probe_target"
            )
            relationship = "cross_target_or_comparator"
            if score_name == "p_true_probe" and target_name == "p_true_high_frozen_BioASQ_threshold":
                relationship = p_true_relationship
            elif score_name == "accuracy_probe" and target_name == "official_label_incorrect":
                relationship = "original_probe_target"
            metric_rows.append(
                {
                    "target": target_name,
                    "score": score_name,
                    "relationship": relationship,
                    "num_examples": len(valid_rows),
                    "num_positive": int(np.sum(labels)),
                    "positive_prevalence": float(np.mean(labels)),
                    "auroc": metrics["auroc"],
                    "average_precision": metrics["average_precision"],
                    "brier": metrics["brier"] if native_probability else "",
                    "target_threshold_fitted_on_target": "false",
                    "probe_refit_on_target": "false",
                }
            )

    continuous_rows = []
    for score_name in ("p_true_probe", "accuracy_probe"):
        score_field = score_fields[score_name][0]
        valid_rows = [row for row in prediction_rows if row.get(score_field) not in (None, "")]
        target = np.asarray([float(row["p_true_blind_uncertainty"]) for row in valid_rows], dtype=np.float64)
        scores = np.asarray([float(row[score_field]) for row in valid_rows], dtype=np.float64)
        association = continuous_metrics(target, scores)
        continuous_rows.append(
            {
                "target": f"continuous_{p_true_score_name}",
                "score": score_name,
                "num_examples": len(valid_rows),
                "mae": association["mae"],
                "rmse": association["rmse"],
                "spearman": association["spearman"],
                "r2": association["r2"],
                "note": "diagnostic association; frozen P(True)-Probe was trained on a BioASQ-derived binary target",
            }
        )

    summary = {
        "schema_version": "pubmedqa_frozen_probe_transfer_evaluation_v1",
        "num_examples": len(prediction_rows),
        "official_label_counts": {
            label: sum(row["gold_label"] == label for row in prediction_rows) for label in PUBMEDQA_LABELS
        },
        "unknown_or_contract_violating_answer_labels": sum(row["label_parse_valid"] == "false" for row in prediction_rows),
        "incorrect_count": sum(int(row["incorrect"]) for row in prediction_rows),
        "p_true_high_count_at_frozen_BioASQ_threshold": sum(int(row["p_true_high_frozen_threshold"]) for row in prediction_rows),
        "frozen_p_true_threshold": p_true_probe.target_threshold,
        "p_true_condition": "provided_pubmed_abstract_context" if context_conditioned else "direct_no_context",
        "probe_bundle_sha256": sha256_file(args.frozen_probe_bundle),
        "probe_policy": "both Probes applied without refitting, recalibration, feature selection, or target-dataset threshold tuning",
        "explanation_policy": "answers and reference LONG_ANSWER saved for a later separate Claude alignment judgment",
        "reference_quality_caveat": "PubMedQA LONG_ANSWER is not assumed to be a manually curated high-quality explanation",
        "high_temperature_uq_generated": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "transfer_metrics.csv", metric_rows, overwrite=args.overwrite)
    write_csv(args.output_dir / "continuous_p_true_association.csv", continuous_rows, overwrite=args.overwrite)
    write_csv(args.output_dir / "transfer_predictions.csv", prediction_rows, overwrite=args.overwrite)
    summary_path = args.output_dir / "transfer_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {summary_path}; pass --overwrite.")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PubMedQA transfer evaluation complete: {len(prediction_rows)} examples -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
