#!/usr/bin/env python3
"""Run the pre-specified first-pass linear Phase-2 BioASQ Probes locally.

The runner consumes saved Phase-2 tensors only.  It never loads a language
model, regenerates an answer, calls Claude, or searches combinations of token
positions/layers.  A candidate is exactly one saved hidden-state vector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_probe import (
    ThresholdSpec,
    binary_metrics,
    continuous_metrics,
    fit_threshold,
)
from archehr_sebaseline.frozen_probe import write_frozen_probe_bundle


SPLITS = ("train", "validation", "test")
THRESHOLD_METHODS = ("even", "minimum_within_variance")
DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "bioasq_phase2_gemma3_12b_single_answer_seed31"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs" / "bioasq_phase2_linear_probes_seed31"


@dataclass
class SplitData:
    example_ids: list[str]
    bioasq_types: list[str]
    vectors: np.ndarray
    position_valid: np.ndarray
    uncertainty: np.ndarray
    incorrect: np.ndarray | None
    accuracy_label_valid: np.ndarray | None


@dataclass
class Candidate:
    analysis: str
    transformer_block: int
    token_position: str
    model_name: str
    model: Pipeline
    validation_ids: list[str]
    validation_types: list[str]
    validation_target: np.ndarray
    validation_scores: np.ndarray
    test_ids: list[str]
    test_types: list[str]
    test_target: np.ndarray
    test_features: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-accuracy-probe", action="store_true")
    parser.add_argument(
        "--allow-incomplete-accuracy-labels",
        action="store_true",
        help="Exclude invalid/blank Claude labels from Accuracy-Probe only; record counts in the run config.",
    )
    parser.add_argument("--logistic-c", type=float, default=1.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--elasticnet-alpha", type=float, default=0.1)
    parser.add_argument("--elasticnet-l1-ratio", type=float, default=0.5)
    parser.add_argument("--random-seed", type=int, default=20260720)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))
    by_id = {str(row.get("example_id") or ""): row for row in rows}
    if not by_id or "" in by_id or len(by_id) != len(rows):
        raise ValueError(f"{path} has an empty or duplicate example ID.")
    return by_id


def finite_float(value: object, *, name: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return result


def load_hidden_artifact(path: Path) -> dict[str, Any]:
    """Load a trusted local Phase-2 tensor artifact on CPU only."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required to read Phase-2 .pt hidden-state artifacts. "
            "Install archehr_sebaseline/requirements.txt in the local virtual environment."
        ) from exc
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_index(path: Path, *, expected_rows: int, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"{path} has {len(rows)} rows but tensor has {expected_rows}.")
    rows = sorted(rows, key=lambda row: int(row.get("tensor_row", -1)))
    if [int(row.get("tensor_row", -1)) for row in rows] != list(range(expected_rows)):
        raise ValueError(f"{path} tensor_row values are not a complete ordered range.")
    if any(str(row.get("split")) != split for row in rows):
        raise ValueError(f"{path} has a split inconsistent with {split}.")
    return rows


def load_phase2_data(
    run_dir: Path,
    *,
    require_accuracy_labels: bool,
    allow_incomplete_accuracy_labels: bool = False,
) -> dict[str, SplitData]:
    """Join saved targets, type metadata, and tensors by immutable example ID."""

    examples = read_jsonl(run_dir / "examples.jsonl")
    examples_by_id = {str(row.get("id") or ""): row for row in examples}
    if not examples_by_id or "" in examples_by_id or len(examples_by_id) != len(examples):
        raise ValueError("examples.jsonl has an empty or duplicate ID.")
    self_report = read_csv_by_id(run_dir / "uq_baselines" / "self_report_examples.csv")
    if set(examples_by_id) != set(self_report):
        raise ValueError("examples.jsonl and self_report_examples.csv IDs do not match.")

    labels: dict[str, dict[str, str]] = {}
    if require_accuracy_labels:
        labels_path = run_dir / "claude_binary_main_answer_judge" / "claude_generation_labels.csv"
        labels = read_csv_by_id(labels_path)
        if set(labels) != set(examples_by_id):
            raise ValueError("Claude label IDs do not match Phase-2 examples.")
        invalid = [
            example_id
            for example_id, row in labels.items()
            if str(row.get("label_valid")).lower() != "true"
            or str(row.get("label")).lower() not in {"correct", "incorrect"}
        ]
        if invalid and not allow_incomplete_accuracy_labels:
            raise ValueError(f"Claude labels are incomplete or invalid ({len(invalid)} rows).")

    result: dict[str, SplitData] = {}
    reference_blocks: list[int] | None = None
    reference_positions: list[str] | None = None
    for split in SPLITS:
        artifact = load_hidden_artifact(run_dir / "hidden_states" / f"phase2_hidden_states_{split}.pt")
        vectors = artifact.get("vectors")
        valid = artifact.get("position_valid")
        if vectors is None or valid is None or len(vectors.shape) != 4:
            raise ValueError(f"Hidden-state artifact for {split} has no valid [N, block, position, hidden] tensor.")
        blocks = [int(value) for value in artifact.get("transformer_blocks") or []]
        positions = [str(value) for value in artifact.get("token_positions") or []]
        if reference_blocks is None:
            reference_blocks, reference_positions = blocks, positions
        if blocks != reference_blocks or positions != reference_positions:
            raise ValueError("Hidden-state block/position metadata differs across splits.")
        index_rows = _load_index(
            run_dir / "hidden_states" / f"phase2_hidden_states_{split}_index.jsonl",
            expected_rows=int(vectors.shape[0]),
            split=split,
        )
        example_ids = [str(row["example_id"]) for row in index_rows]
        if len(set(example_ids)) != len(example_ids):
            raise ValueError(f"Hidden-state index for {split} has duplicate example IDs.")
        if any(str(examples_by_id[example_id].get("split")) != split for example_id in example_ids):
            raise ValueError(f"Hidden-state index for {split} disagrees with examples.jsonl.")
        uncertainties = np.asarray(
            [finite_float(self_report[example_id].get("p_true_blind_uncertainty"), name=f"P(True) target {example_id}") for example_id in example_ids],
            dtype=np.float64,
        )
        incorrect = None
        accuracy_label_valid = None
        if require_accuracy_labels:
            accuracy_label_valid = np.asarray(
                [
                    str(labels[example_id].get("label_valid")).lower() == "true"
                    and str(labels[example_id].get("label")).lower() in {"correct", "incorrect"}
                    for example_id in example_ids
                ],
                dtype=bool,
            )
            incorrect = np.asarray(
                [int(str(labels[example_id].get("label")).lower() == "incorrect") for example_id in example_ids],
                dtype=np.int64,
            )
        result[split] = SplitData(
            example_ids=example_ids,
            bioasq_types=[str(examples_by_id[example_id].get("bioasq_type") or "unknown").lower() for example_id in example_ids],
            vectors=vectors.float().numpy(),
            position_valid=valid.numpy().astype(bool),
            uncertainty=uncertainties,
            incorrect=incorrect,
            accuracy_label_valid=accuracy_label_valid,
        )
    if set().union(*(set(item.example_ids) for item in result.values())) != set(examples_by_id):
        raise ValueError("The union of hidden-state split IDs does not match examples.jsonl.")
    return result


def _make_model(model_name: str, args: argparse.Namespace) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(C=args.logistic_c, solver="lbfgs", max_iter=1000, random_state=args.random_seed)
    elif model_name == "ridge":
        estimator = Ridge(alpha=args.ridge_alpha)
    elif model_name == "elasticnet":
        estimator = ElasticNet(
            alpha=args.elasticnet_alpha,
            l1_ratio=args.elasticnet_l1_ratio,
            max_iter=100000,
            tol=1e-3,
            random_state=args.random_seed,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}.")
    return Pipeline((("standardize", StandardScaler()), ("model", estimator)))


def _feature_rows(
    data: SplitData,
    block_index: int,
    position_index: int,
    *,
    eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    valid = data.position_valid[:, position_index].copy()
    if eligible is not None:
        if eligible.shape != valid.shape:
            raise ValueError("Eligibility mask shape does not match a hidden-state split.")
        valid &= eligible
    # Gemma states are stored compactly as bf16 and loaded as float32.  Keep the
    # full artifact compact, but promote one candidate at a time: LBFGS/linear
    # algebra can otherwise overflow in float32 while fitting 3,840 features.
    features = data.vectors[valid, block_index, position_index, :].astype(np.float64, copy=False)
    if not np.isfinite(features).all():
        raise ValueError("Hidden-state feature vector contains a non-finite value.")
    return valid, features


def _score(model: Pipeline, model_name: str, features: np.ndarray) -> np.ndarray:
    # Apple Accelerate as used by this local NumPy build emits spurious FPE
    # RuntimeWarnings for finite float64 dense matmul (including Z @ 0).  The
    # narrow filter is paired with the explicit finite check below; it does not
    # hide convergence warnings or accept a non-finite prediction.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        scores = model.predict_proba(features)[:, 1] if model_name == "logistic_regression" else model.predict(features)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise RuntimeError(f"{model_name} produced a non-finite score.")
    return scores


def _fit_model(model: Pipeline, features: np.ndarray, target: np.ndarray, *, model_name: str) -> Pipeline:
    """Fit a linear model and reject a failed/non-finite estimator."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        model.fit(features, target)
    coefficients = np.asarray(model.named_steps["model"].coef_, dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise RuntimeError(f"{model_name} produced non-finite coefficients.")
    return model


def _metric_rows(
    *,
    track: str,
    analysis: str,
    split: str,
    block: int,
    position: str,
    model_name: str,
    target: np.ndarray,
    scores: np.ndarray,
    selected_on_validation: bool,
    binary: bool,
    subset: str,
) -> dict[str, object]:
    metrics = binary_metrics(target, scores) if binary else continuous_metrics(target, scores)
    return {
        "track": track,
        "analysis": analysis,
        "split": split,
        "subset": subset,
        "transformer_block": block,
        "token_position": position,
        "model": model_name,
        "num_examples": int(len(target)),
        "num_positive": int(np.sum(target)) if binary else "",
        "selected_on_validation": str(selected_on_validation).lower(),
        "auroc": metrics.get("auroc", ""),
        "average_precision": metrics.get("average_precision", ""),
        "brier": metrics.get("brier", ""),
        "mae": metrics.get("mae", ""),
        "rmse": metrics.get("rmse", ""),
        "spearman": metrics.get("spearman", ""),
        "r2": metrics.get("r2", ""),
    }


def _select_candidate(candidates: list[Candidate], *, binary: bool) -> Candidate:
    if not candidates:
        raise ValueError("No valid feature candidates were available.")
    if binary:
        def key(candidate: Candidate) -> tuple[float, float, int, str]:
            metrics = binary_metrics(candidate.validation_target, candidate.validation_scores)
            return (
                float(metrics["auroc"]) if metrics["auroc"] is not None else -float("inf"),
                -float(metrics["brier"]),
                -candidate.transformer_block,
                candidate.token_position,
            )
    else:
        def key(candidate: Candidate) -> tuple[float, float, int, str]:
            metrics = continuous_metrics(candidate.validation_target, candidate.validation_scores)
            return (
                -float(metrics["mae"]),
                float(metrics["spearman"]) if metrics["spearman"] is not None else -float("inf"),
                -candidate.transformer_block,
                candidate.token_position,
            )
    return max(candidates, key=key)


def _fit_candidates(
    *,
    data: dict[str, SplitData],
    blocks: list[int],
    positions: list[str],
    analysis: str,
    model_name: str,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    test_target: np.ndarray,
    args: argparse.Namespace,
    eligible_by_split: dict[str, np.ndarray] | None = None,
) -> tuple[list[Candidate], list[dict[str, object]]]:
    candidates: list[Candidate] = []
    metric_rows: list[dict[str, object]] = []
    binary = model_name == "logistic_regression"
    for block_index, block in enumerate(blocks):
        for position_index, position in enumerate(positions):
            train_valid, train_features = _feature_rows(
                data["train"], block_index, position_index,
                eligible=eligible_by_split["train"] if eligible_by_split else None,
            )
            validation_valid, validation_features = _feature_rows(
                data["validation"], block_index, position_index,
                eligible=eligible_by_split["validation"] if eligible_by_split else None,
            )
            test_valid, test_features = _feature_rows(
                data["test"], block_index, position_index,
                eligible=eligible_by_split["test"] if eligible_by_split else None,
            )
            fitted = _make_model(model_name, args)
            _fit_model(fitted, train_features, train_target[train_valid], model_name=model_name)
            train_scores = _score(fitted, model_name, train_features)
            validation_scores = _score(fitted, model_name, validation_features)
            track = "accuracy" if analysis == "accuracy" else "p_true"
            for split, valid, target, scores in (
                ("train", train_valid, train_target[train_valid], train_scores),
                ("validation", validation_valid, validation_target[validation_valid], validation_scores),
            ):
                metric_rows.append(
                    _metric_rows(
                        track=track,
                        analysis=analysis,
                        split=split,
                        block=block,
                        position=position,
                        model_name=model_name,
                        target=target,
                        scores=scores,
                        selected_on_validation=False,
                        binary=binary,
                        subset="overall",
                    )
                )
            candidates.append(
                Candidate(
                    analysis=analysis,
                    transformer_block=block,
                    token_position=position,
                    model_name=model_name,
                    model=fitted,
                    validation_ids=[example_id for example_id, keep in zip(data["validation"].example_ids, validation_valid) if keep],
                    validation_types=[value for value, keep in zip(data["validation"].bioasq_types, validation_valid) if keep],
                    validation_target=validation_target[validation_valid],
                    validation_scores=validation_scores,
                    test_ids=[example_id for example_id, keep in zip(data["test"].example_ids, test_valid) if keep],
                    test_types=[value for value, keep in zip(data["test"].bioasq_types, test_valid) if keep],
                    test_target=test_target[test_valid],
                    test_features=test_features,
                )
            )
    return candidates, metric_rows


def write_csv(path: Path, rows: list[dict[str, object]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    if not rows:
        raise ValueError(f"No rows to write to {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.logistic_c <= 0 or args.ridge_alpha < 0 or args.elasticnet_alpha <= 0:
        raise ValueError("Regularisation strengths must be positive (Ridge may be zero).")
    if not 0.0 < args.elasticnet_l1_ratio <= 1.0:
        raise ValueError("--elasticnet-l1-ratio must be in (0, 1].")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}; pass --overwrite.")

    if args.allow_incomplete_accuracy_labels and not args.include_accuracy_probe:
        raise ValueError("--allow-incomplete-accuracy-labels requires --include-accuracy-probe.")
    data = load_phase2_data(
        args.run_dir,
        require_accuracy_labels=args.include_accuracy_probe,
        allow_incomplete_accuracy_labels=args.allow_incomplete_accuracy_labels,
    )
    sample_artifact = load_hidden_artifact(args.run_dir / "hidden_states" / "phase2_hidden_states_train.pt")
    blocks = [int(value) for value in sample_artifact["transformer_blocks"]]
    positions = [str(value) for value in sample_artifact["token_positions"]]
    train_uncertainty = data["train"].uncertainty
    threshold_specs = {method: fit_threshold(method, train_uncertainty) for method in THRESHOLD_METHODS}

    metric_rows: list[dict[str, object]] = []
    selected: list[tuple[Candidate, bool]] = []
    for method, spec in threshold_specs.items():
        candidates, rows = _fit_candidates(
            data=data,
            blocks=blocks,
            positions=positions,
            analysis=f"hard_threshold_{method}",
            model_name="logistic_regression",
            train_target=spec.labels(train_uncertainty),
            validation_target=spec.labels(data["validation"].uncertainty),
            test_target=spec.labels(data["test"].uncertainty),
            args=args,
        )
        metric_rows.extend(rows)
        selected.append((_select_candidate(candidates, binary=True), True))
    for analysis, model_name in (("ridge", "ridge"), ("elasticnet", "elasticnet")):
        candidates, rows = _fit_candidates(
            data=data,
            blocks=blocks,
            positions=positions,
            analysis=analysis,
            model_name=model_name,
            train_target=train_uncertainty,
            validation_target=data["validation"].uncertainty,
            test_target=data["test"].uncertainty,
            args=args,
        )
        metric_rows.extend(rows)
        selected.append((_select_candidate(candidates, binary=False), False))
    if args.include_accuracy_probe:
        assert data["train"].incorrect is not None and data["validation"].incorrect is not None and data["test"].incorrect is not None
        assert data["train"].accuracy_label_valid is not None and data["validation"].accuracy_label_valid is not None and data["test"].accuracy_label_valid is not None
        candidates, rows = _fit_candidates(
            data=data,
            blocks=blocks,
            positions=positions,
            analysis="accuracy",
            model_name="logistic_regression",
            train_target=data["train"].incorrect,
            validation_target=data["validation"].incorrect,
            test_target=data["test"].incorrect,
            args=args,
            eligible_by_split={split: data[split].accuracy_label_valid for split in SPLITS},
        )
        metric_rows.extend(rows)
        selected.append((_select_candidate(candidates, binary=True), True))

    prediction_rows: list[dict[str, object]] = []
    selected_summary: list[dict[str, object]] = []
    frozen_export_specs: list[dict[str, object]] = []
    for candidate, binary in selected:
        # Test predictions are deliberately made only after validation has fixed
        # this candidate.  Other candidates never receive a test metric.
        test_scores = _score(candidate.model, candidate.model_name, candidate.test_features)
        track = "accuracy" if candidate.analysis == "accuracy" else "p_true"
        metric_rows.append(
            _metric_rows(
                track=track, analysis=candidate.analysis, split="test", block=candidate.transformer_block,
                position=candidate.token_position, model_name=candidate.model_name,
                target=candidate.test_target, scores=test_scores, selected_on_validation=True,
                binary=binary, subset="overall",
            )
        )
        for bioasq_type in sorted(set(candidate.test_types)):
            mask = np.asarray([value == bioasq_type for value in candidate.test_types])
            metric_rows.append(
                _metric_rows(
                    track=track, analysis=candidate.analysis, split="test", block=candidate.transformer_block,
                    position=candidate.token_position, model_name=candidate.model_name,
                    target=candidate.test_target[mask], scores=test_scores[mask], selected_on_validation=True,
                    binary=binary, subset=bioasq_type,
                )
            )
        selected_summary.append({
            "track": track, "analysis": candidate.analysis, "model": candidate.model_name,
            "transformer_block": candidate.transformer_block, "token_position": candidate.token_position,
            "validation_rows": len(candidate.validation_ids), "test_rows": len(candidate.test_ids),
        })
        if candidate.analysis in {"hard_threshold_even", "accuracy"}:
            frozen_export_specs.append(
                {
                    "name": "p_true_probe" if candidate.analysis == "hard_threshold_even" else "accuracy_probe",
                    "track": track,
                    "analysis": candidate.analysis,
                    "pipeline": candidate.model,
                    "transformer_block": candidate.transformer_block,
                    "token_position": candidate.token_position,
                    "target_threshold": (
                        threshold_specs["even"].threshold
                        if candidate.analysis == "hard_threshold_even"
                        else None
                    ),
                    "positive_class": (
                        "p_true_blind_uncertainty_at_or_above_frozen_BioASQ_train_threshold"
                        if candidate.analysis == "hard_threshold_even"
                        else "answer_incorrect"
                    ),
                }
            )
        for example_id, bioasq_type, target, score in zip(candidate.test_ids, candidate.test_types, candidate.test_target, test_scores):
            prediction_rows.append({
                "example_id": example_id, "bioasq_type": bioasq_type, "track": track,
                "analysis": candidate.analysis, "target": float(target), "score": float(score),
            })

    config = {
        "schema_version": "phase2_linear_probe_v1",
        "random_seed": args.random_seed,
        "run_dir": str(args.run_dir.resolve()),
        "input_artifacts": {
            "targets": "uq_baselines/self_report_examples.csv:p_true_blind_uncertainty",
            "hidden_states": "hidden_states/phase2_hidden_states_{train,validation,test}.pt",
            "accuracy_labels": "claude_binary_main_answer_judge/claude_generation_labels.csv" if args.include_accuracy_probe else None,
        },
        "feature_contract": {"transformer_blocks": blocks, "token_positions": positions, "one_vector_per_candidate": True},
        "thresholds_from_train_only": {method: spec.to_dict() for method, spec in threshold_specs.items()},
        "models": {
            "logistic_regression": {"penalty": "l2", "C": args.logistic_c, "solver": "lbfgs"},
            "ridge": {"alpha": args.ridge_alpha},
            "elasticnet": {"alpha": args.elasticnet_alpha, "l1_ratio": args.elasticnet_l1_ratio},
        },
        "selection": {
            "binary": "validation AUROC, then lower Brier; no test candidate metrics are written",
            "continuous": "validation MAE, then higher Spearman; no test candidate metrics are written",
        },
        "accuracy_probe_included": args.include_accuracy_probe,
        "accuracy_label_policy": (
            {
                "allow_incomplete": args.allow_incomplete_accuracy_labels,
                "valid_rows_by_split": {
                    split: int(np.sum(data[split].accuracy_label_valid)) for split in SPLITS
                },
                "excluded_rows_by_split": {
                    split: int(len(data[split].example_ids) - np.sum(data[split].accuracy_label_valid)) for split in SPLITS
                },
            }
            if args.include_accuracy_probe
            else None
        ),
        "selected_models": selected_summary,
        "frozen_transfer_bundle": {
            "metadata": "frozen_probe_bundle.json",
            "parameters": "frozen_probe_parameters.npz",
            "included": ["p_true_probe", "accuracy_probe"] if args.include_accuracy_probe else ["p_true_probe"],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "probe_run_config.json"
    if config_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {config_path}; pass --overwrite.")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_frozen_probe_bundle(
        args.output_dir / "frozen_probe_bundle.json",
        args.output_dir / "frozen_probe_parameters.npz",
        frozen_export_specs,
        source={
            "dataset": "BioASQ training13b",
            "fit_split": "train",
            "selection_split": "validation",
            "test_used_for_fitting_or_selection": False,
            "random_seed": args.random_seed,
            "feature_selection": "validation-selected before the BioASQ test evaluation",
        },
        overwrite=args.overwrite,
    )
    write_csv(args.output_dir / "candidate_metrics.csv", metric_rows, overwrite=args.overwrite)
    write_csv(args.output_dir / "selected_test_predictions.csv", prediction_rows, overwrite=args.overwrite)
    print(f"wrote {len(metric_rows)} metrics and {len(prediction_rows)} selected test predictions to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
