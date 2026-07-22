#!/usr/bin/env python3
"""Materialize already-selected Phase-2 Probes from BioASQ train artifacts only.

This is a provenance repair for the first analysis, which saved predictions
but not fitted scaler/linear parameters. It does not inspect validation/test
features, select a candidate, or consume any target-dataset data.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.frozen_probe import sha256_file, write_frozen_probe_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bioasq-run-dir", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path, required=True)
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
        raise ValueError(f"{path} has an empty or duplicate example ID.")
    return result


def fit_logistic(features: np.ndarray, labels: np.ndarray, *, c: float, seed: int) -> Pipeline:
    pipeline = Pipeline(
        (
            ("standardize", StandardScaler()),
            ("model", LogisticRegression(C=c, solver="lbfgs", max_iter=1000, random_state=seed)),
        )
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        pipeline.fit(features.astype(np.float64, copy=False), labels)
    coefficients = np.asarray(pipeline.named_steps["model"].coef_, dtype=np.float64)
    if not np.isfinite(coefficients).all():
        raise RuntimeError("Materialized Probe produced non-finite coefficients.")
    return pipeline


def main() -> int:
    args = parse_args()
    config = json.loads(args.probe_config.read_text(encoding="utf-8"))
    selected = {
        (str(row.get("track")), str(row.get("analysis"))): row
        for row in config.get("selected_models") or []
    }
    p_true_selected = selected.get(("p_true", "hard_threshold_even"))
    accuracy_selected = selected.get(("accuracy", "accuracy"))
    if p_true_selected is None or accuracy_selected is None:
        raise ValueError("Probe config lacks the selected P(True) or Accuracy Probe.")
    for row in (p_true_selected, accuracy_selected):
        if int(row.get("transformer_block", -1)) != 24 or str(row.get("token_position")) != "LT":
            raise ValueError("Frozen transfer contract requires both selected Probes at block 24/LT.")

    run_dir = args.bioasq_run_dir
    examples = read_jsonl(run_dir / "examples.jsonl")
    train_examples = {str(row["id"]): row for row in examples if str(row.get("split")) == "train"}
    self_report = read_csv_by_id(run_dir / "uq_baselines" / "self_report_examples.csv")
    accuracy_labels = read_csv_by_id(
        run_dir / "claude_binary_main_answer_judge" / "claude_generation_labels.csv"
    )
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to read the frozen BioASQ train tensor.") from exc
    hidden_path = run_dir / "hidden_states" / "phase2_hidden_states_train.pt"
    index_path = run_dir / "hidden_states" / "phase2_hidden_states_train_index.jsonl"
    artifact = torch.load(hidden_path, map_location="cpu", weights_only=True)
    blocks = [int(value) for value in artifact.get("transformer_blocks") or []]
    positions = [str(value) for value in artifact.get("token_positions") or []]
    if 24 not in blocks or "LT" not in positions:
        raise ValueError("BioASQ train tensor lacks block 24/LT.")
    index_rows = sorted(read_jsonl(index_path), key=lambda row: int(row.get("tensor_row", -1)))
    vectors = artifact["vectors"]
    valid = artifact["position_valid"]
    if len(index_rows) != int(vectors.shape[0]):
        raise ValueError("BioASQ train hidden tensor and index counts differ.")
    example_ids = [str(row["example_id"]) for row in index_rows]
    if set(example_ids) != set(train_examples):
        raise ValueError("BioASQ train hidden IDs do not match train examples.")
    block_index = blocks.index(24)
    position_index = positions.index("LT")
    position_valid = valid[:, position_index].numpy().astype(bool)
    if not position_valid.all():
        raise ValueError("The selected LT position must be valid for every BioASQ train row.")
    features = vectors[:, block_index, position_index, :].float().numpy().astype(np.float64, copy=False)

    threshold = float(config["thresholds_from_train_only"]["even"]["threshold"])
    p_true_target = np.asarray(
        [float(self_report[example_id]["p_true_blind_uncertainty"]) >= threshold for example_id in example_ids],
        dtype=np.int64,
    )
    accuracy_valid = np.asarray(
        [
            str(accuracy_labels[example_id].get("label_valid")).lower() == "true"
            and str(accuracy_labels[example_id].get("label")).lower() in {"correct", "incorrect"}
            for example_id in example_ids
        ],
        dtype=bool,
    )
    accuracy_target = np.asarray(
        [int(str(accuracy_labels[example_id].get("label")).lower() == "incorrect") for example_id in example_ids],
        dtype=np.int64,
    )
    c = float(config["models"]["logistic_regression"]["C"])
    seed = int(config.get("random_seed", 20260720))
    p_true_pipeline = fit_logistic(features, p_true_target, c=c, seed=seed)
    accuracy_pipeline = fit_logistic(features[accuracy_valid], accuracy_target[accuracy_valid], c=c, seed=seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_frozen_probe_bundle(
        args.output_dir / "frozen_probe_bundle.json",
        args.output_dir / "frozen_probe_parameters.npz",
        [
            {
                "name": "p_true_probe",
                "track": "p_true",
                "analysis": "hard_threshold_even",
                "pipeline": p_true_pipeline,
                "transformer_block": 24,
                "token_position": "LT",
                "target_threshold": threshold,
                "positive_class": "p_true_blind_uncertainty_at_or_above_frozen_BioASQ_train_threshold",
            },
            {
                "name": "accuracy_probe",
                "track": "accuracy",
                "analysis": "accuracy",
                "pipeline": accuracy_pipeline,
                "transformer_block": 24,
                "token_position": "LT",
                "target_threshold": None,
                "positive_class": "answer_incorrect",
            },
        ],
        source={
            "dataset": "BioASQ training13b",
            "fit_split": "train only",
            "selection_replayed": False,
            "validation_or_test_features_read": False,
            "target_dataset_data_read": False,
            "reason": "materialize parameters omitted from the completed first-pass analysis",
            "probe_config_sha256": sha256_file(args.probe_config),
            "train_hidden_state_sha256": sha256_file(hidden_path),
            "train_hidden_index_sha256": sha256_file(index_path),
            "self_report_sha256": sha256_file(run_dir / "uq_baselines" / "self_report_examples.csv"),
            "accuracy_labels_sha256": sha256_file(run_dir / "claude_binary_main_answer_judge" / "claude_generation_labels.csv"),
            "p_true_train_rows": len(example_ids),
            "accuracy_train_rows": int(np.sum(accuracy_valid)),
        },
        overwrite=args.overwrite,
    )
    print(f"Materialized two frozen BioASQ Probes without reading target data: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
