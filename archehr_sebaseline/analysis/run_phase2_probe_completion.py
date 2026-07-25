#!/usr/bin/env python3
"""Complete the frozen Phase-2 Probe statistics using saved artifacts only.

This analysis performs exactly two pre-declared operations:

1. paired test-set AUROC bootstrap for Accuracy-Probe versus blind P(True);
2. one two-score logistic fusion fitted on validation and evaluated once on test.

It never regenerates answers, calls an external judge, changes either Probe,
or searches layers, token positions, thresholds, or feature combinations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.frozen_probe import load_frozen_probe_bundle, score_frozen_probe


DEFAULT_RUN_DIR = PROJECT_ROOT / "outputs" / "bioasq_phase2_gemma3_12b_single_answer_seed31"
DEFAULT_BUNDLE = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "bioasq_phase2_frozen_probe_bundle_seed31"
    / "frozen_probe_bundle.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "analysis_outputs" / "bioasq_phase2_probe_completion_20260725"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--frozen-probe-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260725)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def read_csv_by_id(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))
    by_id = {str(row.get("example_id") or ""): row for row in rows}
    if not by_id or "" in by_id or len(by_id) != len(rows):
        raise ValueError(f"{path} contains an empty or duplicate example ID.")
    return by_id


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split(
    run_dir: Path,
    split: str,
    *,
    labels: dict[str, dict[str, str]],
    self_report: dict[str, dict[str, str]],
    examples: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required to read saved hidden states.") from exc

    hidden_path = run_dir / "hidden_states" / f"phase2_hidden_states_{split}.pt"
    index_path = run_dir / "hidden_states" / f"phase2_hidden_states_{split}_index.jsonl"
    artifact = torch.load(hidden_path, map_location="cpu", weights_only=True)
    vectors = artifact.get("vectors")
    position_valid = artifact.get("position_valid")
    blocks = [int(value) for value in artifact.get("transformer_blocks") or []]
    positions = [str(value) for value in artifact.get("token_positions") or []]
    if vectors is None or position_valid is None or len(vectors.shape) != 4:
        raise ValueError(f"{hidden_path} lacks the required hidden-state tensors.")
    if 24 not in blocks or "LT" not in positions:
        raise ValueError(f"{hidden_path} lacks block 24 / LT.")
    rows = sorted(read_jsonl(index_path), key=lambda row: int(row["tensor_row"]))
    if len(rows) != int(vectors.shape[0]):
        raise ValueError(f"{index_path} row count does not match the tensor.")

    block_index = blocks.index(24)
    position_index = positions.index("LT")
    ids = np.asarray([str(row["example_id"]) for row in rows], dtype=object)
    valid_label = np.asarray(
        [
            str(labels[example_id].get("label_valid")).lower() == "true"
            and str(labels[example_id].get("label")).lower() in {"correct", "incorrect"}
            for example_id in ids
        ],
        dtype=bool,
    )
    valid_position = position_valid.numpy().astype(bool)[:, position_index]
    eligible = valid_label & valid_position
    features = (
        vectors[eligible, block_index, position_index, :]
        .float()
        .numpy()
        .astype(np.float64, copy=False)
    )
    selected_ids = ids[eligible].tolist()
    return {
        "split": split,
        "example_ids": selected_ids,
        "bioasq_types": [
            str(examples[example_id].get("bioasq_type") or "unknown").lower()
            for example_id in selected_ids
        ],
        "features": features,
        "incorrect": np.asarray(
            [
                int(str(labels[example_id]["label"]).lower() == "incorrect")
                for example_id in selected_ids
            ],
            dtype=np.int64,
        ),
        "blind_p_true_uncertainty": np.asarray(
            [
                float(self_report[example_id]["p_true_blind_uncertainty"])
                for example_id in selected_ids
            ],
            dtype=np.float64,
        ),
        "hidden_path": hidden_path,
        "index_path": index_path,
    }


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
    }


def paired_auc_bootstrap(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    samples: int,
    random_seed: int,
) -> dict[str, Any]:
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    observed = float(
        roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)
    )
    rng = np.random.default_rng(random_seed)
    differences = []
    skipped = 0
    for _ in range(samples):
        indexes = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indexes]
        if len(np.unique(sampled_labels)) != 2:
            skipped += 1
            continue
        differences.append(
            float(
                roc_auc_score(sampled_labels, candidate[indexes])
                - roc_auc_score(sampled_labels, baseline[indexes])
            )
        )
    if not differences:
        raise RuntimeError("All paired-bootstrap resamples had one class.")
    values = np.asarray(differences, dtype=np.float64)
    return {
        "comparison": "accuracy_probe_minus_blind_p_true",
        "metric": "auroc",
        "observed_difference": observed,
        "ci_lower_95": float(np.quantile(values, 0.025)),
        "ci_upper_95": float(np.quantile(values, 0.975)),
        "fraction_at_or_below_zero": float(np.mean(values <= 0.0)),
        "requested_resamples": samples,
        "valid_resamples": int(values.size),
        "skipped_single_label_resamples": skipped,
        "random_seed": random_seed,
    }


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    bundle_path = args.frozen_probe_bundle.resolve()
    labels_path = (
        run_dir
        / "claude_binary_main_answer_judge"
        / "claude_generation_labels.csv"
    )
    self_report_path = run_dir / "uq_baselines" / "self_report_examples.csv"
    examples_path = run_dir / "examples.jsonl"

    labels = read_csv_by_id(labels_path)
    self_report = read_csv_by_id(self_report_path)
    example_rows = read_jsonl(examples_path)
    examples = {str(row["id"]): row for row in example_rows}
    if not (set(labels) == set(self_report) == set(examples)):
        raise ValueError("Labels, P(True), and examples do not share the same IDs.")

    probes = load_frozen_probe_bundle(bundle_path)
    if set(probes) != {"p_true_probe", "accuracy_probe"}:
        raise ValueError("Expected exactly the two accepted frozen Probes.")

    split_data = {
        split: load_split(
            run_dir,
            split,
            labels=labels,
            self_report=self_report,
            examples=examples,
        )
        for split in ("validation", "test")
    }
    for data in split_data.values():
        data["p_true_probe"] = score_frozen_probe(
            probes["p_true_probe"], data["features"]
        )
        data["accuracy_probe"] = score_frozen_probe(
            probes["accuracy_probe"], data["features"]
        )

    validation = split_data["validation"]
    test = split_data["test"]
    if len(validation["incorrect"]) != 384 or len(test["incorrect"]) != 384:
        raise ValueError(
            "Expected exactly 384 valid-labelled rows in validation and test."
        )
    fusion = Pipeline(
        (
            ("standardize", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=args.random_seed,
                ),
            ),
        )
    )
    validation_features = np.column_stack(
        [validation["p_true_probe"], validation["accuracy_probe"]]
    )
    test_features = np.column_stack([test["p_true_probe"], test["accuracy_probe"]])
    fusion.fit(validation_features, validation["incorrect"])
    validation["two_probe_fusion"] = fusion.predict_proba(validation_features)[:, 1]
    test["two_probe_fusion"] = fusion.predict_proba(test_features)[:, 1]

    metric_rows = []
    for split, data in split_data.items():
        for method in (
            "p_true_probe",
            "accuracy_probe",
            "blind_p_true_uncertainty",
            "two_probe_fusion",
        ):
            metric_rows.append(
                {
                    "split": split,
                    "method": method,
                    "examples": len(data["incorrect"]),
                    **ranking_metrics(data["incorrect"], data[method]),
                    "fit_or_selection_role": (
                        "fusion_fit"
                        if split == "validation" and method == "two_probe_fusion"
                        else "frozen_evaluation"
                    ),
                }
            )

    bootstrap = paired_auc_bootstrap(
        test["incorrect"],
        test["blind_p_true_uncertainty"],
        test["accuracy_probe"],
        samples=args.bootstrap_samples,
        random_seed=args.random_seed,
    )
    prediction_rows = []
    for split, data in split_data.items():
        for index, example_id in enumerate(data["example_ids"]):
            prediction_rows.append(
                {
                    "split": split,
                    "example_id": example_id,
                    "bioasq_type": data["bioasq_types"][index],
                    "incorrect": int(data["incorrect"][index]),
                    "p_true_probe": float(data["p_true_probe"][index]),
                    "accuracy_probe": float(data["accuracy_probe"][index]),
                    "blind_p_true_uncertainty": float(
                        data["blind_p_true_uncertainty"][index]
                    ),
                    "two_probe_fusion": float(data["two_probe_fusion"][index]),
                }
            )

    model = fusion.named_steps["model"]
    scaler = fusion.named_steps["standardize"]
    summary = {
        "schema_version": "bioasq_phase2_probe_completion_v1",
        "status": "complete",
        "protocol": {
            "bootstrap": (
                "paired test-row resampling for Accuracy-Probe minus blind "
                "P(True) AUROC"
            ),
            "fusion": (
                "standardized two-score logistic regression fitted once on "
                "validation and evaluated once on test"
            ),
            "exploratory_fusion": True,
            "test_used_for_fitting_or_selection": False,
            "new_generation_or_judging": False,
        },
        "rows": {"validation": 384, "test": 384},
        "bootstrap": bootstrap,
        "fusion": {
            "features": ["p_true_probe", "accuracy_probe"],
            "standardizer_mean": scaler.mean_.tolist(),
            "standardizer_scale": scaler.scale_.tolist(),
            "standardized_coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
        },
        "metrics": metric_rows,
        "input_sha256": {
            "frozen_probe_bundle": sha256_file(bundle_path),
            "frozen_probe_parameters": sha256_file(
                bundle_path.parent / "frozen_probe_parameters.npz"
            ),
            "claude_labels": sha256_file(labels_path),
            "self_report": sha256_file(self_report_path),
            "validation_hidden_states": sha256_file(validation["hidden_path"]),
            "validation_hidden_index": sha256_file(validation["index_path"]),
            "test_hidden_states": sha256_file(test["hidden_path"]),
            "test_hidden_index": sha256_file(test["index_path"]),
        },
    }

    write_csv(
        output_dir / "metrics.csv",
        metric_rows,
        overwrite=args.overwrite,
    )
    write_csv(
        output_dir / "paired_bootstrap.csv",
        [bootstrap],
        overwrite=args.overwrite,
    )
    write_csv(
        output_dir / "predictions.csv",
        prediction_rows,
        overwrite=args.overwrite,
    )
    write_json(
        output_dir / "summary.json",
        summary,
        overwrite=args.overwrite,
    )

    print(
        f"phase2_completion=PASS validation={len(validation['incorrect'])} "
        f"test={len(test['incorrect'])} output={output_dir}"
    )
    print(
        "accuracy_minus_blind_auroc="
        f"{bootstrap['observed_difference']:.6f} "
        f"ci95=[{bootstrap['ci_lower_95']:.6f},"
        f"{bootstrap['ci_upper_95']:.6f}]"
    )
    for row in metric_rows:
        if row["split"] == "test":
            print(
                f"{row['method']} auroc={row['auroc']:.6f} "
                f"ap={row['average_precision']:.6f} brier={row['brier']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
