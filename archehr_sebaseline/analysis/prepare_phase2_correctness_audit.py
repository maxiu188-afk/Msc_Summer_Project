#!/usr/bin/env python3
"""Prepare or analyze a blinded human audit of Phase-2 correctness labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


QUESTION_TYPES = ("factoid", "list", "summary")
SENSITIVITY_METHODS = (
    "accuracy_probe",
    "blind_p_true",
    "discrete_semantic_entropy",
    "p_true_probe",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_run = project_root / "outputs" / "bioasq_phase2_gemma3_12b_single_answer_seed31"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase2-predictions",
        type=Path,
        default=project_root
        / "analysis_outputs"
        / "bioasq_phase2_probe_completion_20260725"
        / "predictions.csv",
    )
    parser.add_argument("--run-dir", type=Path, default=default_run)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20260801)
    parser.add_argument(
        "--completed-review",
        type=Path,
        default=None,
        help="Completed blinded CSV to analyze instead of preparing a new audit.",
    )
    parser.add_argument(
        "--calibrated-predictions",
        type=Path,
        default=None,
        help="Optional complete test predictions for label-sensitivity diagnostics.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    # ``utf-8-sig`` also reads ordinary UTF-8 while removing the BOM commonly
    # added when a completed audit is exported from spreadsheet software.
    with path.open("r", encoding="utf-8-sig", newline="") as infile:
        return list(csv.DictReader(infile))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as infile:
        return [json.loads(line) for line in infile if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}.")
    if not rows:
        raise ValueError(f"Refusing to write empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def select_stratified(
    rows: list[dict[str, str]],
    *,
    per_type: int,
    random_seed: int,
) -> list[dict[str, str]]:
    if per_type <= 0:
        raise ValueError("per_type must be positive.")
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split")) == "test":
            buckets[str(row.get("bioasq_type"))].append(row)
    rng = np.random.default_rng(random_seed)
    selected = []
    for question_type in QUESTION_TYPES:
        bucket = sorted(buckets[question_type], key=lambda row: str(row["example_id"]))
        if len(bucket) < per_type:
            raise ValueError(f"Only {len(bucket)} {question_type} rows for {per_type} requested.")
        indexes = rng.choice(len(bucket), size=per_type, replace=False)
        for index in indexes:
            row = dict(bucket[int(index)])
            row["audit_type_population"] = str(len(bucket))
            row["audit_sampling_probability"] = str(per_type / len(bucket))
            row["audit_sampling_weight"] = str(len(bucket) / per_type)
            selected.append(row)
    order = rng.permutation(len(selected))
    return [selected[int(index)] for index in order]


def reference_for(example: dict[str, Any]) -> tuple[str, str]:
    question_type = str(example.get("bioasq_type") or "")
    if question_type in {"factoid", "list"}:
        key = "exact_answers"
    else:
        key = "ideal_answers"
    values = example.get(key)
    if not isinstance(values, list) or not values:
        raise ValueError(f"Example {example.get('id')} lacks {key}.")
    return key, json.dumps(values, ensure_ascii=False)


def prepare(args: argparse.Namespace) -> int:
    prediction_path = args.phase2_predictions.resolve()
    run_dir = args.run_dir.resolve()
    predictions = read_csv(prediction_path)
    selected = select_stratified(
        predictions, per_type=args.per_type, random_seed=args.random_seed
    )
    examples = {str(row["id"]): row for row in read_jsonl(run_dir / "examples.jsonl")}
    generations = {
        str(row["example_id"]): row
        for row in read_jsonl(run_dir / "best_generations.jsonl")
        if int(row.get("sample_id", 0)) == 0
    }
    blinded_rows = []
    key_rows = []
    for index, row in enumerate(selected, start=1):
        review_id = f"QA{index:03d}"
        example_id = str(row["example_id"])
        example = examples[example_id]
        generation = generations[example_id]
        reference_key, reference_answer = reference_for(example)
        blinded_rows.append(
            {
                "review_id": review_id,
                "bioasq_type": row["bioasq_type"],
                "question": str(example.get("question") or ""),
                "reference_key": reference_key,
                "reference_answer": reference_answer,
                "model_answer": str(
                    generation.get("clean_answer") or generation.get("raw_answer") or ""
                ),
                "human_label": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "example_id": example_id,
                "bioasq_type": row["bioasq_type"],
                "claude_label": "incorrect" if int(row["incorrect"]) else "correct",
                "audit_type_population": row["audit_type_population"],
                "audit_sampling_probability": row["audit_sampling_probability"],
                "audit_sampling_weight": row["audit_sampling_weight"],
                "p_true_probe": row["p_true_probe"],
                "accuracy_probe": row["accuracy_probe"],
                "blind_p_true_uncertainty": row["blind_p_true_uncertainty"],
            }
        )
    output_dir = args.output_dir.resolve()
    blinded_path = output_dir / "correctness_audit_blinded.csv"
    key_path = output_dir / "correctness_audit_key.csv"
    write_csv(blinded_path, blinded_rows, overwrite=args.overwrite)
    write_csv(key_path, key_rows, overwrite=args.overwrite)
    write_json(
        output_dir / "correctness_audit_manifest.json",
        {
            "schema_version": "phase2_correctness_audit_v1",
            "status": "awaiting_human_review",
            "sampling": "simple random sample without replacement within each question type",
            "per_type": args.per_type,
            "total": len(blinded_rows),
            "random_seed": args.random_seed,
            "allowed_human_labels": ["correct", "incorrect", "unsure"],
            "blinding": "Claude label and all UQ scores appear only in the separate key",
            "input_sha256": {
                "phase2_predictions": sha256_file(prediction_path),
                "examples": sha256_file(run_dir / "examples.jsonl"),
                "best_generations": sha256_file(run_dir / "best_generations.jsonl"),
            },
        },
        overwrite=args.overwrite,
    )
    print(f"audit=PREPARED rows={len(blinded_rows)} blinded={blinded_path} key={key_path}")
    return 0


def cohens_kappa(first: list[str], second: list[str]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("Kappa inputs must be non-empty and aligned.")
    observed = sum(a == b for a, b in zip(first, second)) / len(first)
    labels = ("correct", "incorrect")
    expected = sum(
        (sum(value == label for value in first) / len(first))
        * (sum(value == label for value in second) / len(second))
        for label in labels
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


def uq_sensitivity_metrics(
    merged: list[dict[str, str]],
    calibrated_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    predictions: dict[tuple[str, str], dict[str, str]] = {}
    for row in calibrated_rows:
        key = (str(row.get("example_id") or ""), str(row.get("method") or ""))
        if not all(key) or key in predictions:
            raise ValueError("Calibrated predictions contain an empty or duplicate key.")
        predictions[key] = row

    results = []
    for question_type in ("overall",) + QUESTION_TYPES:
        cohort = [
            row
            for row in merged
            if row["human_label"] != "unsure"
            and (question_type == "overall" or row["bioasq_type"] == question_type)
        ]
        if not cohort:
            continue
        weights = np.asarray(
            [float(row["audit_sampling_weight"]) for row in cohort], dtype=np.float64
        )
        for label_source in ("claude", "human"):
            labels = np.asarray(
                [
                    int(row[f"{label_source}_label"] == "incorrect")
                    for row in cohort
                ],
                dtype=np.int64,
            )
            if len(np.unique(labels)) != 2:
                continue
            for method in SENSITIVITY_METHODS:
                selected = []
                for row in cohort:
                    prediction = predictions.get((row["example_id"], method))
                    if prediction is None:
                        raise ValueError(
                            f"Missing calibrated prediction for {row['example_id']} / {method}."
                        )
                    selected.append(prediction)
                raw_scores = np.asarray(
                    [float(row["raw_score"]) for row in selected], dtype=np.float64
                )
                probabilities = np.asarray(
                    [float(row["calibrated_error_probability"]) for row in selected],
                    dtype=np.float64,
                )
                if not np.all(np.isfinite(raw_scores)) or not np.all(
                    np.isfinite(probabilities)
                ):
                    raise ValueError("Sensitivity inputs must be finite.")
                results.append(
                    {
                        "bioasq_type": question_type,
                        "label_source": label_source,
                        "method": method,
                        "role": (
                            "auxiliary_cross_target"
                            if method == "p_true_probe"
                            else "primary"
                        ),
                        "examples": len(cohort),
                        "positive_class": "incorrect",
                        "weighting": "inverse_sampling_probability_by_type",
                        "estimated_error_prevalence": float(
                            np.average(labels, weights=weights)
                        ),
                        "auroc": float(
                            roc_auc_score(labels, raw_scores, sample_weight=weights)
                        ),
                        "average_precision": float(
                            average_precision_score(
                                labels, raw_scores, sample_weight=weights
                            )
                        ),
                        "brier": float(
                            brier_score_loss(
                                labels, probabilities, sample_weight=weights
                            )
                        ),
                        "log_loss": float(
                            log_loss(
                                labels,
                                probabilities,
                                sample_weight=weights,
                                labels=[0, 1],
                            )
                        ),
                    }
                )
    return results


def analyze(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    completed_path = args.completed_review.resolve()
    key_path = output_dir / "correctness_audit_key.csv"
    review_rows = read_csv(completed_path)
    key_rows = read_csv(key_path)
    review = {row["review_id"]: row for row in review_rows}
    key = {row["review_id"]: row for row in key_rows}
    if len(review) != len(review_rows) or len(key) != len(key_rows):
        raise ValueError("Completed review or key contains duplicate review IDs.")
    if set(review) != set(key):
        raise ValueError("Completed review and key review IDs differ.")
    merged = []
    for review_id in sorted(review):
        human = str(review[review_id].get("human_label") or "").strip().lower()
        if human not in {"correct", "incorrect", "unsure"}:
            raise ValueError(f"Invalid human label for {review_id}: {human!r}.")
        if review[review_id].get("bioasq_type") != key[review_id].get("bioasq_type"):
            raise ValueError(f"Question type differs for {review_id}.")
        merged.append(
            {
                **key[review_id],
                "human_label": human,
                "human_notes": str(review[review_id].get("notes") or ""),
            }
        )

    metric_rows = []
    for question_type in ("overall",) + QUESTION_TYPES:
        rows = [
            row
            for row in merged
            if question_type == "overall" or row["bioasq_type"] == question_type
        ]
        decided = [row for row in rows if row["human_label"] != "unsure"]
        if not decided:
            continue
        claude = [row["claude_label"] for row in decided]
        human = [row["human_label"] for row in decided]
        weights = np.asarray(
            [float(row["audit_sampling_weight"]) for row in decided], dtype=np.float64
        )
        agreement = np.asarray([a == b for a, b in zip(claude, human)], dtype=np.float64)
        metric_rows.append(
            {
                "bioasq_type": question_type,
                "sampled": len(rows),
                "decided": len(decided),
                "unsure": len(rows) - len(decided),
                "agreement": float(np.mean(agreement)),
                "design_weighted_agreement": float(np.average(agreement, weights=weights)),
                "cohens_kappa": cohens_kappa(claude, human),
                "claude_incorrect_human_correct": sum(
                    a == "incorrect" and b == "correct" for a, b in zip(claude, human)
                ),
                "claude_correct_human_incorrect": sum(
                    a == "correct" and b == "incorrect" for a, b in zip(claude, human)
                ),
            }
        )
    write_csv(
        output_dir / "correctness_audit_agreement.csv",
        metric_rows,
        overwrite=args.overwrite,
    )
    write_csv(
        output_dir / "correctness_audit_merged.csv",
        merged,
        overwrite=args.overwrite,
    )
    output_files = ["correctness_audit_agreement.csv", "correctness_audit_merged.csv"]
    input_sha256 = {
        "completed_review": sha256_file(completed_path),
        "audit_key": sha256_file(key_path),
    }
    if args.calibrated_predictions is not None:
        calibrated_path = args.calibrated_predictions.resolve()
        sensitivity_rows = uq_sensitivity_metrics(merged, read_csv(calibrated_path))
        write_csv(
            output_dir / "correctness_audit_uq_sensitivity.csv",
            sensitivity_rows,
            overwrite=args.overwrite,
        )
        output_files.append("correctness_audit_uq_sensitivity.csv")
        input_sha256["calibrated_predictions"] = sha256_file(calibrated_path)
    overall = next(row for row in metric_rows if row["bioasq_type"] == "overall")
    write_json(
        output_dir / "correctness_audit_analysis_summary.json",
        {
            "schema_version": "phase2_correctness_audit_analysis_v1",
            "status": "complete",
            "rows": len(merged),
            "human_label_counts": {
                label: sum(row["human_label"] == label for row in merged)
                for label in ("correct", "incorrect", "unsure")
            },
            "overall_agreement": overall,
            "uq_sensitivity_scope": (
                "design-weighted diagnostic on the fixed audit sample; no fitting or selection"
                if args.calibrated_predictions is not None
                else "not_requested"
            ),
            "input_sha256": input_sha256,
            "outputs": output_files,
        },
        overwrite=args.overwrite,
    )
    print(f"audit=ANALYZED rows={len(merged)} output={output_dir}")
    return 0


def main() -> int:
    args = parse_args()
    return analyze(args) if args.completed_review is not None else prepare(args)


if __name__ == "__main__":
    raise SystemExit(main())
