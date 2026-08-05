#!/usr/bin/env python3
"""Collect and evaluate Semantic Entropy for the accepted PubMedQA context-v2 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.baseline_uq import (
    EXAMPLE_UQ_FIELDS,
    GENERATION_UQ_FIELDS,
    example_uq_rows,
    generation_uq_rows,
)
from archehr_sebaseline.cleaning import clean_generation_records
from archehr_sebaseline.data_io import read_jsonl, write_csv, write_jsonl
from archehr_sebaseline.generation import (
    GenerationConfig,
    HuggingFaceCausalLMGenerator,
    MissingGenerationDependency,
    generate_answer_records,
)
from archehr_sebaseline.nli_clustering import (
    PUBMEDBERT_MNLI_MODEL,
    EntailmentScorer,
    HuggingFaceNLIScorer,
    NLIConfig,
    cluster_by_bidirectional_entailment,
)
from archehr_sebaseline.phase2_artifacts import sha256_file
from archehr_sebaseline.phase2_probe import binary_ranking_metrics
from archehr_sebaseline.pipeline_level4 import LEVEL4_SCORE_FIELDS, score_level4_clusters
from archehr_sebaseline.pubmedqa_transfer import PUBMEDQA_CONTEXT_PROMPT_VERSION


NUM_SAMPLES = 10
TEMPERATURE = 1.0
TOP_P = 0.9
TOP_K = 50
SEED = 31


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-dir", type=Path, required=True)
    parser.add_argument("--source-analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, default=500)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--model-name", default="google/gemma-3-12b-it")
    parser.add_argument("--nli-model-name", default=PUBMEDBERT_MNLI_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--nli-max-input-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--nli-torch-dtype", default="")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as infile:
        return list(csv.DictReader(infile))


def _indexed(rows: list[dict[str, Any]], *, id_field: str, name: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row.get(id_field) or ""): row for row in rows}
    if not indexed or "" in indexed or len(indexed) != len(rows):
        raise ValueError(f"{name} contains an empty or duplicate ID.")
    return indexed


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return result


def load_frozen_v2_source(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, str]], dict[str, Any]]:
    """Load the accepted v2 source and reject any prompt or label drift."""

    metadata_path = args.source_run_dir / "pubmedqa_transfer_run_metadata.json"
    examples_path = args.source_run_dir / "examples.jsonl"
    prompts_path = args.source_run_dir / "prompts.jsonl"
    predictions_path = args.source_analysis_dir / "transfer_predictions.csv"
    for path in (metadata_path, examples_path, prompts_path, predictions_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required accepted v2 artifact is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_summary = metadata.get("dataset_summary") or {}
    generation = metadata.get("generation") or {}
    if dataset_summary.get("prompt_version") != PUBMEDQA_CONTEXT_PROMPT_VERSION:
        raise ValueError("Source run is not the accepted PubMedQA context-v2 prompt.")
    if generation.get("evidence_mode") != "provided_pubmed_abstract_context":
        raise ValueError("Source run did not use the official PubMed abstract context.")
    if int(generation.get("num_samples", -1)) != 1:
        raise ValueError("Source v2 run must retain its original single low-temperature answer.")
    if str(metadata.get("model_name") or "") != args.model_name:
        raise ValueError("Requested model differs from the accepted v2 source model.")

    examples = read_jsonl(examples_path)
    prompts = read_jsonl(prompts_path)
    predictions = _read_csv(predictions_path)
    examples_by_id = _indexed(examples, id_field="id", name="examples")
    prompts_by_id = _indexed(prompts, id_field="example_id", name="prompts")
    predictions_by_id = _indexed(predictions, id_field="example_id", name="predictions")
    if not (set(examples_by_id) == set(prompts_by_id) == set(predictions_by_id)):
        raise ValueError("Accepted v2 examples, prompts, and correctness predictions have different IDs.")
    if len(examples) != args.expected_examples:
        raise ValueError(
            f"Expected {args.expected_examples} accepted v2 examples, found {len(examples)}."
        )

    for example_id, prompt in prompts_by_id.items():
        if prompt.get("prompt_version") != PUBMEDQA_CONTEXT_PROMPT_VERSION:
            raise ValueError(f"Prompt {example_id} is not context-v2.")
        if prompt.get("evidence_mode") != "provided":
            raise ValueError(f"Prompt {example_id} does not retain provided abstract context.")
        gold_answer = str(examples_by_id[example_id].get("gold_answer") or "").strip()
        if gold_answer and gold_answer in str(prompt.get("prompt") or ""):
            raise ValueError(f"Prompt {example_id} leaks the reference LONG_ANSWER.")
        incorrect = str(predictions_by_id[example_id].get("incorrect") or "")
        if incorrect not in {"0", "1"}:
            raise ValueError(f"Prediction {example_id} has invalid incorrect label {incorrect!r}.")

    ordered_ids = [str(row["id"]) for row in examples]
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("max_examples must be positive.")
        ordered_ids = ordered_ids[: args.max_examples]
    selected_examples = [examples_by_id[example_id] for example_id in ordered_ids]
    selected_prompts = [prompts_by_id[example_id] for example_id in ordered_ids]
    selected_predictions = {example_id: predictions_by_id[example_id] for example_id in ordered_ids}
    source_hashes = {
        "pubmedqa_transfer_run_metadata": sha256_file(metadata_path),
        "examples": sha256_file(examples_path),
        "prompts": sha256_file(prompts_path),
        "transfer_predictions": sha256_file(predictions_path),
    }
    return selected_examples, selected_prompts, selected_predictions, source_hashes


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(
    args: argparse.Namespace,
    *,
    generator: Any | None = None,
    nli_scorer: EntailmentScorer | None = None,
) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}")
    if args.analysis_dir.exists() and any(args.analysis_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Analysis directory is non-empty: {args.analysis_dir}")

    examples, prompts, source_predictions, source_hashes = load_frozen_v2_source(args)
    if generator is None:
        generator = HuggingFaceCausalLMGenerator(
            GenerationConfig(
                model_name=args.model_name,
                num_samples=NUM_SAMPLES,
                max_new_tokens=args.max_new_tokens,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                top_k=TOP_K,
                seed=SEED,
                device=args.device,
                max_input_tokens=args.max_input_tokens,
                local_files_only=args.local_files_only,
                torch_dtype=args.torch_dtype,
            )
        )

    started_at = time.monotonic()

    def generation_progress(completed: int, total: int, example_id: str, sample_id: int) -> None:
        elapsed = time.monotonic() - started_at
        remaining = elapsed * (total - completed) / completed if completed else 0.0
        print(
            f"[generation] {completed}/{total} example={example_id} "
            f"sample={sample_id + 1}/{NUM_SAMPLES} elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(remaining)}",
            flush=True,
        )

    generations = generate_answer_records(
        prompts,
        generator,
        num_samples=NUM_SAMPLES,
        model_name=args.model_name,
        generation_level="pubmedqa_context_v2_semantic_entropy_high_temperature",
        include_token_scores=True,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        progress_callback=generation_progress,
    )
    cleaned_generations = clean_generation_records(generations)

    if nli_scorer is None:
        nli_scorer = HuggingFaceNLIScorer(
            NLIConfig(
                model_name=args.nli_model_name,
                device=args.device,
                max_input_tokens=args.nli_max_input_tokens,
                local_files_only=args.local_files_only,
                torch_dtype=args.nli_torch_dtype or None,
                strict_entailment=False,
                condition_on_question=True,
            )
        )
    nli_started_at = time.monotonic()

    def nli_progress(completed: int, total: int, example_id: str) -> None:
        elapsed = time.monotonic() - nli_started_at
        remaining = elapsed * (total - completed) / completed if completed else 0.0
        print(
            f"[nli] {completed}/{total} example={example_id} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
            flush=True,
        )

    clusters = cluster_by_bidirectional_entailment(
        cleaned_generations,
        scorer=nli_scorer,
        examples_by_id={str(row["id"]): row for row in examples},
        strict_entailment=False,
        progress_callback=nli_progress,
    )
    score_rows = score_level4_clusters(clusters, cleaned_generations)
    expected_generations = len(examples) * NUM_SAMPLES
    if len(generations) != expected_generations or len(cleaned_generations) != expected_generations:
        raise RuntimeError("High-temperature generation count is incomplete.")
    if len(clusters) != len(examples) or len(score_rows) != len(examples):
        raise RuntimeError("PubMedQA SE cluster or score row count is incomplete.")

    write_jsonl(examples, args.output_dir / "examples.jsonl", overwrite=args.overwrite)
    write_jsonl(prompts, args.output_dir / "prompts.jsonl", overwrite=args.overwrite)
    write_jsonl(generations, args.output_dir / "generations.jsonl", overwrite=args.overwrite)
    write_jsonl(
        cleaned_generations,
        args.output_dir / "cleaned_generations.jsonl",
        overwrite=args.overwrite,
    )
    write_jsonl(clusters, args.output_dir / "clusters.jsonl", overwrite=args.overwrite)
    write_csv(
        score_rows,
        args.output_dir / "se_scores.csv",
        LEVEL4_SCORE_FIELDS,
        overwrite=args.overwrite,
    )
    write_csv(
        generation_uq_rows(cleaned_generations),
        args.output_dir / "generation_uq.csv",
        GENERATION_UQ_FIELDS,
        overwrite=args.overwrite,
    )
    write_csv(
        example_uq_rows(cleaned_generations),
        args.output_dir / "example_uq.csv",
        EXAMPLE_UQ_FIELDS,
        overwrite=args.overwrite,
    )

    analysis_predictions = []
    labels = []
    scores = []
    for score_row in score_rows:
        example_id = str(score_row["example_id"])
        label = int(source_predictions[example_id]["incorrect"])
        score = _finite_float(
            score_row["discrete_semantic_entropy"],
            name=f"discrete_semantic_entropy {example_id}",
        )
        labels.append(label)
        scores.append(score)
        analysis_predictions.append(
            {
                "example_id": example_id,
                "incorrect": label,
                "discrete_semantic_entropy": score,
                "num_clusters": int(score_row["num_clusters"]),
                "cluster_sizes": score_row["cluster_sizes"],
                "source_v2_predicted_label": source_predictions[example_id].get("predicted_label", ""),
                "source_v2_gold_label": source_predictions[example_id].get("gold_label", ""),
            }
        )
    metrics = binary_ranking_metrics(labels, scores)
    metric_rows = [
        {
            "target": "official_label_incorrect",
            "method": "discrete_semantic_entropy",
            "num_examples": len(labels),
            "num_positive": sum(labels),
            "positive_prevalence": sum(labels) / len(labels),
            "auroc": metrics["auroc"],
            "average_precision": metrics["average_precision"],
            "higher_score_means": "higher_error_risk",
            "new_correctness_judging": "false",
        }
    ]
    write_csv(
        analysis_predictions,
        args.analysis_dir / "predictions.csv",
        list(analysis_predictions[0]),
        overwrite=args.overwrite,
    )
    write_csv(
        metric_rows,
        args.analysis_dir / "metrics.csv",
        list(metric_rows[0]),
        overwrite=args.overwrite,
    )

    metadata = {
        "schema_version": "pubmedqa_context_v2_semantic_entropy_v1",
        "status": "complete",
        "scope": "Semantic Entropy only; reuse accepted context-v2 prompts and correctness labels",
        "source_run_dir": str(args.source_run_dir),
        "source_analysis_dir": str(args.source_analysis_dir),
        "source_sha256": source_hashes,
        "prompt_version": PUBMEDQA_CONTEXT_PROMPT_VERSION,
        "model_name": args.model_name,
        "generation": {
            "num_samples": NUM_SAMPLES,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
        },
        "clustering": {
            "model_name": args.nli_model_name,
            "method": "nli_bidirectional_entailment",
            "condition_on_question": True,
            "strict_entailment": False,
        },
        "rows": {
            "examples": len(examples),
            "generations": len(generations),
            "clusters": len(clusters),
            "analysis_predictions": len(analysis_predictions),
        },
        "result": metric_rows[0],
        "excluded_work": [
            "low_temperature_answer_regeneration",
            "Probe_rescoring_or_refitting",
            "P_True_rescoring",
            "calibration",
            "new_correctness_judging",
            "prompt_tuning",
        ],
    }
    _write_json(args.output_dir / "run_metadata.json", metadata, overwrite=args.overwrite)
    _write_json(args.analysis_dir / "summary.json", metadata, overwrite=args.overwrite)
    return metadata


def main() -> int:
    args = parse_args()
    try:
        result = run_experiment(args)
    except (
        FileNotFoundError,
        MissingGenerationDependency,
        RuntimeError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    auroc = result["result"]["auroc"]
    average_precision = result["result"]["average_precision"]
    auroc_text = "undefined" if auroc is None else f"{auroc:.6f}"
    ap_text = "undefined" if average_precision is None else f"{average_precision:.6f}"
    print(
        "PubMedQA context-v2 Semantic Entropy complete: "
        f"examples={result['rows']['examples']} auroc={auroc_text} ap={ap_text}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
