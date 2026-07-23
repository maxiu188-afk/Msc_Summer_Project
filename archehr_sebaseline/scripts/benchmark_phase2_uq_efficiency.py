#!/usr/bin/env python3
"""Benchmark incremental UQ cost on the labelled BioASQ Phase-2 test set.

The main answer is treated as already available.  Timings therefore measure
only the additional work needed by each UQ method.  One hidden-state replay is
shared by the two frozen Probes, while ten high-temperature samples are shared
by normalized NLL, Semantic Entropy, and cluster count.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.cleaning import clean_generation_records
from archehr_sebaseline.data_io import read_jsonl
from archehr_sebaseline.frozen_probe import (
    load_frozen_probe_bundle,
    score_frozen_probe,
    sha256_file,
)
from archehr_sebaseline.generation import (
    GenerationConfig,
    HuggingFaceCausalLMGenerator,
    generate_answer_records,
)
from archehr_sebaseline.nli_clustering import (
    HuggingFaceNLIScorer,
    NLIConfig,
    cluster_by_bidirectional_entailment,
)
from archehr_sebaseline.phase2_probe import binary_ranking_metrics
from archehr_sebaseline.pipeline_level4 import score_level4_clusters
from archehr_sebaseline.self_report_uq import build_p_true_prompt


METHOD_ORDER = (
    "p_true_probe",
    "accuracy_probe",
    "both_probes_joint",
    "blind_p_true",
    "normalized_nll_10_samples",
    "discrete_semantic_entropy",
    "num_clusters",
)
KNOWN_OUTPUTS = (
    "selection.csv",
    "stage_timing_per_example.jsonl",
    "method_efficiency_per_example.jsonl",
    "uq_scores.jsonl",
    "stage_timing_per_example.csv",
    "method_efficiency_per_example.csv",
    "uq_scores.csv",
    "efficiency_summary.csv",
    "uq_ranking_metrics.csv",
    "benchmark_summary.json",
    "benchmark_status.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--frozen-probe-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-3-12b-it")
    parser.add_argument("--nli-model-name", default="pritamdeka/PubMedBERT-MNLI-MedNLI")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--nli-torch-dtype", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--expected-examples", type=int, default=384)
    parser.add_argument("--include-semantic-entropy", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, sort_keys=True, allow_nan=False))
            outfile.write("\n")
        outfile.flush()


def _jsonl_to_csv(jsonl_path: Path, csv_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(jsonl_path)
    _write_csv(csv_path, rows)
    return rows


def _balanced_cap(rows: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row["example_id"]))
    if limit is None or limit >= len(ordered):
        return ordered
    if limit <= 0:
        raise ValueError("--max-examples must be positive.")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        buckets[str(row["bioasq_type"])].append(row)
    selected: list[dict[str, Any]] = []
    type_order = [name for name in ("factoid", "list", "summary") if buckets[name]]
    cursor = 0
    while len(selected) < limit:
        made_progress = False
        for name in type_order:
            if cursor < len(buckets[name]) and len(selected) < limit:
                selected.append(buckets[name][cursor])
                made_progress = True
        if not made_progress:
            break
        cursor += 1
    if len(selected) != limit:
        raise RuntimeError(f"Could select only {len(selected)} of {limit} requested examples.")
    return selected


def load_selection(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = {str(row["id"]): row for row in read_jsonl(args.run_dir / "examples.jsonl")}
    prompts = {
        str(row["example_id"]): row for row in read_jsonl(args.run_dir / "prompts.jsonl")
    }
    generations = {
        str(row["example_id"]): row
        for row in read_jsonl(args.run_dir / "best_generations.jsonl")
        if int(row.get("sample_id", 0)) == 0
    }
    labels = _read_csv(args.labels_path)
    selected = []
    seen: set[str] = set()
    for label_row in labels:
        example_id = str(label_row.get("example_id") or "")
        label = str(label_row.get("label") or "").strip().lower()
        if not _truthy(label_row.get("label_valid")) or label not in {"correct", "incorrect"}:
            continue
        example = examples.get(example_id)
        if example is None or str(example.get("split")) != "test":
            continue
        if example_id in seen:
            raise ValueError(f"Duplicate valid test label for example {example_id}.")
        if example_id not in prompts or example_id not in generations:
            raise ValueError(f"Missing prompt or main answer for test example {example_id}.")
        generation = generations[example_id]
        token_ids = generation.get("generated_token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError(f"Main answer {example_id} lacks generated_token_ids.")
        seen.add(example_id)
        selected.append(
            {
                "example_id": example_id,
                "bioasq_type": str(example.get("bioasq_type") or ""),
                "correctness_label": label,
                "incorrect": int(label == "incorrect"),
                "example": example,
                "prompt_record": prompts[example_id],
                "generation": generation,
            }
        )
    selected = _balanced_cap(selected, args.max_examples)
    if len(selected) != args.expected_examples:
        raise ValueError(
            f"Expected {args.expected_examples} selected test examples, got {len(selected)}."
        )
    if len(selected) >= 3:
        observed_types = {str(row["bioasq_type"]) for row in selected}
        if observed_types != {"factoid", "list", "summary"}:
            raise ValueError(f"Selection does not contain all BioASQ types: {observed_types}.")
    return selected


def _timed_cuda(torch: Any, operation: Callable[[], Any]) -> tuple[Any, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the GPU efficiency benchmark.")
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter()
    start_event.record()
    result = operation()
    end_event.record()
    end_event.synchronize()
    wall_seconds = time.perf_counter() - wall_started
    gpu_seconds = float(start_event.elapsed_time(end_event)) / 1000.0
    return result, wall_seconds, gpu_seconds


def _timed_cpu(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    result = operation()
    return result, time.perf_counter() - started


class CountingNLIScorer:
    def __init__(self, scorer: HuggingFaceNLIScorer):
        self.scorer = scorer
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        *,
        question: str | None = None,
    ) -> str:
        self.calls += 1
        return self.scorer.check_implication(premise, hypothesis, question=question)


def _stage_row(
    *,
    selection: dict[str, Any],
    stage: str,
    wall_seconds: float,
    gpu_seconds: float,
    extra_generated_tokens: int = 0,
    causal_lm_calls: int = 0,
    nli_forward_calls: int = 0,
) -> dict[str, Any]:
    return {
        "example_id": selection["example_id"],
        "bioasq_type": selection["bioasq_type"],
        "stage": stage,
        "wall_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
        "extra_generated_tokens": extra_generated_tokens,
        "causal_lm_calls": causal_lm_calls,
        "nli_forward_calls": nli_forward_calls,
    }


def _method_row(
    *,
    selection: dict[str, Any],
    method: str,
    wall_seconds: float,
    gpu_seconds: float,
    extra_generated_tokens: int,
    causal_lm_calls: int,
    nli_forward_calls: int,
) -> dict[str, Any]:
    return {
        "example_id": selection["example_id"],
        "bioasq_type": selection["bioasq_type"],
        "method": method,
        "wall_seconds": wall_seconds,
        "gpu_seconds": gpu_seconds,
        "extra_generated_tokens": extra_generated_tokens,
        "causal_lm_calls": causal_lm_calls,
        "nli_forward_calls": nli_forward_calls,
    }


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def summarize_efficiency(
    rows: list[dict[str, Any]],
    *,
    nli_load_seconds: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    summaries = []
    for method in METHOD_ORDER:
        method_rows = grouped.get(method)
        if not method_rows:
            continue
        walls = [_finite_float(row["wall_seconds"], name="wall_seconds") for row in method_rows]
        gpus = [_finite_float(row["gpu_seconds"], name="gpu_seconds") for row in method_rows]
        token_count = sum(int(row["extra_generated_tokens"]) for row in method_rows)
        total_wall = sum(walls)
        total_gpu = sum(gpus)
        method_load = (
            nli_load_seconds
            if method in {"discrete_semantic_entropy", "num_clusters"}
            else 0.0
        )
        total_with_load = total_wall + method_load
        summaries.append(
            {
                "method": method,
                "examples": len(method_rows),
                "mean_latency_seconds": statistics.fmean(walls),
                "median_latency_seconds": statistics.median(walls),
                "p95_latency_seconds": _percentile(walls, 95),
                "total_wall_seconds": total_wall,
                "total_gpu_seconds": total_gpu,
                "mean_gpu_seconds": statistics.fmean(gpus),
                "extra_generated_tokens_total": token_count,
                "extra_generated_tokens_per_question": token_count / len(method_rows),
                "causal_lm_calls_total": sum(
                    int(row["causal_lm_calls"]) for row in method_rows
                ),
                "nli_forward_calls_total": sum(
                    int(row["nli_forward_calls"]) for row in method_rows
                ),
                "steady_state_questions_per_second": len(method_rows) / total_wall,
                "steady_state_questions_per_minute": 60.0 * len(method_rows) / total_wall,
                "generated_tokens_per_second": (
                    token_count / total_wall if token_count else 0.0
                ),
                "method_specific_load_seconds": method_load,
                "total_wall_including_method_specific_load_seconds": total_with_load,
                "questions_per_second_including_method_specific_load": (
                    len(method_rows) / total_with_load
                ),
            }
        )
    return summaries


def ranking_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    method_to_field = {
        "p_true_probe": "p_true_probe_uncertainty",
        "accuracy_probe": "accuracy_probe_uncertainty",
        "blind_p_true": "blind_p_true_uncertainty",
        "normalized_nll_10_samples": "normalized_nll_10_samples",
        "discrete_semantic_entropy": "discrete_semantic_entropy",
        "num_clusters": "num_clusters",
    }
    result = []
    labels = np.asarray([int(row["incorrect"]) for row in rows], dtype=np.int64)
    for method, field in method_to_field.items():
        if not rows or any(row.get(field) is None for row in rows):
            continue
        scores = np.asarray(
            [_finite_float(row[field], name=f"{field} score") for row in rows],
            dtype=np.float64,
        )
        metrics = binary_ranking_metrics(labels, scores)
        result.append(
            {
                "method": method,
                "examples": len(rows),
                "positive_class": "incorrect",
                "auroc": metrics["auroc"],
                "average_precision": metrics["average_precision"],
            }
        )
    return result


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _prepare_output(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in KNOWN_OUTPUTS if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Benchmark output already exists in {args.output_dir}; pass --overwrite."
        )
    if args.overwrite:
        for path in existing:
            path.unlink()


def main() -> int:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError("This benchmark requires --device cuda; CPU/MPS timing is not comparable.")
    if args.num_samples != 10:
        raise ValueError("Phase-1 comparison requires exactly ten high-temperature samples.")
    for path in (
        args.run_dir / "examples.jsonl",
        args.run_dir / "prompts.jsonl",
        args.run_dir / "best_generations.jsonl",
        args.labels_path,
        args.frozen_probe_bundle,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    _prepare_output(args)
    selected = load_selection(args)
    selection_rows = [
        {
            "example_id": row["example_id"],
            "bioasq_type": row["bioasq_type"],
            "correctness_label": row["correctness_label"],
            "incorrect": row["incorrect"],
        }
        for row in selected
    ]
    _write_csv(args.output_dir / "selection.csv", selection_rows)

    probes = load_frozen_probe_bundle(args.frozen_probe_bundle)
    if set(probes) != {"p_true_probe", "accuracy_probe"}:
        raise ValueError("Benchmark requires exactly p_true_probe and accuracy_probe.")
    p_true_probe = probes["p_true_probe"]
    accuracy_probe = probes["accuracy_probe"]
    feature_contract = {
        (p_true_probe.transformer_block, p_true_probe.token_position),
        (accuracy_probe.transformer_block, accuracy_probe.token_position),
    }
    if len(feature_contract) != 1:
        raise ValueError(f"The two Probes do not share one feature: {feature_contract}.")
    transformer_block, token_position = next(iter(feature_contract))
    if token_position != "LT":
        raise ValueError(f"Expected the selected Phase-2 Probes at LT, got {token_position}.")

    status_path = args.output_dir / "benchmark_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "loading_models",
                "selected_examples": len(selected),
                "include_semantic_entropy": args.include_semantic_entropy,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    generator_load_started = time.perf_counter()
    generator = HuggingFaceCausalLMGenerator(
        GenerationConfig(
            model_name=args.model_name,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
        )
    )
    torch = generator._torch
    torch.cuda.synchronize()
    generator_load_seconds = time.perf_counter() - generator_load_started

    nli_scorer: CountingNLIScorer | None = None
    nli_load_seconds = 0.0
    if args.include_semantic_entropy:
        nli_load_started = time.perf_counter()
        nli_scorer = CountingNLIScorer(
            HuggingFaceNLIScorer(
                NLIConfig(
                    model_name=args.nli_model_name,
                    device=args.device,
                    max_input_tokens=512,
                    local_files_only=args.local_files_only,
                    torch_dtype=args.nli_torch_dtype,
                    strict_entailment=False,
                    condition_on_question=True,
                )
            )
        )
        torch.cuda.synchronize()
        nli_load_seconds = time.perf_counter() - nli_load_started

    warm = selected[0]
    warm_prompt = str(warm["prompt_record"]["prompt"])
    warm_generation = warm["generation"]
    warm_answer = str(
        warm_generation.get("clean_answer") or warm_generation.get("raw_answer") or ""
    ).strip()
    generator.extract_answer_hidden_states(
        warm_prompt,
        generated_token_ids=[int(value) for value in warm_generation["generated_token_ids"]],
        transformer_blocks=(transformer_block,),
    )
    generator.binary_continuation_probability(
        build_p_true_prompt(warm["example"], warm_answer),
        true_text=" True",
        false_text=" False",
    )
    generator.generate_with_scores(
        warm_prompt,
        sample_index=10_000,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    if nli_scorer is not None:
        nli_scorer.check_implication(
            warm_answer,
            warm_answer,
            question=str(warm["example"].get("question") or ""),
        )
        nli_scorer.reset()
    torch.cuda.synchronize()

    stage_jsonl = args.output_dir / "stage_timing_per_example.jsonl"
    method_jsonl = args.output_dir / "method_efficiency_per_example.jsonl"
    scores_jsonl = args.output_dir / "uq_scores.jsonl"
    benchmark_started = time.monotonic()
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "selected_examples": len(selected),
                "completed_examples": 0,
                "include_semantic_entropy": args.include_semantic_entropy,
                "generator_load_seconds": generator_load_seconds,
                "nli_load_seconds": nli_load_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    for completed, selection in enumerate(selected, start=1):
        example_id = str(selection["example_id"])
        prompt_record = selection["prompt_record"]
        prompt = str(prompt_record["prompt"])
        main_generation = selection["generation"]
        answer = str(
            main_generation.get("clean_answer") or main_generation.get("raw_answer") or ""
        ).strip()

        hidden_result, replay_wall, replay_gpu = _timed_cuda(
            torch,
            lambda: generator.extract_answer_hidden_states(
                prompt,
                generated_token_ids=[
                    int(value) for value in main_generation["generated_token_ids"]
                ],
                transformer_blocks=(transformer_block,),
            ),
        )
        position_index = {"TBG": 0, "SLT": 1, "LT": 2}[token_position]
        feature = (
            hidden_result["vectors"][0, position_index]
            .to(dtype=torch.float32)
            .numpy()
            .reshape(1, -1)
        )
        p_true_score, p_true_head_wall = _timed_cpu(
            lambda: float(score_frozen_probe(p_true_probe, feature)[0])
        )
        accuracy_score, accuracy_head_wall = _timed_cpu(
            lambda: float(score_frozen_probe(accuracy_probe, feature)[0])
        )

        blind_p_true, p_true_wall, p_true_gpu = _timed_cuda(
            torch,
            lambda: generator.binary_continuation_probability(
                build_p_true_prompt(selection["example"], answer),
                true_text=" True",
                false_text=" False",
            ),
        )

        sampled_generations, generation_wall, generation_gpu = _timed_cuda(
            torch,
            lambda: generate_answer_records(
                [prompt_record],
                generator,
                num_samples=args.num_samples,
                model_name=args.model_name,
                generation_level="phase2_efficiency_high_temperature",
                include_token_scores=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            ),
        )
        cleaned_generations = clean_generation_records(sampled_generations)
        extra_tokens = sum(
            int(row.get("num_generated_tokens") or 0) for row in sampled_generations
        )
        normalized_nll_values = [
            _finite_float(row.get("normalized_nll"), name="normalized_nll")
            for row in sampled_generations
        ]
        normalized_nll = statistics.fmean(normalized_nll_values)

        nli_wall = 0.0
        nli_gpu = 0.0
        nli_calls = 0
        discrete_entropy: float | None = None
        num_clusters: int | None = None
        if nli_scorer is not None:
            nli_scorer.reset()
            cluster_records, nli_wall, nli_gpu = _timed_cuda(
                torch,
                lambda: cluster_by_bidirectional_entailment(
                    cleaned_generations,
                    scorer=nli_scorer,
                    examples_by_id={example_id: selection["example"]},
                    strict_entailment=False,
                ),
            )
            nli_calls = nli_scorer.calls
            score_row = score_level4_clusters(cluster_records, cleaned_generations)[0]
            discrete_entropy = _finite_float(
                score_row["discrete_semantic_entropy"],
                name="discrete_semantic_entropy",
            )
            num_clusters = int(score_row["num_clusters"])

        stage_rows = [
            _stage_row(
                selection=selection,
                stage="probe_shared_hidden_replay",
                wall_seconds=replay_wall,
                gpu_seconds=replay_gpu,
                causal_lm_calls=1,
            ),
            _stage_row(
                selection=selection,
                stage="p_true_probe_cpu_head",
                wall_seconds=p_true_head_wall,
                gpu_seconds=0.0,
            ),
            _stage_row(
                selection=selection,
                stage="accuracy_probe_cpu_head",
                wall_seconds=accuracy_head_wall,
                gpu_seconds=0.0,
            ),
            _stage_row(
                selection=selection,
                stage="blind_p_true_two_continuations",
                wall_seconds=p_true_wall,
                gpu_seconds=p_true_gpu,
                causal_lm_calls=2,
            ),
            _stage_row(
                selection=selection,
                stage="high_temperature_generation_10",
                wall_seconds=generation_wall,
                gpu_seconds=generation_gpu,
                extra_generated_tokens=extra_tokens,
                causal_lm_calls=args.num_samples,
            ),
        ]
        if nli_scorer is not None:
            stage_rows.append(
                _stage_row(
                    selection=selection,
                    stage="semantic_nli_clustering",
                    wall_seconds=nli_wall,
                    gpu_seconds=nli_gpu,
                    nli_forward_calls=nli_calls,
                )
            )

        method_rows = [
            _method_row(
                selection=selection,
                method="p_true_probe",
                wall_seconds=replay_wall + p_true_head_wall,
                gpu_seconds=replay_gpu,
                extra_generated_tokens=0,
                causal_lm_calls=1,
                nli_forward_calls=0,
            ),
            _method_row(
                selection=selection,
                method="accuracy_probe",
                wall_seconds=replay_wall + accuracy_head_wall,
                gpu_seconds=replay_gpu,
                extra_generated_tokens=0,
                causal_lm_calls=1,
                nli_forward_calls=0,
            ),
            _method_row(
                selection=selection,
                method="both_probes_joint",
                wall_seconds=replay_wall + p_true_head_wall + accuracy_head_wall,
                gpu_seconds=replay_gpu,
                extra_generated_tokens=0,
                causal_lm_calls=1,
                nli_forward_calls=0,
            ),
            _method_row(
                selection=selection,
                method="blind_p_true",
                wall_seconds=p_true_wall,
                gpu_seconds=p_true_gpu,
                extra_generated_tokens=0,
                causal_lm_calls=2,
                nli_forward_calls=0,
            ),
            _method_row(
                selection=selection,
                method="normalized_nll_10_samples",
                wall_seconds=generation_wall,
                gpu_seconds=generation_gpu,
                extra_generated_tokens=extra_tokens,
                causal_lm_calls=args.num_samples,
                nli_forward_calls=0,
            ),
        ]
        if nli_scorer is not None:
            for method in ("discrete_semantic_entropy", "num_clusters"):
                method_rows.append(
                    _method_row(
                        selection=selection,
                        method=method,
                        wall_seconds=generation_wall + nli_wall,
                        gpu_seconds=generation_gpu + nli_gpu,
                        extra_generated_tokens=extra_tokens,
                        causal_lm_calls=args.num_samples,
                        nli_forward_calls=nli_calls,
                    )
                )

        score_row = {
            "example_id": example_id,
            "bioasq_type": selection["bioasq_type"],
            "correctness_label": selection["correctness_label"],
            "incorrect": selection["incorrect"],
            "p_true_probe_uncertainty": 1.0 - p_true_score,
            "accuracy_probe_uncertainty": 1.0 - accuracy_score,
            "blind_p_true_uncertainty": 1.0 - float(blind_p_true),
            "normalized_nll_10_samples": normalized_nll,
            "discrete_semantic_entropy": discrete_entropy,
            "num_clusters": num_clusters,
        }
        _append_jsonl(stage_jsonl, stage_rows)
        _append_jsonl(method_jsonl, method_rows)
        _append_jsonl(scores_jsonl, [score_row])

        elapsed = time.monotonic() - benchmark_started
        eta = elapsed * (len(selected) - completed) / completed
        print(
            f"[benchmark] {completed}/{len(selected)} example={example_id} "
            f"type={selection['bioasq_type']} extra_tokens={extra_tokens} "
            f"nli_calls={nli_calls} elapsed={_format_duration(elapsed)} "
            f"eta={_format_duration(eta)}",
            flush=True,
        )
        status_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "selected_examples": len(selected),
                    "completed_examples": completed,
                    "include_semantic_entropy": args.include_semantic_entropy,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "generator_load_seconds": generator_load_seconds,
                    "nli_load_seconds": nli_load_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    stage_rows = _jsonl_to_csv(
        stage_jsonl, args.output_dir / "stage_timing_per_example.csv"
    )
    method_rows = _jsonl_to_csv(
        method_jsonl, args.output_dir / "method_efficiency_per_example.csv"
    )
    score_rows = _jsonl_to_csv(scores_jsonl, args.output_dir / "uq_scores.csv")
    efficiency_rows = summarize_efficiency(
        method_rows,
        nli_load_seconds=nli_load_seconds,
    )
    ranking_rows = ranking_metrics(score_rows)
    _write_csv(args.output_dir / "efficiency_summary.csv", efficiency_rows)
    _write_csv(args.output_dir / "uq_ranking_metrics.csv", ranking_rows)

    completed_seconds = time.monotonic() - benchmark_started
    summary = {
        "schema_version": "bioasq_phase2_uq_efficiency_v1",
        "cost_boundary": (
            "incremental UQ cost after the saved T=0.1 main answer exists; "
            "common Gemma model load and main-answer generation excluded"
        ),
        "selection": {
            "split": "test",
            "valid_claude_binary_labels_only": True,
            "examples": len(selected),
            "type_counts": {
                name: sum(row["bioasq_type"] == name for row in selected)
                for name in ("factoid", "list", "summary")
            },
            "incorrect": sum(int(row["incorrect"]) for row in selected),
            "correct": sum(1 - int(row["incorrect"]) for row in selected),
        },
        "probe_contract": {
            "shared_hidden_state_replay": True,
            "transformer_block": transformer_block,
            "token_position": token_position,
            "extra_generated_tokens": 0,
            "bundle_sha256": sha256_file(args.frozen_probe_bundle),
        },
        "sampling_contract": {
            "num_samples": args.num_samples,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
        },
        "semantic_entropy": {
            "included": args.include_semantic_entropy,
            "nli_model_name": args.nli_model_name if args.include_semantic_entropy else None,
            "nli_model_load_seconds": nli_load_seconds,
            "generation_shared_with_normalized_nll": True,
            "generation_and_nli_shared_by_se_and_cluster_count": True,
        },
        "hardware": {
            "device": args.device,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "generator_model_name": args.model_name,
        "generator_load_seconds_excluded_from_incremental_metrics": generator_load_seconds,
        "measured_benchmark_seconds": completed_seconds,
        "stage_rows": len(stage_rows),
        "method_rows": len(method_rows),
        "score_rows": len(score_rows),
        "efficiency_summary": efficiency_rows,
        "ranking_metrics": ranking_rows,
    }
    (args.output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "selected_examples": len(selected),
                "completed_examples": len(selected),
                "include_semantic_entropy": args.include_semantic_entropy,
                "measured_benchmark_seconds": completed_seconds,
                "generator_load_seconds": generator_load_seconds,
                "nli_load_seconds": nli_load_seconds,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Completed UQ efficiency benchmark: {len(selected)} examples in "
        f"{_format_duration(completed_seconds)}; output={args.output_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
