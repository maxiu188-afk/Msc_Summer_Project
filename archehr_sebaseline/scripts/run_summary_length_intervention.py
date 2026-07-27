#!/usr/bin/env python3
"""Collect the paired BioASQ summary-length intervention artifacts.

The accepted current-condition main answer, Claude label, and blind P(True)
score are reused.  The current high-temperature samples are regenerated
because the completed efficiency benchmark retained aggregate scores but not
the sampled answer texts or clusters required by this experiment.  The short
condition generates one new T=0.1 main answer plus ten T=1.0 samples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.cleaning import clean_generation_records
from archehr_sebaseline.data_io import read_jsonl
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
from archehr_sebaseline.pipeline_level4 import score_level4_clusters
from archehr_sebaseline.self_report_uq import build_p_true_prompt
from archehr_sebaseline.summary_length import (
    largest_cluster_fraction,
    prompt_record_for_condition,
    text_length_stats,
)


CONDITIONS = ("current", "short")
KNOWN_ROOT_OUTPUTS = ("cohort_manifest.jsonl", "collection_summary.json", "status.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-run-dir", type=Path, required=True)
    parser.add_argument("--labels-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="google/gemma-3-12b-it")
    parser.add_argument("--nli-model-name", default="pritamdeka/PubMedBERT-MNLI-MedNLI")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--main-temperature", type=float, default=0.1)
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    parser.add_argument("--expected-examples", type=int, default=123)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--nli-torch-dtype", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        outfile.flush()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not numeric: {value!r}.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} is not finite: {value!r}.")
    return number


def _answer(record: dict[str, Any]) -> str:
    answer = str(record.get("clean_answer") or record.get("raw_answer") or "").strip()
    if not answer:
        raise ValueError(f"Empty answer for {record.get('example_id')}.")
    return answer


def load_cohort(args: argparse.Namespace) -> list[dict[str, Any]]:
    examples = {
        str(row["id"]): row for row in read_jsonl(args.phase2_run_dir / "examples.jsonl")
    }
    prompts = {
        str(row["example_id"]): row
        for row in read_jsonl(args.phase2_run_dir / "prompts.jsonl")
    }
    generations = {
        str(row["example_id"]): row
        for row in read_jsonl(args.phase2_run_dir / "best_generations.jsonl")
        if int(row.get("sample_id", 0)) == 0
    }
    p_true_rows = {
        str(row["example_id"]): row
        for row in _read_csv(
            args.phase2_run_dir / "uq_baselines" / "self_report_examples.csv"
        )
    }
    cohort = []
    seen: set[str] = set()
    for label_row in _read_csv(args.labels_path):
        example_id = str(label_row.get("example_id") or "")
        label = str(label_row.get("label") or "").strip().lower()
        label_valid = str(label_row.get("label_valid") or "").strip().lower()
        if label_valid not in {"true", "1", "yes"} or label not in {"correct", "incorrect"}:
            continue
        example = examples.get(example_id)
        if (
            example is None
            or str(example.get("split")) != "test"
            or str(example.get("bioasq_type")) != "summary"
        ):
            continue
        if example_id in seen:
            raise ValueError(f"Duplicate valid label for {example_id}.")
        if example_id not in prompts or example_id not in generations or example_id not in p_true_rows:
            raise ValueError(f"Missing accepted current-condition artifact for {example_id}.")
        seen.add(example_id)
        cohort.append(
            {
                "example_id": example_id,
                "example": example,
                "current_prompt": prompt_record_for_condition(prompts[example_id], "current"),
                "current_main": generations[example_id],
                "current_label": label,
                "current_blind_p_true_uncertainty": _finite_float(
                    p_true_rows[example_id]["p_true_blind_uncertainty"],
                    "p_true_blind_uncertainty",
                ),
            }
        )
    cohort.sort(key=lambda row: row["example_id"])
    if args.max_examples is not None:
        if args.max_examples <= 0:
            raise ValueError("--max-examples must be positive.")
        cohort = cohort[: args.max_examples]
    expected = args.max_examples if args.max_examples is not None else args.expected_examples
    if len(cohort) != expected:
        raise ValueError(f"Expected {expected} summary questions, got {len(cohort)}.")
    return cohort


def _prepare_output(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to use non-empty output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        (args.output_dir / condition).mkdir()


def _condition_score(
    *,
    example_id: str,
    condition: str,
    main_generation: dict[str, Any],
    blind_p_true_uncertainty: float,
    sampled_generations: list[dict[str, Any]],
    cluster_record: dict[str, Any],
) -> dict[str, Any]:
    level4 = score_level4_clusters([cluster_record], sampled_generations)[0]
    cluster_sizes = [int(value) for value in cluster_record["cluster_sizes"]]
    main_stats = text_length_stats(_answer(main_generation))
    sample_stats = [text_length_stats(_answer(row)) for row in sampled_generations]
    return {
        "example_id": example_id,
        "condition": condition,
        "blind_p_true_uncertainty": blind_p_true_uncertainty,
        "discrete_semantic_entropy": _finite_float(
            level4["discrete_semantic_entropy"], "discrete_semantic_entropy"
        ),
        "num_clusters": len(cluster_sizes),
        "cluster_sizes": json.dumps(cluster_sizes),
        "largest_cluster_fraction": largest_cluster_fraction(cluster_sizes),
        "normalized_nll_10_samples": statistics.fmean(
            _finite_float(row["normalized_nll"], "normalized_nll")
            for row in sampled_generations
        ),
        "main_generated_tokens": int(main_generation.get("num_generated_tokens") or 0),
        "main_word_count": main_stats["word_count"],
        "main_sentence_count": main_stats["sentence_count"],
        "main_one_or_two_sentence_compliant": int(
            main_stats["one_or_two_sentence_compliant"]
        ),
        "sample_generated_tokens_mean": statistics.fmean(
            int(row.get("num_generated_tokens") or 0) for row in sampled_generations
        ),
        "sample_word_count_mean": statistics.fmean(
            int(row["word_count"]) for row in sample_stats
        ),
        "sample_sentence_count_mean": statistics.fmean(
            int(row["sentence_count"]) for row in sample_stats
        ),
        "sample_one_or_two_sentence_compliance_rate": statistics.fmean(
            bool(row["one_or_two_sentence_compliant"]) for row in sample_stats
        ),
    }


def _write_static_condition_inputs(
    output_dir: Path,
    cohort: list[dict[str, Any]],
    condition: str,
) -> None:
    condition_dir = output_dir / condition
    examples = [row["example"] for row in cohort]
    prompts = [
        (
            row["current_prompt"]
            if condition == "current"
            else prompt_record_for_condition(row["current_prompt"], "short")
        )
        for row in cohort
    ]
    _append_jsonl(condition_dir / "examples.jsonl", examples)
    _append_jsonl(condition_dir / "prompts.jsonl", prompts)


def main() -> int:
    args = parse_args()
    if args.device != "cuda":
        raise ValueError("The formal intervention requires --device cuda.")
    if args.num_samples != 10:
        raise ValueError("The frozen protocol requires exactly ten high-temperature samples.")
    required = (
        args.phase2_run_dir / "examples.jsonl",
        args.phase2_run_dir / "prompts.jsonl",
        args.phase2_run_dir / "best_generations.jsonl",
        args.phase2_run_dir / "uq_baselines" / "self_report_examples.csv",
        args.labels_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    _prepare_output(args)
    cohort = load_cohort(args)
    _append_jsonl(
        args.output_dir / "cohort_manifest.jsonl",
        [
            {
                "example_id": row["example_id"],
                "split": "test",
                "bioasq_type": "summary",
                "current_correctness_label": row["current_label"],
            }
            for row in cohort
        ],
    )
    for condition in CONDITIONS:
        _write_static_condition_inputs(args.output_dir, cohort, condition)

    status_path = args.output_dir / "status.json"
    status_path.write_text(
        json.dumps({"status": "loading_models", "examples": len(cohort)}, indent=2) + "\n",
        encoding="utf-8",
    )
    generator = HuggingFaceCausalLMGenerator(
        GenerationConfig(
            model_name=args.model_name,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.sample_temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
        )
    )
    nli_scorer = HuggingFaceNLIScorer(
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
    started = time.monotonic()
    status_path.write_text(
        json.dumps({"status": "running", "examples": len(cohort), "completed": 0}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    score_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in CONDITIONS}
    for completed, row in enumerate(cohort, start=1):
        example_id = row["example_id"]
        example = row["example"]
        current_prompt = row["current_prompt"]
        short_prompt = prompt_record_for_condition(current_prompt, "short")
        current_main = row["current_main"]
        short_main = clean_generation_records(
            generate_answer_records(
                [short_prompt],
                generator,
                num_samples=1,
                model_name=args.model_name,
                generation_level="summary_length_short_main_t0_1",
                include_token_scores=True,
                temperature=args.main_temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        )[0]
        short_p_true = generator.binary_continuation_probability(
            build_p_true_prompt(example, _answer(short_main)),
            true_text=" True",
            false_text=" False",
        )

        condition_inputs = {
            "current": (
                current_prompt,
                current_main,
                row["current_blind_p_true_uncertainty"],
            ),
            "short": (short_prompt, short_main, 1.0 - float(short_p_true)),
        }
        for condition, (prompt, main_generation, p_true_uncertainty) in condition_inputs.items():
            sampled = clean_generation_records(
                generate_answer_records(
                    [prompt],
                    generator,
                    num_samples=args.num_samples,
                    model_name=args.model_name,
                    generation_level=f"summary_length_{condition}_high_temperature",
                    include_token_scores=True,
                    temperature=args.sample_temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                )
            )
            clusters = cluster_by_bidirectional_entailment(
                sampled,
                scorer=nli_scorer,
                examples_by_id={example_id: example},
                strict_entailment=False,
            )
            if len(clusters) != 1:
                raise RuntimeError(f"Expected one cluster record for {example_id}/{condition}.")
            condition_dir = args.output_dir / condition
            _append_jsonl(condition_dir / "best_generations.jsonl", [main_generation])
            _append_jsonl(condition_dir / "generations.jsonl", sampled)
            _append_jsonl(condition_dir / "cleaned_generations.jsonl", sampled)
            _append_jsonl(condition_dir / "clusters.jsonl", clusters)
            score_rows[condition].append(
                _condition_score(
                    example_id=example_id,
                    condition=condition,
                    main_generation=main_generation,
                    blind_p_true_uncertainty=float(p_true_uncertainty),
                    sampled_generations=sampled,
                    cluster_record=clusters[0],
                )
            )
        elapsed = time.monotonic() - started
        eta = elapsed * (len(cohort) - completed) / completed
        print(
            f"[summary-length] {completed}/{len(cohort)} example={example_id} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
            flush=True,
        )
        status_path.write_text(
            json.dumps(
                {
                    "status": "running",
                    "examples": len(cohort),
                    "completed": completed,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    for condition in CONDITIONS:
        _write_csv(args.output_dir / condition / "condition_scores.csv", score_rows[condition])
    summary = {
        "schema_version": "bioasq_summary_length_collection_v1",
        "status": "complete",
        "examples": len(cohort),
        "cohort": "valid-Claude-labelled Phase-2 test summary questions",
        "conditions": list(CONDITIONS),
        "current_reuse": {
            "main_answer": True,
            "claude_correctness_label": True,
            "blind_p_true": True,
            "high_temperature_samples": False,
            "reason": (
                "The efficiency benchmark retained aggregate UQ scores but not sampled "
                "answer texts or full clusters needed for paired length diagnostics."
            ),
        },
        "short_generation": {
            "main_answer_temperature": args.main_temperature,
            "high_temperature_samples": args.num_samples,
            "sample_temperature": args.sample_temperature,
        },
        "frozen_generation": {
            "model_name": args.model_name,
            "seed": args.seed,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "no_new_word_or_token_cap": True,
        },
        "nli": {
            "model_name": args.nli_model_name,
            "condition_on_question": True,
            "strict_entailment": False,
        },
        "input_sha256": {
            str(path.relative_to(args.phase2_run_dir.parent)): sha256_file(path)
            for path in required[:-1]
        }
        | {"labels_path": sha256_file(args.labels_path)},
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    status_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "examples": len(cohort),
                "completed": len(cohort),
                "elapsed_seconds": summary["elapsed_seconds"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Completed summary-length collection: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
