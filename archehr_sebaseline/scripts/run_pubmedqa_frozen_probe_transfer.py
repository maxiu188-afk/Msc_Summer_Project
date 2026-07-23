#!/usr/bin/env python3
"""Collect PubMedQA artifacts for zero-shot evaluation of frozen BioASQ Probes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl
from archehr_sebaseline.generation import GenerationConfig, HuggingFaceCausalLMGenerator, MissingGenerationDependency
from archehr_sebaseline.phase2_artifacts import (
    FEATURE_BLOCKS,
    sha256_file,
    write_phase2_core_artifacts,
    write_phase2_hidden_state_artifacts,
    write_phase2_self_report_artifacts,
)
from archehr_sebaseline.pubmedqa_transfer import (
    load_pubmedqa_transfer_examples,
    pubmedqa_prompt_version,
    pubmedqa_transfer_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ground-truth-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "generation", "self_report", "hidden_states"), default="all")
    parser.add_argument("--model-name", default="google/gemma-3-12b-it")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--include-context",
        action="store_true",
        help="Include official PubMed abstract CONTEXTS in generation and self-report prompts.",
    )
    parser.add_argument("--max-examples", type=int, default=None, help="Smoke-only deterministic PMID cap.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _write_metadata(args: argparse.Namespace, examples: list[dict[str, object]]) -> None:
    path = args.output_dir / "pubmedqa_transfer_run_metadata.json"
    if path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    payload = {
        "schema_version": "pubmedqa_frozen_probe_transfer_v1",
        "dataset": "PubMedQA PQA-L official test subset",
        "data_sha256": sha256_file(args.data_path),
        "ground_truth_sha256": sha256_file(args.ground_truth_path),
        "selection_policy": "official 500 test PMIDs only; no target fitting, selection, calibration, or threshold tuning",
        "dataset_summary": pubmedqa_transfer_summary(examples, include_context=args.include_context),
        "model_name": args.model_name,
        "prompt_version": pubmedqa_prompt_version(include_context=args.include_context),
        "answer_contract": "leading yes/no/maybe plus a concise 1-to-3-sentence explanation",
        "reference_policy": {
            "official_decision_label": "evaluation_only",
            "long_answer": "saved with examples for later explanation judging; never shown to generation, self-report scoring, or Probes",
            "abstract_context": "generation and self-report input" if args.include_context else "stored but not shown",
            "quality_caveat": "LONG_ANSWER is a dataset reference, not assumed to be a manually curated high-quality explanation",
        },
        "generation": {
            "num_samples": 1,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "max_input_tokens": args.max_input_tokens,
            "evidence_mode": "provided_pubmed_abstract_context" if args.include_context else "none",
        },
        "uq": {
            "included": [
                "blind_p_true_uncertainty",
                "verbalized_confidence_uncertainty",
                "sequence_nll",
                "normalized_nll",
                "mean_token_entropy",
                "max_token_entropy",
            ],
            "excluded": ["semantic_entropy", "cluster_count", "sample_disagreement", "p_true_10"],
        },
        "hidden_state_contract": {
            "transformer_blocks": list(FEATURE_BLOCKS),
            "token_positions": ["TBG", "SLT", "LT"],
            "primary_frozen_probe_feature": "block 24 LT",
        },
        "future_claude_stage": "judge whether the saved explanation agrees with the stored LONG_ANSWER; separate artifact, no Probe input",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.data_path.is_file() or not args.ground_truth_path.is_file():
        print("PubMedQA source or official ground truth is missing.", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.stage in {"all", "generation"} and not args.overwrite:
        print(f"Refusing to write generation into non-empty output directory: {args.output_dir}", file=sys.stderr)
        return 2
    try:
        examples, prompts = load_pubmedqa_transfer_examples(
            args.data_path,
            args.ground_truth_path,
            limit=args.max_examples,
            include_context=args.include_context,
        )
        generator = HuggingFaceCausalLMGenerator(
            GenerationConfig(
                model_name=args.model_name,
                num_samples=1,
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
        started_at = time.monotonic()

        if args.stage in {"all", "generation"}:
            def generation_progress(completed: int, total: int, example_id: str, _: int) -> None:
                elapsed = time.monotonic() - started_at
                remaining = elapsed * (total - completed) / completed if completed else 0.0
                print(f"[generation] {completed}/{total} example={example_id} elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}", flush=True)

            best_generations = write_phase2_core_artifacts(
                output_dir=args.output_dir,
                examples=examples,
                prompts=prompts,
                generator=generator,
                model_name=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                generation_level="pubmedqa_transfer_low_temperature_explained_answer",
                progress_callback=generation_progress,
                overwrite=args.overwrite,
            )
            _write_metadata(args, examples)
        else:
            best_generations = read_jsonl(args.output_dir / "best_generations.jsonl")
            stored_examples = read_jsonl(args.output_dir / "examples.jsonl")
            stored_prompts = read_jsonl(args.output_dir / "prompts.jsonl")
            if {str(row["id"]) for row in stored_examples} != {str(row["id"]) for row in examples}:
                raise ValueError("Stored PubMedQA examples do not match the official test selection.")
            examples, prompts = stored_examples, stored_prompts

        if args.stage in {"all", "self_report"}:
            print("[self_report] scoring verbal confidence and blind P(True)", flush=True)
            write_phase2_self_report_artifacts(
                output_dir=args.output_dir,
                examples=examples,
                best_generations=best_generations,
                generator=generator,
                overwrite=args.overwrite,
            )
        if args.stage in {"all", "hidden_states"}:
            print(f"[hidden_states] extracting blocks={FEATURE_BLOCKS}", flush=True)

            def hidden_progress(completed: int, total: int, example_id: str) -> None:
                elapsed = time.monotonic() - started_at
                remaining = elapsed * (total - completed) / completed if completed else 0.0
                print(f"[hidden_states] {completed}/{total} example={example_id} elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}", flush=True)

            write_phase2_hidden_state_artifacts(
                output_dir=args.output_dir,
                prompts=prompts,
                best_generations=best_generations,
                generator=generator,
                transformer_blocks=FEATURE_BLOCKS,
                splits=("test",),
                progress_callback=hidden_progress,
                overwrite=args.overwrite,
            )
    except (FileNotFoundError, MissingGenerationDependency, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(f"PubMedQA frozen-Probe transfer stage {args.stage!r} completed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
