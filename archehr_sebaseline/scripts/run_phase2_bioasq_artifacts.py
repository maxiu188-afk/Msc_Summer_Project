"""Collect single-answer BioASQ Phase-2 UQ and hidden-state artifacts."""

from __future__ import annotations

import argparse
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
    load_manifest_examples,
    validate_phase2_examples,
    write_phase2_core_artifacts,
    write_phase2_hidden_state_artifacts,
    write_phase2_self_report_artifacts,
    write_run_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "generation", "self_report", "hidden_states"), default="all")
    parser.add_argument("--model_name", default="google/gemma-3-12b-it")
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--max_examples_per_split",
        type=int,
        default=None,
        help="Smoke-only cap applied independently to train, validation, and test.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main() -> int:
    args = parse_args()
    if not args.data_path.is_file():
        print(f"BioASQ data is missing: {args.data_path}", file=sys.stderr)
        return 2
    if not args.split_manifest.is_file():
        print(f"Phase-2 split manifest is missing: {args.split_manifest}", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.stage in {"all", "generation"} and not args.overwrite:
        print(f"Refusing to write generation into non-empty output directory: {args.output_dir}", file=sys.stderr)
        return 2

    try:
        examples, prompts = load_manifest_examples(args.data_path, args.split_manifest)
        if args.max_examples_per_split is not None:
            if args.max_examples_per_split <= 0:
                raise ValueError("--max_examples_per_split must be positive.")
            selected_ids = {
                str(row["id"])
                for split in ("train", "validation", "test")
                for row in [item for item in examples if item["split"] == split][:args.max_examples_per_split]
            }
            examples = [row for row in examples if str(row["id"]) in selected_ids]
            prompts = [row for row in prompts if str(row["example_id"]) in selected_ids]
        counts = validate_phase2_examples(examples)
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
                print(
                    f"[generation] {completed}/{total} example={example_id} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
                    flush=True,
                )

            best_generations = write_phase2_core_artifacts(
                output_dir=args.output_dir,
                examples=examples,
                prompts=prompts,
                generator=generator,
                model_name=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                progress_callback=generation_progress,
                overwrite=args.overwrite,
            )
            write_run_metadata(
                output_dir=args.output_dir,
                data_path=args.data_path,
                split_manifest_path=args.split_manifest,
                model_name=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                max_new_tokens=args.max_new_tokens,
                max_input_tokens=args.max_input_tokens,
                counts=counts,
                overwrite=args.overwrite,
            )
        else:
            best_generations = read_jsonl(args.output_dir / "best_generations.jsonl")
            stored_examples = read_jsonl(args.output_dir / "examples.jsonl")
            stored_prompts = read_jsonl(args.output_dir / "prompts.jsonl")
            if {str(row["id"]) for row in stored_examples} != {str(row["id"]) for row in examples}:
                raise ValueError("Stored examples do not match the supplied Phase-2 manifest.")
            examples, prompts = stored_examples, stored_prompts

        if args.stage in {"all", "self_report"}:
            print("[self_report] scoring verbal confidence and blind P(True) on low-temperature answers", flush=True)
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
                print(
                    f"[hidden_states] {completed}/{total} example={example_id} "
                    f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)}",
                    flush=True,
                )

            write_phase2_hidden_state_artifacts(
                output_dir=args.output_dir,
                prompts=prompts,
                best_generations=best_generations,
                generator=generator,
                transformer_blocks=FEATURE_BLOCKS,
                progress_callback=hidden_progress,
                overwrite=args.overwrite,
            )
    except (FileNotFoundError, MissingGenerationDependency, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print(f"Phase-2 artifact stage {args.stage!r} completed: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
