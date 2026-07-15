"""Run fixed-subset independent LLM-judge validation for a BioASQ run."""

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

from archehr_sebaseline.evaluation.bioasq_llm_judge import run_llm_judge_validation
from archehr_sebaseline.generation import GenerationConfig, HuggingFaceCausalLMGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BioASQ quality with a fixed LLM judge.")
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--judge_model_name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sample_size", type=int, default=30)
    parser.add_argument("--selection_seed", type=int, default=20260715)
    parser.add_argument("--judge_low_quality_threshold", type=float, default=0.5)
    parser.add_argument("--max_new_tokens", type=int, default=160)
    parser.add_argument("--max_input_tokens", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "llm_judge"
    generator = HuggingFaceCausalLMGenerator(
        GenerationConfig(
            model_name=args.judge_model_name,
            num_samples=1,
            max_new_tokens=args.max_new_tokens,
            seed=args.selection_seed,
            device=args.device,
            max_input_tokens=args.max_input_tokens,
            local_files_only=args.local_files_only,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )
    started = time.monotonic()

    def progress(completed: int, total: int) -> None:
        elapsed = time.monotonic() - started
        rate = elapsed / completed
        remaining = rate * (total - completed)
        print(
            f"judge {completed}/{total} elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
            flush=True,
        )

    result = run_llm_judge_validation(
        args.run_dir,
        output_dir,
        judge=lambda prompt: generator.generate_deterministic(prompt, max_new_tokens=args.max_new_tokens),
        judge_model_name=args.judge_model_name,
        sample_size=args.sample_size,
        selection_seed=args.selection_seed,
        low_quality_threshold=args.judge_low_quality_threshold,
        overwrite=args.overwrite,
        progress_callback=progress,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
