"""Run verbalized-confidence and P(True) UQ over completed Level 4 answers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl, write_csv
from archehr_sebaseline.generation import GenerationConfig, HuggingFaceCausalLMGenerator
from archehr_sebaseline.self_report_uq import (
    SELF_REPORT_EXAMPLE_FIELDS,
    SELF_REPORT_GENERATION_FIELDS,
    score_self_report_generations,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute verbalized confidence and P(True) for an existing Level 4 run.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--max_input_tokens", type=int, default=2048)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    output_dir = args.output_dir or run_dir / "uq_baselines"
    generator = HuggingFaceCausalLMGenerator(
        GenerationConfig(
            model_name=args.model_name,
            device=args.device,
            torch_dtype=args.torch_dtype,
            max_input_tokens=args.max_input_tokens,
            local_files_only=args.local_files_only,
        )
    )
    generation_rows, example_rows = score_self_report_generations(
        read_jsonl(run_dir / "examples.jsonl"),
        read_jsonl(run_dir / "cleaned_generations.jsonl"),
        generator,
    )
    write_csv(generation_rows, output_dir / "self_report_generations.csv", SELF_REPORT_GENERATION_FIELDS, overwrite=args.overwrite)
    write_csv(example_rows, output_dir / "self_report_examples.csv", SELF_REPORT_EXAMPLE_FIELDS, overwrite=args.overwrite)
    print(f"Self-report UQ complete: generations={len(generation_rows)}, examples={len(example_rows)}")
    print(f"output_dir: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
