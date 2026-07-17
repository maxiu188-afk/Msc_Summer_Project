"""Generate the low-temperature accuracy target for an existing BioASQ SE run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.cleaning import clean_generation_records
from archehr_sebaseline.data_io import read_jsonl, write_jsonl
from archehr_sebaseline.generation import GenerationConfig, HuggingFaceCausalLMGenerator, generate_answer_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", type=Path, required=True)
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.run_dir / "best_generations.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite.")
    prompts = read_jsonl(args.run_dir / "prompts.jsonl")
    config = GenerationConfig(
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
    generator = HuggingFaceCausalLMGenerator(config)
    records = generate_answer_records(
        prompts,
        generator,
        num_samples=1,
        model_name=args.model_name,
        generation_level="best_generation",
        include_token_scores=False,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    write_jsonl(clean_generation_records(records), output_path, overwrite=args.overwrite)
    print(f"wrote {len(records)} low-temperature best generations to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
