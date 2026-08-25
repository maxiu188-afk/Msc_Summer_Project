"""CLI for the simple ArchEHR-QA Semantic Entropy baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.generation import GenerationConfig
from archehr_sebaseline.nli_clustering import NLIConfig
from archehr_sebaseline.pipeline_archehr_se import run_archehr_se


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ArchEHR-QA Semantic Entropy baseline.")
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--model_name", default="google/gemma-3-12b-it")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_input_tokens", type=int, default=4096)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--torch_dtype", default="bfloat16")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--clustering_method", choices=["exact", "nli"], default="nli")
    parser.add_argument("--nli_model_name", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_local_files_only", action="store_true")
    parser.add_argument("--nli_torch_dtype", default=None)
    parser.add_argument("--nli_strict_entailment", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = GenerationConfig(
        model_name=args.model_name,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        max_input_tokens=args.max_input_tokens,
        local_files_only=args.local_files_only,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    nli_config = NLIConfig(
        model_name=args.nli_model_name,
        device=args.device,
        local_files_only=args.nli_local_files_only,
        torch_dtype=args.nli_torch_dtype,
        strict_entailment=args.nli_strict_entailment,
    )
    result = run_archehr_se(
        data_path=args.data_path,
        split=args.split,
        output_dir=args.output_dir,
        config=config,
        limit_examples=args.max_examples,
        overwrite=args.overwrite,
        clustering_method=args.clustering_method,
        nli_config=nli_config,
    )
    print("ArchEHR-QA SE baseline complete")
    print(f"examples: {result['num_examples']}")
    print(f"generations: {result['num_generations']}")
    print(f"output_dir: {result['output_dir']}")
    print(f"clustering_method: {result['clustering_method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
