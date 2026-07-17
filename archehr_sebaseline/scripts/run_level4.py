"""CLI entry point for Level 4 token-score UQ pilot runs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.dataset_adapters import SUPPORTED_DATASETS
from archehr_sebaseline.generation import GenerationConfig, MissingGenerationDependency
from archehr_sebaseline.nli_clustering import NLIConfig
from archehr_sebaseline.pipeline_level4 import format_summary, run_level4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 4 token-score SE pilot.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="pubmedqa")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "level4")
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument(
        "--max_examples",
        "--limit_examples",
        dest="max_examples",
        type=int,
        default=50,
        help="Maximum number of examples to run for the Level 4 pilot.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--clustering_method", choices=["nli", "exact"], default="nli")
    parser.add_argument("--nli_model_name", default="microsoft/deberta-v2-xlarge-mnli")
    parser.add_argument("--nli_device", default=None)
    parser.add_argument("--nli_max_input_tokens", type=int, default=512)
    parser.add_argument("--nli_torch_dtype", default=None)
    parser.add_argument("--nli_local_files_only", action="store_true")
    parser.add_argument("--nli_trust_remote_code", action="store_true")
    parser.add_argument("--strict_entailment", action="store_true")
    parser.add_argument("--no_condition_on_question", action="store_true")
    parser.add_argument(
        "--without_evidence",
        action="store_true",
        help="For BioASQ, omit snippets from the generation prompt while retaining gold metadata for evaluation.",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Print generation and NLI progress with elapsed time and estimated remaining time.",
    )
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
        device=args.nli_device or args.device,
        max_input_tokens=args.nli_max_input_tokens,
        local_files_only=args.nli_local_files_only,
        torch_dtype=args.nli_torch_dtype,
        trust_remote_code=args.nli_trust_remote_code,
        strict_entailment=args.strict_entailment,
        condition_on_question=not args.no_condition_on_question,
    )

    try:
        result = run_level4(
            dataset=args.dataset,
            data_path=args.data_path,
            split=args.split,
            output_dir=args.output_dir,
            config=config,
            limit_examples=args.max_examples,
            overwrite=args.overwrite,
            clustering_method=args.clustering_method,
            nli_config=nli_config,
            show_progress=args.show_progress,
            include_evidence=not args.without_evidence,
        )
    except MissingGenerationDependency as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    print(format_summary(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
