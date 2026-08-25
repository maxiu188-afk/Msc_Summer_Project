"""CLI entry point for Level 3 common-schema stability runs."""

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
from archehr_sebaseline.pipeline_level3 import format_summary, run_level3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Level 3 common-schema SE stability test.")
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, default="fake")
    parser.add_argument("--data_path", type=Path, default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs" / "level3")
    parser.add_argument("--model_name", default="sshleifer/tiny-gpt2")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch_dtype", default=None)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument(
        "--max_examples",
        "--limit_examples",
        dest="max_examples",
        type=int,
        default=10,
        help="Maximum number of examples to run for the Level 3 stability test.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
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

    try:
        result = run_level3(
            dataset=args.dataset,
            data_path=args.data_path,
            split=args.split,
            output_dir=args.output_dir,
            config=config,
            limit_examples=args.max_examples,
            overwrite=args.overwrite,
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
