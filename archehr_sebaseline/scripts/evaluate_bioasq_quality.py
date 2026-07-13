"""CLI for the lightweight BioASQ Task B evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.bioasq_quality import (
    DEFAULT_QUALITY_THRESHOLD,
    evaluate_level4_bioasq,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BioASQ Level 4 artifacts with lightweight official-metric approximations.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--quality_threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD, help="Operational low-quality cut-off for SE analysis; not an official BioASQ pass mark.")
    parser.add_argument("--quality_target", choices=["sample0", "mean"], default="mean", help="Generation aggregation used for each example's SE risk label.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_level4_bioasq(args.run_dir, output_dir=args.output_dir, quality_threshold=args.quality_threshold, quality_target=args.quality_target, overwrite=args.overwrite)
    summary = result["summary"]
    print("Lightweight BioASQ evaluation complete")
    print(f"examples: {summary['num_examples']}")
    print(f"generations: {summary['num_generations']}")
    print(f"mean_example_quality_score: {summary['mean_example_quality_score']:.3f}")
    print(f"low_quality_rate: {summary['low_quality_rate']:.3f}")
    print(f"summary_path: {result['paths']['summary']}")
    print(f"report_path: {result['paths']['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
