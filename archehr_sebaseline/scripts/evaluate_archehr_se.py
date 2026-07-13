"""CLI for lightweight ArchEHR-QA SE baseline evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.archehr_answer_quality import evaluate_archehr_se_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ArchEHR-QA SE outputs with lightweight answer-quality metrics."
    )
    parser.add_argument("--run_dir", type=Path, required=True, help="Directory from run_archehr_se.py.")
    parser.add_argument(
        "--key_path",
        type=Path,
        required=True,
        help="ArchEHR-QA dev key JSON containing clinician answers and relevance labels.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Evaluation output directory. Defaults to RUN_DIR/eval.",
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.4,
        help="Low-quality threshold when gold evidence labels are available.",
    )
    parser.add_argument(
        "--reference_only_threshold",
        type=float,
        default=0.2,
        help="Low-quality threshold when only clinician reference answers are available.",
    )
    parser.add_argument(
        "--quality_target",
        choices=["sample0", "mean"],
        default="sample0",
        help="Example-level target used for SE evaluation.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_archehr_se_run(
        args.run_dir,
        args.key_path,
        output_dir=args.output_dir,
        quality_threshold=args.quality_threshold,
        reference_only_threshold=args.reference_only_threshold,
        quality_target=args.quality_target,
        overwrite=args.overwrite,
    )
    summary = result["summary"]
    print("ArchEHR-QA SE evaluation complete")
    print(f"examples: {summary['num_examples']}")
    print(f"generations: {summary['num_generations']}")
    print(f"low_quality_rate: {summary['low_quality_rate']:.3f}")
    print(f"mean_example_quality_score: {summary['mean_example_quality_score']:.3f}")
    print(f"summary_path: {result['paths']['summary']}")
    print(f"report_path: {result['paths']['report']}")
    print(f"auroc_plot_path: {result['paths']['auroc_plot']}")
    print(f"rejection_plot_path: {result['paths']['rejection_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
