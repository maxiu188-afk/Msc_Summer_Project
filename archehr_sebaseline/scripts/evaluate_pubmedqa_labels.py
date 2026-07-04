"""CLI for PubMedQA label evaluation on Level 4 artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.pubmedqa_labels import evaluate_level4_pubmedqa_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PubMedQA yes/no/maybe labels from Level 4 outputs."
    )
    parser.add_argument("output_dir", type=Path, help="Directory containing Level 4 artifacts.")
    parser.add_argument("--predictions_path", type=Path, default=None)
    parser.add_argument("--summary_path", type=Path, default=None)
    parser.add_argument("--rejection_curve_path", type=Path, default=None)
    parser.add_argument("--auroc_plot_path", type=Path, default=None)
    parser.add_argument("--rejection_plot_path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_level4_pubmedqa_labels(
        args.output_dir,
        predictions_path=args.predictions_path,
        summary_path=args.summary_path,
        rejection_curve_path=args.rejection_curve_path,
        auroc_plot_path=args.auroc_plot_path,
        rejection_plot_path=args.rejection_plot_path,
        overwrite=args.overwrite,
    )
    summary = result["summary"]
    print("PubMedQA label evaluation complete")
    print(f"examples: {summary['num_examples']}")
    print(f"generations: {summary['num_generations']}")
    print(f"majority_accuracy: {summary['majority_accuracy']:.3f}")
    print(f"unknown_sample_predictions: {summary['unknown_sample_predictions']}")
    print(f"summary_path: {result['paths']['summary']}")
    print(f"auroc_plot_path: {result['paths']['auroc_plot']}")
    print(f"rejection_plot_path: {result['paths']['rejection_plot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
