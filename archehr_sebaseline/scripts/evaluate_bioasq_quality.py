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
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_RELATIVE_RISK_FRACTION,
    evaluate_level4_bioasq,
)
from archehr_sebaseline.nli_clustering import HuggingFaceNLIScorer, NLIConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BioASQ Level 4 artifacts with lightweight official-metric approximations.")
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--quality_threshold", type=float, default=DEFAULT_QUALITY_THRESHOLD, help="Fixed operational low-quality cut-off for SE sensitivity analysis; not an official BioASQ pass mark.")
    parser.add_argument("--quality_target", choices=["sample0", "mean"], default="mean", help="Generation aggregation used for the fixed-threshold quality target.")
    parser.add_argument("--relative_risk_fraction", type=float, default=DEFAULT_RELATIVE_RISK_FRACTION, help="Within-question-type lowest-quality fraction used for threshold-sensitivity AUROC.")
    parser.add_argument("--bootstrap_samples", type=int, default=DEFAULT_BOOTSTRAP_SAMPLES, help="Number of deterministic bootstrap resamples for AUROC and rank-correlation confidence intervals; 0 disables intervals.")
    parser.add_argument("--bootstrap_seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--quality_mode", choices=["reference", "grounded", "three_axis"], default="reference", help="Use lexical reference agreement, NLI evidence grounding, or the three-axis ROUGE/citation/NLI-reference target.")
    parser.add_argument("--grounding_nli_model", default=None, help="Optional Hugging Face NLI model used by grounded quality mode.")
    parser.add_argument("--grounding_nli_device", default="cpu", help="Device for the optional grounding NLI model, for example cuda.")
    parser.add_argument("--grounding_nli_max_input_tokens", type=int, default=512)
    parser.add_argument("--grounding_max_claims", type=int, default=4)
    parser.add_argument("--grounding_max_evidence_sentences", type=int, default=10)
    parser.add_argument("--reference_nli_model", default=None, help="Optional Hugging Face NLI model used to classify ideal/exact-answer coverage for three-axis mode.")
    parser.add_argument("--reference_nli_device", default="cpu", help="Device for the ideal-answer NLI classifier, for example cuda.")
    parser.add_argument("--reference_nli_max_input_tokens", type=int, default=512)
    parser.add_argument("--reference_nli_local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quality_mode == "grounded" and not args.grounding_nli_model:
        raise ValueError("--quality_mode grounded requires --grounding_nli_model.")
    if args.quality_mode == "three_axis" and not args.reference_nli_model:
        raise ValueError("--quality_mode three_axis requires --reference_nli_model.")
    grounding_scorer = None
    if args.grounding_nli_model:
        grounding_scorer = HuggingFaceNLIScorer(
            NLIConfig(
                model_name=args.grounding_nli_model,
                device=args.grounding_nli_device,
                max_input_tokens=args.grounding_nli_max_input_tokens,
            )
        )
    reference_nli_scorer = None
    if args.reference_nli_model:
        reference_nli_scorer = HuggingFaceNLIScorer(
            NLIConfig(
                model_name=args.reference_nli_model,
                device=args.reference_nli_device,
                max_input_tokens=args.reference_nli_max_input_tokens,
                local_files_only=args.reference_nli_local_files_only,
            )
        )
    result = evaluate_level4_bioasq(
        args.run_dir,
        output_dir=args.output_dir,
        quality_threshold=args.quality_threshold,
        quality_target=args.quality_target,
        relative_risk_fraction=args.relative_risk_fraction,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        quality_mode=args.quality_mode,
        grounding_scorer=grounding_scorer,
        reference_nli_scorer=reference_nli_scorer,
        max_claims=args.grounding_max_claims,
        max_evidence_sentences=args.grounding_max_evidence_sentences,
        overwrite=args.overwrite,
    )
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
