"""CLI health check for Level 4 output artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.health_check import check_level4_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Level 4 output artifacts.")
    parser.add_argument("output_dir", type=Path, help="Directory containing Level 4 outputs.")
    parser.add_argument("--expected_examples", type=int, default=None)
    parser.add_argument("--expected_num_samples", type=int, default=None)
    parser.add_argument("--expected_generations", type=int, default=None)
    parser.add_argument("--require_cuda", action="store_true")
    parser.add_argument("--require_nli", action="store_true")
    parser.add_argument("--no_require_token_scores", action="store_true")
    parser.add_argument(
        "--write_report",
        type=Path,
        default=None,
        help="Optional path to write the formatted health-check report.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = check_level4_output_dir(
        args.output_dir,
        expected_examples=args.expected_examples,
        expected_generations=args.expected_generations,
        expected_num_samples=args.expected_num_samples,
        require_cuda=args.require_cuda,
        require_nli=args.require_nli,
        require_token_scores=not args.no_require_token_scores,
    )
    report = result.format_report()
    print(report)
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(report + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
