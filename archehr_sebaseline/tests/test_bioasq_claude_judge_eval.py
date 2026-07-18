from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

EVALUATOR_PATH = PROJECT_ROOT / "scripts" / "evaluate_bioasq_claude_judge.py"
SPEC = importlib.util.spec_from_file_location("bioasq_claude_judge_evaluator", EVALUATOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load evaluator module from {EVALUATOR_PATH}")
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


class ClaudeJudgeEvaluationTests(unittest.TestCase):
    def test_load_uq_rows_adds_all_derived_uncertainties(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "examples.jsonl").write_text(
                '{"id": "q1", "bioasq_type": "factoid"}\n', encoding="utf-8"
            )
            for filename, fieldnames, row in (
                (
                    "se_scores.csv",
                    ["example_id", "mean_token_entropy", "max_token_entropy"],
                    {"example_id": "q1", "mean_token_entropy": "0.2", "max_token_entropy": "0.7"},
                ),
                (
                    "example_uq.csv",
                    ["example_id", "mean_token_logprob", "sample_consistency_exact"],
                    {"example_id": "q1", "mean_token_logprob": "-1.25", "sample_consistency_exact": "0.4"},
                ),
            ):
                with (run_dir / filename).open("w", encoding="utf-8", newline="") as outfile:
                    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row)

            rows = EVALUATOR.load_uq_rows(run_dir)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bioasq_type"], "factoid")
        self.assertAlmostEqual(float(rows[0]["avg_token_logprob_uncertainty"]), 1.25)
        self.assertAlmostEqual(float(rows[0]["sample_consistency_exact_uncertainty"]), 0.6)
        self.assertIn("max_token_entropy", EVALUATOR.UQ_SCORE_NAMES)
        self.assertIn("p_true_blind_uncertainty", EVALUATOR.UQ_SCORE_NAMES)
        self.assertNotIn("p_true_with_samples_uncertainty", EVALUATOR.UQ_SCORE_NAMES)


if __name__ == "__main__":
    unittest.main()
