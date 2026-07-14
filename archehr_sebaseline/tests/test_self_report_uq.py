from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.self_report_uq import parse_verbalized_confidence, score_self_report_generations


class FakeSelfReportScorer:
    def generate_deterministic(self, prompt: str, *, max_new_tokens: int) -> str:
        del max_new_tokens
        return "80" if "supported" in prompt.lower() else "20"

    def binary_continuation_probability(self, prompt: str, *, true_text: str, false_text: str) -> float:
        del true_text, false_text
        return 0.75 if "supported" in prompt.lower() else 0.25


class SelfReportUQTests(unittest.TestCase):
    def test_confidence_parser(self) -> None:
        self.assertEqual(parse_verbalized_confidence("85"), 0.85)
        self.assertEqual(parse_verbalized_confidence("Confidence: 100.0"), 1.0)
        self.assertIsNone(parse_verbalized_confidence("high"))

    def test_scores_and_aggregates_each_generation(self) -> None:
        examples = [{"id": "a", "dataset": "bioasq", "split": "test", "question": "Q", "evidence_sentences": ["Evidence"]}]
        generations = [
            {"example_id": "a", "sample_id": 0, "clean_answer": "Supported answer."},
            {"example_id": "a", "sample_id": 1, "clean_answer": "Supported answer."},
        ]
        generation_rows, example_rows = score_self_report_generations(examples, generations, FakeSelfReportScorer())
        self.assertEqual(len(generation_rows), 2)
        self.assertAlmostEqual(generation_rows[0]["verbalized_confidence_uncertainty"], 0.2)
        self.assertEqual(generation_rows[0]["p_true_uncertainty"], 0.25)
        self.assertEqual(example_rows[0]["mean_verbalized_confidence"], 0.8)
        self.assertEqual(example_rows[0]["mean_p_true"], 0.75)


if __name__ == "__main__":
    unittest.main()
