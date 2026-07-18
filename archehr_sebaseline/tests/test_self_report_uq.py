from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.self_report_uq import parse_verbalized_confidence, score_self_report_best_answers


class FakeSelfReportScorer:
    def generate_deterministic(self, prompt: str, *, max_new_tokens: int) -> str:
        del max_new_tokens
        return "80" if "supported" in prompt.lower() else "20"

    def binary_continuation_probability(self, prompt: str, *, true_text: str, false_text: str) -> float:
        del true_text, false_text
        if "other possible answers sampled" in prompt.lower():
            return 0.75
        return 0.50


class SelfReportUQTests(unittest.TestCase):
    def test_confidence_parser(self) -> None:
        self.assertEqual(parse_verbalized_confidence("85"), 0.85)
        self.assertEqual(parse_verbalized_confidence("Confidence: 100.0"), 1.0)
        self.assertIsNone(parse_verbalized_confidence("high"))

    def test_scores_low_temperature_answer_with_high_temperature_p_true_context(self) -> None:
        examples = [{"id": "a", "dataset": "bioasq", "split": "test", "question": "Q", "evidence_sentences": ["Evidence"]}]
        best_generations = [
            {"example_id": "a", "sample_id": 0, "clean_answer": "Supported answer."},
        ]
        sampled_generations = [
            {"example_id": "a", "sample_id": 0, "clean_answer": "Alternative one."},
            {"example_id": "a", "sample_id": 1, "clean_answer": "Supported answer."},
        ]
        generation_rows, example_rows = score_self_report_best_answers(
            examples, best_generations, sampled_generations, FakeSelfReportScorer()
        )
        self.assertEqual(len(generation_rows), 1)
        self.assertAlmostEqual(generation_rows[0]["verbalized_confidence_uncertainty"], 0.2)
        self.assertEqual(generation_rows[0]["p_true_uncertainty"], 0.25)
        self.assertEqual(generation_rows[0]["p_true_blind_uncertainty"], 0.5)
        self.assertEqual(generation_rows[0]["p_true_with_samples_uncertainty"], 0.25)
        self.assertEqual(generation_rows[0]["answer_source"], "best_generation_low_temperature")
        self.assertEqual(generation_rows[0]["num_high_temperature_samples"], 2)
        self.assertEqual(example_rows[0]["mean_verbalized_confidence"], 0.8)
        self.assertEqual(example_rows[0]["mean_p_true"], 0.75)
        self.assertEqual(example_rows[0]["mean_p_true_blind"], 0.5)


if __name__ == "__main__":
    unittest.main()
