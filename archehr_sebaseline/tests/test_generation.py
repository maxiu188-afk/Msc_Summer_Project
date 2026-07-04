from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.generation import GenerationConfig, StaticGenerator, generate_answer_records


class GenerationLevel1Tests(unittest.TestCase):
    def test_generation_config_validates_num_samples(self) -> None:
        with self.assertRaises(ValueError):
            GenerationConfig(num_samples=0).validate()

    def test_generate_answer_records_shape(self) -> None:
        prompts = [
            {"example_id": "ex1", "prompt": "Prompt 1"},
            {"example_id": "ex2", "prompt": "Prompt 2"},
        ]
        records = generate_answer_records(
            prompts,
            StaticGenerator(["A", "B", "C"]),
            num_samples=3,
            model_name="test-model",
        )
        self.assertEqual(len(records), 6)
        self.assertEqual(records[0]["example_id"], "ex1")
        self.assertEqual(records[0]["sample_id"], 0)
        self.assertEqual(records[0]["raw_answer"], "A")
        self.assertEqual(records[-1]["example_id"], "ex2")
        self.assertEqual(records[-1]["sample_id"], 2)
        self.assertEqual(records[-1]["raw_answer"], "C")

    def test_generate_answer_records_can_include_token_scores(self) -> None:
        records = generate_answer_records(
            [{"example_id": "ex1", "prompt": "Prompt 1"}],
            StaticGenerator(["A scored answer"]),
            num_samples=1,
            model_name="test-model",
            include_token_scores=True,
        )
        self.assertEqual(records[0]["raw_answer"], "A scored answer")
        self.assertEqual(records[0]["num_generated_tokens"], 3)
        self.assertEqual(records[0]["sequence_logprob"], -1.5)
        self.assertEqual(records[0]["normalized_nll"], 0.5)
        self.assertEqual(records[0]["mean_token_entropy"], 0.25)


if __name__ == "__main__":
    unittest.main()
