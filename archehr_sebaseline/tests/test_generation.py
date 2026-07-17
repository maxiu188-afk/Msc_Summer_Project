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

    def test_generation_overrides_reach_scored_generator_without_retaining_scores(self) -> None:
        class RecordingGenerator:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(self, prompt: str, *, sample_index: int = 0) -> str:
                raise AssertionError("Override generation should use generate_with_scores.")

            def generate_with_scores(self, prompt: str, **kwargs: object) -> dict[str, object]:
                self.calls.append(kwargs)
                return {"raw_answer": "best answer", "sequence_nll": 99.0}

        generator = RecordingGenerator()
        records = generate_answer_records(
            [{"example_id": "ex1", "prompt": "Prompt 1"}],
            generator,
            num_samples=1,
            model_name="test-model",
            temperature=0.1,
            top_p=0.9,
            top_k=50,
        )
        self.assertEqual(generator.calls[0]["temperature"], 0.1)
        self.assertEqual(generator.calls[0]["top_p"], 0.9)
        self.assertEqual(generator.calls[0]["top_k"], 50)
        self.assertEqual(records[0]["raw_answer"], "best answer")
        self.assertNotIn("sequence_nll", records[0])


if __name__ == "__main__":
    unittest.main()
