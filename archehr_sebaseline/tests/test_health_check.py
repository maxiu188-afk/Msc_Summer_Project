from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.generation import GenerationConfig, StaticGenerator
from archehr_sebaseline.health_check import check_level4_output_dir
from archehr_sebaseline.nli_clustering import ENTAILMENT, NEUTRAL, NLIConfig
from archehr_sebaseline.pipeline_level4 import run_level4


class FakeEntailmentScorer:
    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        *,
        question: str | None = None,
    ) -> str:
        del question
        if "same" in premise and "same" in hypothesis:
            return ENTAILMENT
        if premise == hypothesis:
            return ENTAILMENT
        return NEUTRAL


class HealthCheckTests(unittest.TestCase):
    def test_level4_static_output_passes_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_level4_health"
            run_level4(
                dataset="fake",
                output_dir=output_dir,
                config=GenerationConfig(model_name="static-test", num_samples=3),
                limit_examples=2,
                generator=StaticGenerator(["same answer", "same answer", "different answer"]),
                clustering_method="nli",
                nli_config=NLIConfig(model_name="fake-nli"),
                nli_scorer=FakeEntailmentScorer(),
                overwrite=True,
            )

            result = check_level4_output_dir(
                output_dir,
                expected_examples=2,
                expected_num_samples=3,
                expected_generations=6,
                require_nli=True,
                require_token_scores=True,
            )

            self.assertTrue(result.passed, result.format_report())
            self.assertEqual(result.counts["generations.jsonl"], 6)

    def test_missing_output_file_fails_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "missing_level4_health"
            output_dir.mkdir()

            result = check_level4_output_dir(output_dir)

            self.assertFalse(result.passed)
            self.assertIn("missing required files", result.failures[0])


if __name__ == "__main__":
    unittest.main()
