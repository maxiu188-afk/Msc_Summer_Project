from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.generation import GenerationConfig, StaticGenerator
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


class PipelineLevel4Tests(unittest.TestCase):
    def test_level4_static_generator_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_level4_static"
            result = run_level4(
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
            self.assertEqual(result["level"], "level4")
            self.assertEqual(result["dataset"], "fake")
            self.assertEqual(result["num_examples"], 2)
            self.assertEqual(result["num_generations"], 6)
            self.assertEqual(result["num_best_generations"], 2)
            self.assertTrue((output_dir / "examples.jsonl").exists())
            self.assertTrue((output_dir / "generations.jsonl").exists())
            self.assertTrue((output_dir / "best_generations.jsonl").exists())
            self.assertTrue((output_dir / "generation_uq.csv").exists())
            self.assertTrue((output_dir / "example_uq.csv").exists())
            self.assertTrue((output_dir / "se_scores.csv").exists())
            self.assertEqual(result["clustering_method"], "nli_bidirectional_entailment")

    def test_level4_progress_reports_generation_and_nli(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_level4_progress"
            captured = StringIO()
            with redirect_stdout(captured):
                run_level4(
                    dataset="fake",
                    output_dir=output_dir,
                    config=GenerationConfig(model_name="static-test", num_samples=2),
                    limit_examples=1,
                    generator=StaticGenerator(["same answer", "different answer"]),
                    clustering_method="nli",
                    nli_config=NLIConfig(model_name="fake-nli"),
                    nli_scorer=FakeEntailmentScorer(),
                    show_progress=True,
                    overwrite=True,
                )
        progress_output = captured.getvalue()
        self.assertIn("[generation]", progress_output)
        self.assertIn("[nli]", progress_output)
        self.assertIn("[complete]", progress_output)


if __name__ == "__main__":
    unittest.main()
