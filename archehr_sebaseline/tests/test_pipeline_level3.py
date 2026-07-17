from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.generation import GenerationConfig, StaticGenerator
from archehr_sebaseline.pipeline_level3 import run_level3


class PipelineLevel3Tests(unittest.TestCase):
    def test_level3_static_generator_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_level3_static"
            result = run_level3(
                dataset="fake",
                output_dir=output_dir,
                config=GenerationConfig(model_name="static-test", num_samples=3),
                limit_examples=2,
                generator=StaticGenerator(["same answer", "same answer", "different answer"]),
                overwrite=True,
            )
            self.assertEqual(result["level"], "level3")
            self.assertEqual(result["dataset"], "fake")
            self.assertEqual(result["num_examples"], 2)
            self.assertEqual(result["num_generations"], 6)
            self.assertTrue((output_dir / "examples.jsonl").exists())
            self.assertTrue((output_dir / "prompts.jsonl").exists())
            self.assertTrue((output_dir / "generations.jsonl").exists())
            self.assertTrue((output_dir / "se_scores.csv").exists())


if __name__ == "__main__":
    unittest.main()
