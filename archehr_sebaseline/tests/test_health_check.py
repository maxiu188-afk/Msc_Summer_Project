from __future__ import annotations

import csv
import json
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

    def test_health_check_accepts_set_aware_nli_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "test_level4_set_aware_health"
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
            summary_path = output_dir / "summary.txt"
            summary_path.write_text(
                summary_path.read_text(encoding="utf-8").replace(
                    "clustering_method: nli_bidirectional_entailment",
                    "clustering_method: nli_set_bidirectional_entailment",
                ),
                encoding="utf-8",
            )
            cluster_path = output_dir / "clusters.jsonl"
            clusters = [json.loads(line) for line in cluster_path.read_text(encoding="utf-8").splitlines()]
            for cluster in clusters:
                cluster["clustering_method"] = "nli_set_bidirectional_entailment"
            cluster_path.write_text(
                "".join(json.dumps(cluster) + "\n" for cluster in clusters),
                encoding="utf-8",
            )
            score_path = output_dir / "se_scores.csv"
            with score_path.open(encoding="utf-8", newline="") as infile:
                scores = list(csv.DictReader(infile))
            fieldnames = list(scores[0])
            for score in scores:
                score["clustering_method"] = "nli_set_bidirectional_entailment"
            with score_path.open("w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(scores)

            result = check_level4_output_dir(output_dir, require_nli=True)

            self.assertTrue(result.passed, result.format_report())

    def test_missing_output_file_fails_health_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "missing_level4_health"
            output_dir.mkdir()

            result = check_level4_output_dir(output_dir)

            self.assertFalse(result.passed)
            self.assertIn("missing required files", result.failures[0])


if __name__ == "__main__":
    unittest.main()
