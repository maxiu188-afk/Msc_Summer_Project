from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.answer_parsing import extract_citation_ids, parse_grounded_answer
from archehr_sebaseline.citation_uq import citation_uq_rows, jaccard_similarity
from archehr_sebaseline.generation import GenerationConfig, StaticGenerator
from archehr_sebaseline.pipeline_archehr_se import run_archehr_se


class ArchEHRSETests(unittest.TestCase):
    def test_parse_grounded_answer_json(self) -> None:
        parsed = parse_grounded_answer(
            '[{"statement": "The medication was changed.", "citation": "S2"}]'
        )
        self.assertEqual(parsed["parse_status"], "json")
        self.assertEqual(parsed["citation_ids"], ["S2"])
        self.assertEqual(parsed["answer_text"], "The medication was changed.")

    def test_parse_grounded_answer_fallback_citations(self) -> None:
        self.assertEqual(extract_citation_ids("Supported by [S1, S3] and S2."), ["S1", "S2", "S3"])
        parsed = parse_grounded_answer("The evidence is insufficient [S4].")
        self.assertEqual(parsed["parse_status"], "fallback")
        self.assertEqual(parsed["citation_ids"], ["S4"])

    def test_citation_uq_rows(self) -> None:
        rows = citation_uq_rows(
            [
                {"example_id": "case", "sample_id": 0, "citation_ids": ["S1"], "dataset": "x"},
                {"example_id": "case", "sample_id": 1, "citation_ids": ["S1"], "dataset": "x"},
                {"example_id": "case", "sample_id": 2, "citation_ids": ["S2"], "dataset": "x"},
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["num_unique_citation_sets"], "2")
        self.assertAlmostEqual(float(rows[0]["normalized_citation_set_entropy"]), 0.5793801643)
        self.assertEqual(jaccard_similarity(("S1",), ("S1", "S2")), 0.5)

    def test_run_archehr_se_static_exact(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_archehr_se"
        data_path = output_dir / "fake_archehr.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_path.write_text(
            json.dumps(
                {
                    "id": "case-1",
                    "patient_question": "Why was the medicine stopped?",
                    "clinician_question": "Why was medication discontinued?",
                    "evidence": [
                        {"sentence_id": "S1", "text": "The patient developed a rash."},
                        {"sentence_id": "S2", "text": "The medication was discontinued."},
                    ],
                    "gold_relevant_sentence_ids": ["S1", "S2"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = run_archehr_se(
            data_path=data_path,
            output_dir=output_dir,
            config=GenerationConfig(model_name="static", num_samples=3),
            generator=StaticGenerator(
                [
                    '[{"statement": "The medication was discontinued because of rash.", "citation": "S1"}]',
                    '[{"statement": "The medication was discontinued.", "citation": "S2"}]',
                    '[{"statement": "The medication was discontinued because of rash.", "citation": "S1"}]',
                ]
            ),
            clustering_method="exact",
            overwrite=True,
        )
        self.assertEqual(result["level"], "archehr_se")
        self.assertEqual(result["num_examples"], 1)
        self.assertEqual(result["num_generations"], 3)
        self.assertTrue((output_dir / "answer_se_scores.csv").exists())
        self.assertTrue((output_dir / "citation_uq.csv").exists())
        self.assertTrue((output_dir / "parsed_generations.jsonl").exists())
        self.assertIn(
            "ArchEHR-QA Semantic Entropy Baseline Report",
            (output_dir / "analysis_report.md").read_text(encoding="utf-8"),
        )

        with (output_dir / "citation_uq.csv").open("r", encoding="utf-8", newline="") as infile:
            rows = list(csv.DictReader(infile))
        self.assertEqual(rows[0]["num_unique_citation_sets"], "2")


if __name__ == "__main__":
    unittest.main()
