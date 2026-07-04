from __future__ import annotations

import csv
import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import write_csv, write_jsonl
from archehr_sebaseline.evaluation.pubmedqa_labels import (
    DEFAULT_UNCERTAINTY_FIELDS,
    auroc,
    evaluate_level4_pubmedqa_labels,
    extract_pubmedqa_label,
    majority_vote,
)
from archehr_sebaseline.pipeline_level4 import LEVEL4_SCORE_FIELDS


class PubMedQALabelEvalTests(unittest.TestCase):
    def test_extract_pubmedqa_label_prefers_explicit_decisions(self) -> None:
        self.assertEqual(extract_pubmedqa_label("The answer is: yes. [yes]"), "yes")
        self.assertEqual(extract_pubmedqa_label("[S4] Therefore, the answer is: maybe."), "maybe")
        self.assertEqual(extract_pubmedqa_label("No, the evidence does not support it."), "no")
        self.assertEqual(extract_pubmedqa_label("The evidence is mixed and descriptive."), "unknown")

    def test_majority_vote_ignores_unknown_and_marks_ties(self) -> None:
        self.assertEqual(majority_vote(["yes", "unknown", "yes", "no"])[0], "yes")
        label, count, known_count, is_tie = majority_vote(["yes", "no", "unknown"])
        self.assertEqual(label, "unknown")
        self.assertEqual(count, 1)
        self.assertEqual(known_count, 2)
        self.assertTrue(is_tie)

    def test_auroc_handles_ties(self) -> None:
        self.assertAlmostEqual(auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertIsNone(auroc([1, 1], [0.2, 0.3]))

    def test_evaluate_level4_pubmedqa_labels_writes_artifacts(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_pubmedqa_eval"
        examples = [
            {
                "id": "ex1",
                "dataset": "pubmedqa",
                "split": "test",
                "question": "Q1?",
                "label": "yes",
            },
            {
                "id": "ex2",
                "dataset": "pubmedqa",
                "split": "test",
                "question": "Q2?",
                "label": "no",
            },
        ]
        generations = [
            {"example_id": "ex1", "sample_id": 0, "clean_answer": "The answer is: yes."},
            {"example_id": "ex1", "sample_id": 1, "clean_answer": "Therefore, yes."},
            {"example_id": "ex2", "sample_id": 0, "clean_answer": "The answer is: yes."},
            {"example_id": "ex2", "sample_id": 1, "clean_answer": "No, not supported."},
        ]
        score_rows = []
        for example_id, score in [("ex1", 0.1), ("ex2", 0.8)]:
            row = {field: "" for field in LEVEL4_SCORE_FIELDS}
            row.update(
                {
                    "example_id": example_id,
                    "dataset": "pubmedqa",
                    "split": "test",
                    "num_samples": "2",
                    "num_clusters": "1",
                    "cluster_sizes": "[2]",
                    "clustering_method": "exact",
                }
            )
            for field in DEFAULT_UNCERTAINTY_FIELDS:
                row[field] = str(score)
            score_rows.append(row)

        write_jsonl(examples, output_dir / "examples.jsonl", overwrite=True)
        write_jsonl(generations, output_dir / "cleaned_generations.jsonl", overwrite=True)
        write_csv(score_rows, output_dir / "se_scores.csv", LEVEL4_SCORE_FIELDS, overwrite=True)

        result = evaluate_level4_pubmedqa_labels(output_dir, overwrite=True)
        summary = result["summary"]
        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["num_generations"], 4)
        self.assertEqual(summary["majority_correct"], 1)
        self.assertAlmostEqual(
            summary["auroc_incorrect_by_score"]["normalized_discrete_semantic_entropy"],
            1.0,
        )

        with (output_dir / "pubmedqa_label_predictions.csv").open(
            "r", encoding="utf-8", newline=""
        ) as infile:
            prediction_rows = list(csv.DictReader(infile))
        self.assertEqual(len(prediction_rows), 4)

        loaded_summary = json.loads((output_dir / "pubmedqa_eval_summary.json").read_text())
        self.assertEqual(loaded_summary["majority_correct"], 1)
        self.assertTrue((output_dir / "rejection_curve.csv").exists())
        self.assertIn("<svg", (output_dir / "auroc_bar.svg").read_text(encoding="utf-8"))
        self.assertIn("<svg", (output_dir / "rejection_curve.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
