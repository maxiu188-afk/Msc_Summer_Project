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
from archehr_sebaseline.evaluation.bioasq_quality import (
    evaluate_generation_quality,
    evaluate_level4_bioasq,
    rouge_n_f1,
    rouge_su4_f1,
)
from archehr_sebaseline.pipeline_level4 import LEVEL4_SCORE_FIELDS


class BioASQQualityEvalTests(unittest.TestCase):
    def test_summary_metrics_strip_citations(self) -> None:
        prediction = "RET causes disease [S1]."
        reference = "RET causes disease."
        self.assertAlmostEqual(rouge_n_f1(prediction, reference), 1.0)
        self.assertAlmostEqual(rouge_su4_f1(prediction, reference), 1.0)

    def test_type_specific_scoring(self) -> None:
        yesno = evaluate_generation_quality({"example_id": "y", "clean_answer": "Yes, supported."}, {"id": "y", "bioasq_type": "yesno", "exact_answers": ["yes"]})
        self.assertEqual(yesno["quality_score"], 1.0)
        factoid = evaluate_generation_quality({"example_id": "f", "clean_answer": "The answer is BRCA1."}, {"id": "f", "bioasq_type": "factoid", "exact_answers": ["BRCA1"]})
        self.assertEqual(factoid["factoid_lenient_accuracy"], 1.0)
        listing = evaluate_generation_quality({"example_id": "l", "clean_answer": "BRCA1, BRCA2."}, {"id": "l", "bioasq_type": "list", "exact_answers": ["BRCA1", "BRCA2"]})
        self.assertEqual(listing["list_f1"], 1.0)

    def test_evaluator_writes_summary_and_se_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1] / "outputs" / "test_bioasq_eval"
        run_dir = root / "run"
        eval_dir = root / "eval"
        examples = [
            {"id": "a", "dataset": "bioasq", "split": "test", "bioasq_type": "summary", "ideal_answers": ["RET causes Hirschsprung disease."]},
            {"id": "b", "dataset": "bioasq", "split": "test", "bioasq_type": "summary", "ideal_answers": ["BRCA1 is a tumor suppressor gene."]},
        ]
        generations = [
            {"example_id": "a", "sample_id": 0, "clean_answer": "RET causes Hirschsprung disease [S1]."},
            {"example_id": "a", "sample_id": 1, "clean_answer": "RET causes Hirschsprung disease."},
            {"example_id": "b", "sample_id": 0, "clean_answer": "No relevant answer."},
            {"example_id": "b", "sample_id": 1, "clean_answer": "No relevant answer."},
        ]
        scores = []
        for example_id, uncertainty in (("a", "0.1"), ("b", "0.9")):
            row = {field: "" for field in LEVEL4_SCORE_FIELDS}
            row.update({"example_id": example_id, "dataset": "bioasq", "split": "test", "num_samples": "2", "num_clusters": "1", "normalized_discrete_semantic_entropy": uncertainty, "normalized_likelihood_weighted_semantic_entropy": uncertainty, "predictive_entropy": uncertainty, "mean_normalized_nll": uncertainty, "mean_token_entropy": uncertainty})
            scores.append(row)
        write_jsonl(examples, run_dir / "examples.jsonl", overwrite=True)
        write_jsonl(generations, run_dir / "cleaned_generations.jsonl", overwrite=True)
        write_csv(scores, run_dir / "se_scores.csv", LEVEL4_SCORE_FIELDS, overwrite=True)
        result = evaluate_level4_bioasq(run_dir, output_dir=eval_dir, quality_threshold=0.15, overwrite=True)
        self.assertEqual(result["summary"]["num_examples"], 2)
        self.assertEqual(result["summary"]["low_quality_examples"], 1)
        self.assertAlmostEqual(result["summary"]["auroc_low_quality_by_score"]["normalized_discrete_semantic_entropy"], 1.0)
        with (eval_dir / "bioasq_quality_examples.csv").open("r", encoding="utf-8", newline="") as infile:
            self.assertEqual(len(list(csv.DictReader(infile))), 2)
        self.assertTrue((eval_dir / "bioasq_eval_report.md").exists())
        self.assertEqual(json.loads((eval_dir / "bioasq_eval_summary.json").read_text(encoding="utf-8"))["evaluator"], "lightweight_bioasq_v1")


if __name__ == "__main__":
    unittest.main()
