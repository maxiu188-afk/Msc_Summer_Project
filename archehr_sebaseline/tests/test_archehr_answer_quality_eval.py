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
from archehr_sebaseline.evaluation.archehr_answer_quality import (
    evaluate_generation_quality,
    evaluate_archehr_se_run,
    extract_citations_from_answer_text,
    load_archehr_key,
    prf,
    rouge_l_f1,
    token_f1,
)
from archehr_sebaseline.semantic_scores import SCORE_FIELDS
from archehr_sebaseline.citation_uq import CITATION_UQ_FIELDS
from archehr_sebaseline.baseline_uq import GENERATION_UQ_FIELDS


class ArchEHRAnswerQualityEvalTests(unittest.TestCase):
    def test_text_and_citation_metrics(self) -> None:
        precision, recall, f1 = prf({"S1", "S3"}, {"S1", "S2"})
        self.assertAlmostEqual(precision, 0.5)
        self.assertAlmostEqual(recall, 0.5)
        self.assertAlmostEqual(f1, 0.5)
        self.assertAlmostEqual(
            token_f1("patient had a rash after amoxicillin", "rash after amoxicillin"),
            2 / 3,
        )
        self.assertAlmostEqual(
            rouge_l_f1("rash appeared after amoxicillin", "rash after amoxicillin"),
            6 / 7,
        )

    def test_load_archehr_key_extracts_answers_and_sentence_labels(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_archehr_eval"
        key_path = output_dir / "key.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        key_path.write_text(
            json.dumps(
                {
                    "examples": [
                        {
                            "case_id": "1",
                            "clinician_answer": "Medication was stopped because of rash.",
                            "sentence_relevance": {"1": "essential", "S2": "supplementary"},
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        loaded = load_archehr_key(key_path)
        self.assertEqual(loaded["1"]["gold_answer"], "Medication was stopped because of rash.")
        self.assertEqual(loaded["1"]["essential_sentence_ids"], ["S1"])
        self.assertEqual(loaded["1"]["lenient_sentence_ids"], ["S1", "S2"])

    def test_load_archehr_key_supports_official_answers_relevance_schema(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_archehr_eval"
        key_path = output_dir / "key_official_schema.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        key_path.write_text(
            json.dumps(
                [
                    {
                        "case_id": "1",
                        "clinician_answer": "The treatment helped [2, 5].",
                        "answers": [
                            {"sentence_id": "1", "relevance": "not-relevant"},
                            {"sentence_id": "2", "relevance": "essential"},
                            {"sentence_id": "3", "relevance": "supplementary"},
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded = load_archehr_key(key_path)
        self.assertEqual(loaded["1"]["essential_sentence_ids"], ["S2"])
        self.assertEqual(loaded["1"]["supplementary_sentence_ids"], ["S3"])
        self.assertEqual(loaded["1"]["lenient_sentence_ids"], ["S2", "S3"])

    def test_extract_citations_from_answer_text(self) -> None:
        self.assertEqual(
            extract_citations_from_answer_text("Text [2,5] and more [6; 7]."),
            {"S2", "S5", "S6", "S7"},
        )

    def test_reference_only_quality_ignores_missing_evidence_labels(self) -> None:
        row = evaluate_generation_quality(
            {
                "example_id": "21",
                "sample_id": 0,
                "citation_ids": ["S1"],
                "clean_answer": "The patient had liver failure and kidney failure.",
            },
            {
                "example_id": "21",
                "gold_answer": "The patient's lifespan was shortened by liver failure and kidney failure.",
                "essential_sentence_ids": [],
                "lenient_sentence_ids": [],
            },
            quality_threshold=0.4,
        )
        self.assertFalse(row["has_gold_evidence"])
        self.assertIsNone(row["citation_score"])
        self.assertGreater(row["quality_score"], 0.0)
        self.assertAlmostEqual(row["quality_score"], row["relevance_score"])

    def test_evaluate_archehr_se_run_writes_artifacts(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_archehr_eval"
        run_dir = output_dir / "run"
        eval_dir = output_dir / "eval"
        key_path = output_dir / "key_full.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        examples = [
            {
                "id": "1",
                "dataset": "archehr_qa",
                "split": "dev",
                "question": "Why was the medication stopped?",
                "evidence_sentence_ids": ["S1", "S2"],
            },
            {
                "id": "2",
                "dataset": "archehr_qa",
                "split": "dev",
                "question": "Was there enough evidence?",
                "evidence_sentence_ids": ["S1", "S2"],
            },
        ]
        generations = [
            {
                "example_id": "1",
                "dataset": "archehr_qa",
                "split": "dev",
                "sample_id": 0,
                "parse_status": "json",
                "citation_ids": ["S1"],
                "clean_answer": "Medication was stopped because of rash.",
            },
            {
                "example_id": "1",
                "dataset": "archehr_qa",
                "split": "dev",
                "sample_id": 1,
                "parse_status": "json",
                "citation_ids": ["S2"],
                "clean_answer": "The notes discuss medication.",
            },
            {
                "example_id": "2",
                "dataset": "archehr_qa",
                "split": "dev",
                "sample_id": 0,
                "parse_status": "json",
                "citation_ids": [],
                "clean_answer": "Yes, it was clearly documented.",
            },
            {
                "example_id": "2",
                "dataset": "archehr_qa",
                "split": "dev",
                "sample_id": 1,
                "parse_status": "json",
                "citation_ids": ["S1"],
                "clean_answer": "Evidence is insufficient.",
            },
        ]
        answer_scores = [
            {
                "example_id": "1",
                "num_samples": "2",
                "num_clusters": "1",
                "cluster_sizes": "[2]",
                "semantic_entropy": "0.0",
                "normalized_semantic_entropy": "0.0",
                "clustering_method": "nli",
            },
            {
                "example_id": "2",
                "num_samples": "2",
                "num_clusters": "2",
                "cluster_sizes": "[1, 1]",
                "semantic_entropy": "0.6931471806",
                "normalized_semantic_entropy": "1.0",
                "clustering_method": "nli",
            },
        ]
        citation_rows = []
        for example_id, score in [("1", "0.0"), ("2", "1.0")]:
            row = {field: "" for field in CITATION_UQ_FIELDS}
            row.update(
                {
                    "example_id": example_id,
                    "dataset": "archehr_qa",
                    "split": "dev",
                    "num_samples": "2",
                    "num_unique_citation_sets": "1" if example_id == "1" else "2",
                    "citation_set_entropy": score,
                    "normalized_citation_set_entropy": score,
                    "mean_pairwise_citation_jaccard": "1.0" if example_id == "1" else "0.0",
                }
            )
            citation_rows.append(row)
        generation_uq_rows = []
        for generation in generations:
            row = {field: "" for field in GENERATION_UQ_FIELDS}
            row.update(
                {
                    "example_id": generation["example_id"],
                    "dataset": "archehr_qa",
                    "split": "dev",
                    "sample_id": str(generation["sample_id"]),
                    "model_name": "test",
                    "generation_level": "archehr_se",
                    "normalized_nll": "0.1" if generation["example_id"] == "1" else "0.9",
                    "mean_token_entropy": "0.1" if generation["example_id"] == "1" else "0.9",
                    "max_token_entropy": "0.2" if generation["example_id"] == "1" else "1.0",
                }
            )
            generation_uq_rows.append(row)
        key_path.write_text(
            json.dumps(
                {
                    "examples": [
                        {
                            "case_id": "1",
                            "clinician_answer": "Medication was stopped because of rash.",
                            "sentence_relevance": {"S1": "essential", "S2": "supplementary"},
                        },
                        {
                            "case_id": "2",
                            "clinician_answer": "Evidence is insufficient.",
                            "sentence_relevance": {"S1": "essential"},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        write_jsonl(examples, run_dir / "examples.jsonl", overwrite=True)
        write_jsonl(generations, run_dir / "cleaned_generations.jsonl", overwrite=True)
        write_csv(answer_scores, run_dir / "answer_se_scores.csv", SCORE_FIELDS, overwrite=True)
        write_csv(citation_rows, run_dir / "citation_uq.csv", CITATION_UQ_FIELDS, overwrite=True)
        write_csv(generation_uq_rows, run_dir / "generation_uq.csv", GENERATION_UQ_FIELDS, overwrite=True)

        result = evaluate_archehr_se_run(
            run_dir,
            key_path,
            output_dir=eval_dir,
            quality_threshold=0.5,
            overwrite=True,
        )
        summary = result["summary"]
        self.assertEqual(summary["num_examples"], 2)
        self.assertEqual(summary["num_generations"], 4)
        self.assertEqual(summary["low_quality_examples"], 1)
        self.assertAlmostEqual(summary["auroc_low_quality_by_score"]["normalized_semantic_entropy"], 1.0)

        with (eval_dir / "answer_quality_examples.csv").open(
            "r", encoding="utf-8", newline=""
        ) as infile:
            rows = list(csv.DictReader(infile))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["is_low_quality"], "false")
        self.assertEqual(rows[1]["is_low_quality"], "true")
        self.assertTrue((eval_dir / "evaluation_report.md").exists())
        self.assertIn("<svg", (eval_dir / "auroc_bar.svg").read_text(encoding="utf-8"))
        self.assertIn("<svg", (eval_dir / "rejection_curve.svg").read_text(encoding="utf-8"))
        self.assertIn("<svg", (eval_dir / "reliability_diagram.svg").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
