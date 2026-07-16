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
    build_quality_rows,
    extract_snippet_citation_ids,
    evaluate_generation_quality,
    evaluate_level4_bioasq,
    rouge_n_f1,
    rouge_su4_f1,
)
from archehr_sebaseline.pipeline_level4 import LEVEL4_SCORE_FIELDS
from archehr_sebaseline.self_report_uq import SELF_REPORT_EXAMPLE_FIELDS


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
        self.assertEqual(factoid["factoid_answer_first_accuracy"], 1.0)
        self.assertEqual(factoid["factoid_lenient_accuracy"], 1.0)
        listing = evaluate_generation_quality({"example_id": "l", "clean_answer": "BRCA1, BRCA2."}, {"id": "l", "bioasq_type": "list", "exact_answers": ["BRCA1", "BRCA2"]})
        self.assertEqual(listing["list_f1"], 1.0)

    def test_factoid_list_and_citation_diagnostics_are_answer_aware(self) -> None:
        factoid = evaluate_generation_quality(
            {"example_id": "f", "clean_answer": "The answer is BRCA1. It is a tumour suppressor [S1]."},
            {"id": "f", "bioasq_type": "factoid", "exact_answers": ["BRCA1"], "evidence_sentence_ids": ["S1"]},
        )
        self.assertEqual(factoid["factoid_strict_accuracy"], 1.0)
        self.assertEqual(factoid["factoid_answer_first_accuracy"], 1.0)
        self.assertEqual(factoid["snippet_citation_validity"], 1.0)
        listing = evaluate_generation_quality(
            {"example_id": "l", "clean_answer": "BRCA1 and BRCA2. Supporting evidence [S1, S3]."},
            {"id": "l", "bioasq_type": "list", "exact_answers": ["BRCA1", "BRCA2"], "evidence_sentence_ids": ["S1", "S2"]},
        )
        self.assertEqual(listing["list_f1"], 1.0)
        self.assertEqual(listing["valid_snippet_citation_count"], 1)
        self.assertEqual(listing["invalid_snippet_citation_count"], 1)
        self.assertEqual(extract_snippet_citation_ids("Claim [S1; S2] and another [S3]."), ["S1", "S2", "S3"])

    def test_summary_quality_combines_reference_coverage_with_gold_document_overlap(self) -> None:
        example = {
            "id": "s",
            "bioasq_type": "summary",
            "ideal_answers": ["RET causes Hirschsprung disease."],
            "evidence_sentence_ids": ["S1", "S2"],
            "snippet_documents": ["pubmed/1", "pubmed/2"],
            "documents": ["pubmed/1", "pubmed/2"],
        }
        scored = evaluate_generation_quality(
            {"example_id": "s", "clean_answer": "RET causes Hirschsprung disease [S1]."},
            example,
        )
        self.assertAlmostEqual(scored["reference_coverage_score"], 1.0)
        self.assertAlmostEqual(scored["citation_document_precision"], 1.0)
        self.assertAlmostEqual(scored["citation_document_recall"], 0.5)
        self.assertAlmostEqual(scored["citation_document_f1"], 2.0 / 3.0)
        self.assertAlmostEqual(scored["quality_score"], (2.0 / 3.0) ** 0.5)
        self.assertEqual(scored["quality_metric"], "citation_document_geometric_mean_mean_rouge2_su4_f1")

    def test_missing_gold_documents_preserves_coverage_only_score(self) -> None:
        scored = evaluate_generation_quality(
            {"example_id": "s", "clean_answer": "RET causes Hirschsprung disease [S1]."},
            {"id": "s", "bioasq_type": "summary", "ideal_answers": ["RET causes Hirschsprung disease."]},
        )
        self.assertIsNone(scored["citation_document_f1"])
        self.assertAlmostEqual(scored["quality_score"], 1.0)
        self.assertEqual(scored["quality_metric"], "mean_rouge2_su4_f1")

    def test_relative_quality_risk_keeps_cutoff_ties_together(self) -> None:
        examples = [
            {"id": example_id, "bioasq_type": "yesno", "exact_answers": [expected]}
            for example_id, expected in (("a", "yes"), ("b", "yes"), ("c", "yes"), ("d", "no"))
        ]
        generations = [
            {"example_id": example_id, "sample_id": 0, "clean_answer": answer}
            for example_id, answer in (("a", "no"), ("b", "no"), ("c", "no"), ("d", "no"))
        ]
        _, rows = build_quality_rows(examples, generations, relative_risk_fraction=0.25)
        risky_ids = {row["example_id"] for row in rows if row["is_bottom_quantile_quality"] == "true"}
        self.assertEqual(risky_ids, {"a", "b", "c"})

    def test_grounded_mode_penalizes_contradicted_claims(self) -> None:
        class ControlledScorer:
            def check_implication(self, premise: str, hypothesis: str, *, question: str | None = None) -> str:
                return "contradiction" if "does not" in hypothesis.lower() else "entailment"

        example = {
            "id": "s",
            "bioasq_type": "summary",
            "question": "What does RET cause?",
            "ideal_answers": ["RET causes Hirschsprung disease."],
            "evidence_sentence_ids": ["S1"],
            "evidence_sentences": ["RET causes Hirschsprung disease."],
        }
        good = evaluate_generation_quality(
            {"example_id": "s", "clean_answer": "RET causes Hirschsprung disease [S1]."},
            example,
            quality_mode="grounded",
            grounding_scorer=ControlledScorer(),
        )
        contradicted = evaluate_generation_quality(
            {"example_id": "s", "clean_answer": "RET does not cause Hirschsprung disease [S1]."},
            example,
            quality_mode="grounded",
            grounding_scorer=ControlledScorer(),
        )
        self.assertEqual(good["evidence_entailment_rate"], 1.0)
        self.assertEqual(contradicted["evidence_contradiction_rate"], 1.0)
        self.assertEqual(good["grounded_quality_score"], good["reference_quality_score"])
        self.assertEqual(contradicted["grounded_quality_score"], 0.0)

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
        write_csv(
            [
                {"example_id": "a", "dataset": "bioasq", "split": "test", "num_generations": 2, "mean_verbalized_confidence": 0.9, "verbalized_confidence_uncertainty": 0.1, "mean_p_true": 0.9, "p_true_uncertainty": 0.1},
                {"example_id": "b", "dataset": "bioasq", "split": "test", "num_generations": 2, "mean_verbalized_confidence": 0.1, "verbalized_confidence_uncertainty": 0.9, "mean_p_true": 0.1, "p_true_uncertainty": 0.9},
            ],
            run_dir / "uq_baselines" / "self_report_examples.csv",
            SELF_REPORT_EXAMPLE_FIELDS,
            overwrite=True,
        )
        result = evaluate_level4_bioasq(
            run_dir,
            output_dir=eval_dir,
            quality_threshold=0.15,
            bootstrap_samples=40,
            overwrite=True,
        )
        self.assertEqual(result["summary"]["num_examples"], 2)
        self.assertEqual(result["summary"]["low_quality_examples"], 1)
        self.assertAlmostEqual(result["summary"]["median_example_quality_score"], 0.5)
        self.assertAlmostEqual(result["summary"]["auroc_low_quality_by_score"]["normalized_discrete_semantic_entropy"], 1.0)
        with (eval_dir / "bioasq_quality_examples.csv").open("r", encoding="utf-8", newline="") as infile:
            self.assertEqual(len(list(csv.DictReader(infile))), 2)
        self.assertTrue((eval_dir / "bioasq_eval_report.md").exists())
        self.assertTrue((eval_dir / "bioasq_se_auroc_sensitivity.csv").exists())
        self.assertTrue((eval_dir / "bioasq_se_association.csv").exists())
        summary = json.loads((eval_dir / "bioasq_eval_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["evaluator"], "lightweight_bioasq_v2")
        self.assertIn("bottom_quality_quantile", summary["auroc_sensitivity_by_target"])
        self.assertIn("normalized_discrete_semantic_entropy", summary["risk_spearman_by_score"])
        self.assertAlmostEqual(summary["auroc_low_quality_by_score"]["p_true_uncertainty"], 1.0)


if __name__ == "__main__":
    unittest.main()
