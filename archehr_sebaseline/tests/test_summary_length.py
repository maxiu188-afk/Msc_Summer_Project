"""Tests for the frozen BioASQ summary-length intervention helpers."""

from __future__ import annotations

import unittest

import importlib.util
from pathlib import Path

from archehr_sebaseline.summary_length import (
    CURRENT_SUMMARY_INSTRUCTION,
    SHORT_SUMMARY_INSTRUCTIONS,
    build_short_summary_prompt,
    largest_cluster_fraction,
    prompt_record_for_condition,
    text_length_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PATH = PROJECT_ROOT / "analysis" / "run_summary_length_comparison.py"
SPEC = importlib.util.spec_from_file_location("summary_length_comparison", ANALYSIS_PATH)
assert SPEC is not None and SPEC.loader is not None
ANALYSIS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYSIS)
REPAIR_PATH = (
    PROJECT_ROOT / "scripts" / "repair_summary_length_sentence_counts.py"
)
REPAIR_SPEC = importlib.util.spec_from_file_location(
    "summary_length_sentence_repair", REPAIR_PATH
)
assert REPAIR_SPEC is not None and REPAIR_SPEC.loader is not None
REPAIR = importlib.util.module_from_spec(REPAIR_SPEC)
REPAIR_SPEC.loader.exec_module(REPAIR)


class SummaryLengthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.record = {
            "example_id": "q1",
            "prompt": "\n".join(
                [
                    "System instruction: direct biomedical answer.",
                    "Task:",
                    CURRENT_SUMMARY_INSTRUCTION,
                    "Answer directly from your biomedical knowledge.",
                ]
            ),
            "prompt_version": "bioasq_summary_direct_v2",
            "evidence_mode": "none",
        }

    def test_short_prompt_only_adds_declared_lines(self) -> None:
        short = build_short_summary_prompt(self.record["prompt"])
        for instruction in SHORT_SUMMARY_INSTRUCTIONS:
            self.assertIn(instruction, short)
        restored = short
        for instruction in SHORT_SUMMARY_INSTRUCTIONS:
            restored = restored.replace("\n" + instruction, "")
        self.assertEqual(restored, self.record["prompt"])

    def test_condition_record_preserves_metadata(self) -> None:
        short = prompt_record_for_condition(self.record, "short")
        self.assertEqual(short["example_id"], "q1")
        self.assertEqual(short["evidence_mode"], "none")
        self.assertEqual(
            short["prompt_version"], "bioasq_summary_direct_one_or_two_sentences_v1"
        )

    def test_prompt_contract_rejects_evidence(self) -> None:
        with self.assertRaises(ValueError):
            prompt_record_for_condition({**self.record, "evidence_mode": "provided"}, "short")

    def test_text_length_stats(self) -> None:
        stats = text_length_stats("First complete sentence. Second complete sentence!")
        self.assertEqual(stats["sentence_count"], 2)
        self.assertEqual(stats["word_count"], 6)
        self.assertTrue(stats["one_or_two_sentence_compliant"])
        self.assertFalse(
            text_length_stats("One. Two. Three.")["one_or_two_sentence_compliant"]
        )

    def test_sentence_count_does_not_split_genus_abbreviation(self) -> None:
        text = (
            "The method was evaluated in C. elegans using RNA sequencing. "
            "It recovered spatial transcription patterns."
        )
        stats = text_length_stats(text)
        self.assertEqual(stats["sentence_count"], 2)
        self.assertTrue(stats["one_or_two_sentence_compliant"])

    def test_largest_cluster_fraction(self) -> None:
        self.assertEqual(largest_cluster_fraction([7, 2, 1]), 0.7)
        with self.assertRaises(ValueError):
            largest_cluster_fraction([])

    def test_paired_analysis_uses_condition_specific_labels(self) -> None:
        def rows(condition: str) -> list[dict[str, str]]:
            values = {
                "q1": (0.1, 0.2, 1, 0.3),
                "q2": (0.9, 0.8, 3, 0.7),
                "q3": (0.2, 0.1, 1, 0.2),
                "q4": (0.8, 0.9, 4, 0.8),
            }
            return [
                {
                    "example_id": example_id,
                    "condition": condition,
                    "blind_p_true_uncertainty": str(scores[0]),
                    "discrete_semantic_entropy": str(scores[1]),
                    "num_clusters": str(scores[2]),
                    "normalized_nll_10_samples": str(scores[3]),
                    "largest_cluster_fraction": "0.7",
                    "main_generated_tokens": "20",
                    "main_word_count": "18",
                    "main_sentence_count": "2",
                    "main_one_or_two_sentence_compliant": "1",
                    "sample_generated_tokens_mean": "22",
                    "sample_word_count_mean": "20",
                    "sample_sentence_count_mean": "2",
                    "sample_one_or_two_sentence_compliance_rate": "0.8",
                }
                for example_id, scores in values.items()
            ]

        per_example, metrics, summary = ANALYSIS.analyze(
            current_rows=rows("current"),
            short_rows=rows("short"),
            current_labels={"q1": 0, "q2": 1, "q3": 0, "q4": 1, "other_split": 1},
            short_labels={"q1": 1, "q2": 1, "q3": 0, "q4": 0},
            bootstrap_resamples=20,
            bootstrap_seed=7,
        )
        self.assertEqual(len(per_example), 4)
        self.assertEqual(summary["paired_correctness_counts"]["current_only_correct"], 1)
        self.assertEqual(summary["paired_correctness_counts"]["short_only_correct"], 1)
        self.assertEqual(len([row for row in metrics if "condition" in row]), 8)

    def test_paired_analysis_excludes_only_missing_label_rows_from_uq(self) -> None:
        def rows() -> list[dict[str, str]]:
            return [
                {
                    "example_id": example_id,
                    "blind_p_true_uncertainty": str(score),
                    "discrete_semantic_entropy": str(score),
                    "num_clusters": str(index + 1),
                    "normalized_nll_10_samples": str(score),
                    "largest_cluster_fraction": "0.8",
                    "main_generated_tokens": "20",
                    "main_word_count": "18",
                    "main_sentence_count": "2",
                    "main_one_or_two_sentence_compliant": "1",
                    "sample_generated_tokens_mean": "22",
                    "sample_word_count_mean": "20",
                    "sample_sentence_count_mean": "2",
                    "sample_one_or_two_sentence_compliance_rate": "1",
                }
                for index, (example_id, score) in enumerate(
                    (("q1", 0.1), ("q2", 0.9), ("q3", 0.2), ("q4", 0.8))
                )
            ]

        per_example, _, summary = ANALYSIS.analyze(
            current_rows=rows(),
            short_rows=rows(),
            current_labels={"q1": 0, "q2": 1, "q3": 0, "q4": 1},
            short_labels={"q1": 0, "q2": 1, "q3": 0},
            bootstrap_resamples=100,
            bootstrap_seed=11,
        )
        self.assertEqual(len(per_example), 4)
        self.assertEqual(summary["collection_examples"], 4)
        self.assertEqual(summary["paired_valid_label_examples"], 3)
        self.assertEqual(summary["missing_paired_label_ids"], ["q4"])
        self.assertEqual(
            summary["paired_correctness_counts"]["missing_paired_label"], 1
        )

    def test_sentence_repair_changes_only_derived_fields(self) -> None:
        scores = [
            {
                "example_id": "q1",
                "blind_p_true_uncertainty": "0.2",
                "discrete_semantic_entropy": "0.3",
                "main_word_count": "999",
                "main_sentence_count": "999",
                "main_one_or_two_sentence_compliant": "0",
                "sample_word_count_mean": "999",
                "sample_sentence_count_mean": "999",
                "sample_one_or_two_sentence_compliance_rate": "0",
            }
        ]
        main = [
            {
                "example_id": "q1",
                "clean_answer": "C. elegans is widely studied. It is a model organism.",
            }
        ]
        samples = [
            {
                "example_id": "q1",
                "sample_id": index,
                "clean_answer": "C. elegans is widely studied. It is a model organism.",
            }
            for index in range(10)
        ]
        repaired = REPAIR.recompute_condition_rows(scores, main, samples)[0]
        self.assertEqual(repaired["blind_p_true_uncertainty"], "0.2")
        self.assertEqual(repaired["discrete_semantic_entropy"], "0.3")
        self.assertEqual(repaired["main_sentence_count"], 2)
        self.assertEqual(repaired["main_one_or_two_sentence_compliant"], 1)
        self.assertEqual(repaired["sample_one_or_two_sentence_compliance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
