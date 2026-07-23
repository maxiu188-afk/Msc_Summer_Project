from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from archehr_sebaseline.pubmedqa_transfer import (
    build_pubmedqa_transfer_prompt,
    extract_leading_pubmedqa_label,
    load_pubmedqa_transfer_examples,
)


class PubMedQATransferTests(unittest.TestCase):
    def test_prompt_requires_explanation_without_exposing_reference(self) -> None:
        prompt = build_pubmedqa_transfer_prompt(
            {"question": "Does treatment help?", "gold_answer": "Hidden reference text."}
        )
        self.assertIn("yes, no, or maybe", prompt)
        self.assertIn("one to three sentences", prompt)
        self.assertNotIn("official PubMedQA annotation criteria", prompt)
        self.assertNotIn("Hidden reference text", prompt)
        self.assertNotIn("Evidence snippets:", prompt)

    def test_context_prompt_includes_abstract_but_not_reference(self) -> None:
        prompt = build_pubmedqa_transfer_prompt(
            {
                "question": "Does treatment help?",
                "evidence_sentences": ["The trial found a significant benefit."],
                "gold_answer": "Hidden reference text.",
            },
            include_context=True,
        )
        self.assertIn("PubMed abstract context:", prompt)
        self.assertIn("[C1] The trial found a significant benefit.", prompt)
        self.assertIn("using only", prompt)
        self.assertIn("official PubMedQA annotation criteria", prompt)
        self.assertIn("Choose YES", prompt)
        self.assertIn("Choose NO", prompt)
        self.assertIn("Choose MAYBE only", prompt)
        self.assertIn("Do not use MAYBE to express your own uncertainty.", prompt)
        self.assertIn("non-significant findings", prompt)
        self.assertNotIn("Hidden reference text", prompt)

    def test_leading_label_parser_rejects_label_only_in_explanation(self) -> None:
        self.assertEqual(extract_leading_pubmedqa_label("Yes. The result supports benefit."), "yes")
        self.assertEqual(extract_leading_pubmedqa_label("[maybe] Evidence is mixed."), "maybe")
        self.assertIsNone(extract_leading_pubmedqa_label("The answer may be yes."))

    def test_loader_uses_only_official_test_ids_and_checks_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "ori_pqal.json"
            truth_path = root / "test_ground_truth.json"
            data_path.write_text(
                json.dumps(
                    {
                        "2": {"QUESTION": "Q2?", "CONTEXTS": ["C2"], "LONG_ANSWER": "R2", "final_decision": "no"},
                        "1": {"QUESTION": "Q1?", "CONTEXTS": ["C1"], "LONG_ANSWER": "R1", "final_decision": "yes"},
                        "3": {"QUESTION": "Q3?", "CONTEXTS": ["C3"], "LONG_ANSWER": "R3", "final_decision": "maybe"},
                    }
                ),
                encoding="utf-8",
            )
            truth_path.write_text(json.dumps({"2": "no", "1": "yes"}), encoding="utf-8")
            examples, prompts = load_pubmedqa_transfer_examples(
                data_path, truth_path, expected_examples=2
            )
        self.assertEqual([row["id"] for row in examples], ["1", "2"])
        self.assertTrue(all(row["prompt_evidence_mode"] == "none" for row in examples))
        self.assertTrue(all(row["evidence_mode"] == "none" for row in prompts))
        self.assertNotIn("R1", prompts[0]["prompt"])

    def test_loader_marks_context_condition_for_generation_and_self_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_path = root / "ori_pqal.json"
            truth_path = root / "test_ground_truth.json"
            data_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "QUESTION": "Q1?",
                            "CONTEXTS": ["Abstract evidence."],
                            "LONG_ANSWER": "Reference conclusion.",
                            "final_decision": "yes",
                        }
                    }
                ),
                encoding="utf-8",
            )
            truth_path.write_text(json.dumps({"1": "yes"}), encoding="utf-8")
            examples, prompts = load_pubmedqa_transfer_examples(
                data_path,
                truth_path,
                expected_examples=1,
                include_context=True,
            )
        self.assertEqual(examples[0]["prompt_evidence_mode"], "provided")
        self.assertEqual(prompts[0]["evidence_mode"], "provided")
        self.assertEqual(prompts[0]["prompt_version"], "pubmedqa_context_explanation_v2")
        self.assertIn("Abstract evidence.", prompts[0]["prompt"])
        self.assertNotIn("Reference conclusion.", prompts[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
