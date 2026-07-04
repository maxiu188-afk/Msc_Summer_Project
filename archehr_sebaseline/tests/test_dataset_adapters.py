from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.dataset_adapters import (
    load_pubmedqa_common_examples,
    pubmedqa_records_to_common,
)


class DatasetAdapterTests(unittest.TestCase):
    def test_pubmedqa_record_to_common_schema(self) -> None:
        examples = pubmedqa_records_to_common(
            {
                "123": {
                    "QUESTION": "Does treatment improve survival?",
                    "CONTEXTS": ["Sentence one.", "Sentence two."],
                    "LONG_ANSWER": "Treatment improved survival in this study.",
                    "final_decision": "yes",
                }
            },
            split="pqal",
        )
        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["dataset"], "pubmedqa")
        self.assertEqual(example["id"], "123")
        self.assertEqual(example["split"], "pqal")
        self.assertEqual(example["question"], "Does treatment improve survival?")
        self.assertEqual(example["evidence_sentences"], ["Sentence one.", "Sentence two."])
        self.assertEqual(example["label"], "yes")
        self.assertEqual(example["options"], ["yes", "no", "maybe"])

    def test_load_pubmedqa_common_examples_limit(self) -> None:
        output_dir = Path(__file__).resolve().parents[1] / "outputs" / "test_fixtures"
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "ori_pqal_test.json"
        path.write_text(
            json.dumps(
                {
                    "2": {"QUESTION": "Q2?", "CONTEXTS": ["C2"], "LONG_ANSWER": "A2"},
                    "1": {"QUESTION": "Q1?", "CONTEXTS": ["C1"], "LONG_ANSWER": "A1"},
                }
            ),
            encoding="utf-8",
        )
        examples = load_pubmedqa_common_examples(path, limit=1)
        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["id"], "1")


if __name__ == "__main__":
    unittest.main()
