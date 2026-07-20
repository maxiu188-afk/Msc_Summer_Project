from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_splits import (
    build_phase2_bioasq_split_manifest,
    validate_phase2_bioasq_split_manifest,
)


class Phase2BioASQSplitTests(unittest.TestCase):
    def _examples(self) -> list[dict[str, str]]:
        examples = []
        for question_type in ("factoid", "list", "summary"):
            for index in range(20):
                examples.append(
                    {
                        "id": f"{question_type}-{index}",
                        "bioasq_type": question_type,
                        "question": f"{question_type} question {index}?",
                    }
                )
        examples.append(
            {
                "id": "factoid-duplicate",
                "bioasq_type": "factoid",
                "question": "Factoid question 19!",
            }
        )
        return examples

    def test_phase1_questions_are_train_only_and_types_are_stratified(self) -> None:
        manifest, summary = build_phase2_bioasq_split_manifest(
            self._examples(),
            phase1_reference_ids=["factoid-0", "list-0", "summary-0"],
            seed=123,
        )
        assignments = {row["example_id"]: row for row in manifest}
        self.assertTrue(all(assignments[example_id]["split"] == "train" for example_id in ("factoid-0", "list-0", "summary-0")))
        self.assertEqual(assignments["factoid-19"]["split"], assignments["factoid-duplicate"]["split"])
        for question_type in ("factoid", "list", "summary"):
            self.assertLessEqual(abs(summary["counts_by_split_and_type"]["validation"][question_type] - 2), 1)
            self.assertLessEqual(abs(summary["counts_by_split_and_type"]["test"][question_type] - 2), 1)

    def test_split_is_reproducible_and_has_no_cross_split_ids(self) -> None:
        first, _ = build_phase2_bioasq_split_manifest(self._examples(), phase1_reference_ids=[], seed=71)
        second, _ = build_phase2_bioasq_split_manifest(self._examples(), phase1_reference_ids=[], seed=71)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len({row["example_id"] for row in first}))
        self.assertEqual(Counter(row["split"] for row in first), Counter(row["split"] for row in second))
        validate_phase2_bioasq_split_manifest(first, phase1_reference_ids=[])

    def test_phase1_id_must_exist_in_full_dataset(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent"):
            build_phase2_bioasq_split_manifest(self._examples(), phase1_reference_ids=["missing-id"])


if __name__ == "__main__":
    unittest.main()
