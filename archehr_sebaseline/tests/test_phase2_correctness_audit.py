from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "analysis" / "prepare_phase2_correctness_audit.py"
SPEC = importlib.util.spec_from_file_location("phase2_correctness_audit", SCRIPT_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class Phase2CorrectnessAuditTests(unittest.TestCase):
    def test_stratified_sample_has_fixed_type_counts_and_no_duplicates(self) -> None:
        rows = []
        for question_type in audit.QUESTION_TYPES:
            for index in range(10):
                rows.append(
                    {
                        "split": "test",
                        "example_id": f"{question_type}-{index}",
                        "bioasq_type": question_type,
                    }
                )
        selected = audit.select_stratified(rows, per_type=4, random_seed=19)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row["example_id"] for row in selected}), 12)
        for question_type in audit.QUESTION_TYPES:
            self.assertEqual(
                sum(row["bioasq_type"] == question_type for row in selected), 4
            )

    def test_reference_uses_exact_for_list_and_ideal_for_summary(self) -> None:
        key, value = audit.reference_for(
            {"id": "l", "bioasq_type": "list", "exact_answers": ["a", "b"]}
        )
        self.assertEqual(key, "exact_answers")
        self.assertIn('"a"', value)
        key, value = audit.reference_for(
            {"id": "s", "bioasq_type": "summary", "ideal_answers": ["answer"]}
        )
        self.assertEqual(key, "ideal_answers")
        self.assertIn("answer", value)

    def test_cohens_kappa_is_one_for_identical_labels(self) -> None:
        labels = ["correct", "incorrect", "correct", "incorrect"]
        self.assertEqual(audit.cohens_kappa(labels, labels), 1.0)


if __name__ == "__main__":
    unittest.main()
