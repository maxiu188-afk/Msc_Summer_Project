from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

SCRIPT_PATH = PROJECT_ROOT / "scripts" / "check_phase1_cohort_alignment.py"
SPEC = importlib.util.spec_from_file_location("phase1_cohort_alignment", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SCRIPT_PATH}.")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def example(example_id: str, question_type: str) -> dict[str, object]:
    return {
        "id": example_id,
        "bioasq_type": question_type,
        "question": f"Question {example_id}",
    }


class Phase1CohortAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = [
            example("f1", "factoid"),
            example("f2", "factoid"),
            example("l1", "list"),
            example("l2", "list"),
            example("s1", "summary"),
            example("s2", "summary"),
        ]

    def test_exact_alignment_requires_same_records_and_counts(self) -> None:
        report = MODULE.compare_cohorts(
            reference_rows=self.reference,
            candidate_rows=list(reversed(self.reference)),
            mode="exact",
            expected_type_limits={"factoid": 2, "list": 2, "summary": 2},
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["candidate_examples"], 6)

    def test_subset_alignment_accepts_phase1_subset(self) -> None:
        report = MODULE.compare_cohorts(
            reference_rows=self.reference,
            candidate_rows=[self.reference[0], self.reference[2], self.reference[4]],
            mode="subset",
            expected_type_limits={"factoid": 1, "list": 1, "summary": 1},
        )
        self.assertTrue(report["candidate_is_phase1_subset"])

    def test_changed_shared_record_fails(self) -> None:
        changed = [dict(row) for row in self.reference]
        changed[0]["question"] = "Changed"
        with self.assertRaises(ValueError):
            MODULE.compare_cohorts(
                reference_rows=self.reference,
                candidate_rows=changed,
                mode="exact",
                expected_type_limits={"factoid": 2, "list": 2, "summary": 2},
            )

    def test_unexpected_subset_id_fails(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.compare_cohorts(
                reference_rows=self.reference,
                candidate_rows=[example("outside", "factoid")],
                mode="subset",
                expected_type_limits={"factoid": 1, "list": 0, "summary": 0},
            )

    def test_candidate_manifest_write_is_exclusive_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cohort.jsonl"
            rows = [self.reference[0], self.reference[2], self.reference[4]]
            MODULE.write_jsonl_exclusive(path, rows)
            saved = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(saved, rows)
            with self.assertRaises(FileExistsError):
                MODULE.write_jsonl_exclusive(path, rows)


if __name__ == "__main__":
    unittest.main()
