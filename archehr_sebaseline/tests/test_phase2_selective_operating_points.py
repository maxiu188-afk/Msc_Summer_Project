from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "analysis" / "run_phase2_selective_operating_points.py"
SPEC = importlib.util.spec_from_file_location("phase2_selective_operating_points", SCRIPT_PATH)
assert SPEC and SPEC.loader
operating_points = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = operating_points
SPEC.loader.exec_module(operating_points)


class Phase2SelectiveOperatingPointTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_inclusive_threshold_retains_all_boundary_ties(self) -> None:
        scores = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        threshold = operating_points.select_inclusive_threshold(scores, 0.5)
        metrics = operating_points.evaluate_threshold(
            np.asarray([0, 1, 0, 1], dtype=np.int64), scores, threshold
        )
        self.assertEqual(threshold, 0.0)
        self.assertEqual(metrics["retained_examples"], 3)
        self.assertEqual(metrics["achieved_coverage"], 0.75)

    def test_validation_threshold_can_produce_shifted_test_coverage(self) -> None:
        validation_scores = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        test_scores = np.asarray([0.05, 0.10, 0.15, 0.20], dtype=np.float64)
        threshold = operating_points.select_inclusive_threshold(validation_scores, 0.5)
        metrics = operating_points.evaluate_threshold(
            np.asarray([0, 0, 1, 1], dtype=np.int64), test_scores, threshold
        )
        self.assertEqual(threshold, 0.2)
        self.assertEqual(metrics["achieved_coverage"], 1.0)

    def test_evaluate_threshold_reports_retained_and_rejected_risk(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        metrics = operating_points.evaluate_threshold(labels, scores, 0.2)
        self.assertEqual(metrics["retained_error_risk"], 0.0)
        self.assertEqual(metrics["rejected_error_risk"], 1.0)
        self.assertEqual(metrics["absolute_risk_reduction"], 0.5)
        self.assertEqual(metrics["relative_risk_reduction"], 1.0)

    def test_full_coverage_has_no_rejected_risk(self) -> None:
        labels = np.asarray([0, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2], dtype=np.float64)
        metrics = operating_points.evaluate_threshold(labels, scores, 0.2)
        self.assertEqual(metrics["achieved_coverage"], 1.0)
        self.assertEqual(metrics["rejected_examples"], 0)
        self.assertIsNone(metrics["rejected_error_risk"])

        indexes = np.asarray([[0, 1], [0, 0], [1, 1]], dtype=np.int32)
        intervals = operating_points.bootstrap_intervals(
            labels, np.asarray([True, True]), indexes
        )
        rejected = next(
            row for row in intervals if row["metric"] == "rejected_error_risk"
        )
        self.assertEqual(rejected["valid_resamples"], 0)
        self.assertIsNone(rejected["ci_lower_95"])
        self.assertIsNone(rejected["ci_upper_95"])

    def test_bootstrap_is_deterministic_for_fixed_indexes(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        accepted = np.asarray([True, True, False, False])
        indexes = np.asarray(
            [[0, 1, 2, 3], [0, 0, 2, 2], [1, 1, 3, 3]], dtype=np.int32
        )
        first = operating_points.bootstrap_intervals(labels, accepted, indexes)
        second = operating_points.bootstrap_intervals(labels, accepted, indexes)
        self.assertEqual(first, second)
        self.assertEqual({row["metric"] for row in first}, {
            "achieved_coverage",
            "retained_error_risk",
            "retained_accuracy",
            "rejected_error_risk",
            "absolute_risk_reduction",
            "relative_risk_reduction",
        })

    def test_load_uq_split_checks_rows_types_and_labels(self) -> None:
        rows = []
        for index in range(6):
            rows.append(
                {
                    "example_id": f"e{index}",
                    "bioasq_type": ("factoid", "list", "summary")[index % 3],
                    "incorrect": index % 2,
                    "discrete_semantic_entropy": index / 10,
                    "blind_p_true_uncertainty": index / 10,
                    "accuracy_probe_uncertainty": index / 10,
                    "p_true_probe_uncertainty": index / 10,
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "scores.csv"
            self._write_csv(path, rows)
            loaded = operating_points.load_uq_split(
                path,
                split="validation",
                expected_sha256=None,
                expected_examples=6,
                expected_type_counts={"factoid": 2, "list": 2, "summary": 2},
                expected_errors=3,
            )
        self.assertEqual(len(loaded["example_ids"]), 6)
        self.assertEqual(set(loaded["scores"]), set(operating_points.ALL_METHODS))


if __name__ == "__main__":
    unittest.main()
