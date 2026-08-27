from __future__ import annotations

import importlib.util
import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "analysis" / "run_phase2_uq_calibration.py"
SPEC = importlib.util.spec_from_file_location("phase2_uq_calibration", SCRIPT_PATH)
assert SPEC and SPEC.loader
calibration = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = calibration
SPEC.loader.exec_module(calibration)


class Phase2UQCalibrationTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def test_scalar_logistic_calibrator_preserves_positive_risk_direction(self) -> None:
        scores = np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0], dtype=np.float64)
        labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
        model = calibration.ScalarLogisticCalibrator(random_seed=7).fit(scores, labels)
        probabilities = model.predict(scores)
        self.assertTrue(np.all(np.diff(probabilities) > 0.0))
        self.assertGreater(model.parameters()["standardized_coefficient"], 0.0)

    def test_equal_frequency_ece_uses_all_examples(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9, 0.4, 0.7], dtype=np.float64)
        ece, rows = calibration.equal_frequency_ece(labels, probabilities, bins=3)
        self.assertEqual(sum(row["examples"] for row in rows), len(labels))
        self.assertGreaterEqual(ece, 0.0)
        self.assertLessEqual(ece, 1.0)

    def test_risk_coverage_retains_lowest_risk_first(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        rows, area = calibration.risk_coverage(labels, scores)
        first = rows[0]
        self.assertEqual(first["requested_coverage"], 0.5)
        self.assertEqual(first["retained_error_risk"], 0.0)
        self.assertEqual(rows[-1]["requested_coverage"], 1.0)
        self.assertEqual(rows[-1]["retained_examples"], len(labels))
        self.assertEqual(rows[-1]["coverage"], 1.0)
        self.assertGreaterEqual(area, 0.0)

    def test_probability_metrics_include_prevalence_skill_baseline(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
        metrics = calibration.probability_metrics(labels, probabilities, bins=2)
        self.assertEqual(metrics["brier_constant_baseline"], 0.25)
        self.assertGreater(metrics["brier_skill"], 0.0)

    def test_load_split_requires_and_joins_the_exact_384_rows(self) -> None:
        base_rows = []
        uq_rows = []
        for index in range(384):
            score = (index + 1) / 385.0
            example_id = f"e{index:03d}"
            base_rows.append(
                {
                    "split": "validation",
                    "example_id": example_id,
                    "bioasq_type": ("factoid", "list", "summary")[index % 3],
                    "incorrect": int(index >= 192),
                    "p_true_probe": score,
                    "accuracy_probe": score,
                    "blind_p_true_uncertainty": score,
                }
            )
            uq_rows.append(
                {
                    "example_id": example_id,
                    "discrete_semantic_entropy": score,
                    "normalized_nll_10_samples": score * 2.0,
                    "p_true_probe_uncertainty": score,
                    "accuracy_probe_uncertainty": score,
                    "blind_p_true_uncertainty": score,
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            uq_path = Path(temporary_directory) / "uq.csv"
            self._write_csv(uq_path, uq_rows)
            loaded = calibration.load_split(base_rows, uq_path, "validation")
        self.assertEqual(len(loaded["example_ids"]), 384)
        self.assertEqual(set(loaded["scores"]), set(calibration.ALL_METHODS))
        np.testing.assert_allclose(
            loaded["scores"]["ten_sample_normalized_nll"],
            2.0 * loaded["scores"]["discrete_semantic_entropy"],
        )

    def test_load_split_can_run_non_se_analysis_without_uq_file(self) -> None:
        base_rows = []
        for index in range(384):
            score = (index + 1) / 385.0
            base_rows.append(
                {
                    "split": "test",
                    "example_id": f"e{index:03d}",
                    "bioasq_type": ("factoid", "list", "summary")[index % 3],
                    "incorrect": int(index >= 192),
                    "p_true_probe": score,
                    "accuracy_probe": score,
                    "blind_p_true_uncertainty": score,
                }
            )
        methods = ("blind_p_true", "accuracy_probe", "p_true_probe")
        loaded = calibration.load_split(base_rows, None, "test", methods=methods)
        self.assertEqual(set(loaded["scores"]), set(methods))

    def test_pubmedqa_transfer_applies_existing_calibrators_without_refit(self) -> None:
        fit_scores = np.linspace(0.01, 0.99, 20)
        fit_labels = np.asarray([0] * 10 + [1] * 10, dtype=np.int64)
        calibrators = {
            method: calibration.ScalarLogisticCalibrator(random_seed=11).fit(
                fit_scores, fit_labels
            )
            for method in calibration.PUBMEDQA_FIELDS
        }
        rows = []
        for index in range(500):
            score = (index + 1) / 501.0
            rows.append(
                {
                    "example_id": f"p{index:03d}",
                    "incorrect": int(index >= 250),
                    "p_true_blind_uncertainty": score,
                    "accuracy_probe_score": score,
                    "p_true_probe_score": score,
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "pubmedqa.csv"
            self._write_csv(path, rows)
            metrics, predictions = calibration.pubmedqa_transfer(
                path, calibrators, bins=10
            )
        self.assertEqual(len(metrics), 6)
        self.assertEqual(len(predictions), 1500)


if __name__ == "__main__":
    unittest.main()
