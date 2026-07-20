from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_probe import (
    binary_metrics,
    continuous_metrics,
    fit_even_threshold,
    fit_minimum_within_variance_threshold,
)


class Phase2ProbeTests(unittest.TestCase):
    def test_even_threshold_is_near_balanced_and_reusable(self) -> None:
        spec = fit_even_threshold(np.asarray([0.05, 0.10, 0.30, 0.50, 0.80, 0.90]))
        self.assertEqual(spec.train_low_count, 3)
        self.assertEqual(spec.train_high_count, 3)
        self.assertAlmostEqual(spec.threshold, 0.4)
        np.testing.assert_array_equal(spec.labels(np.asarray([0.2, 0.4, 0.7])), [0, 1, 1])

    def test_even_threshold_handles_a_middle_tie_with_scalar_boundary(self) -> None:
        spec = fit_even_threshold(np.asarray([0.0, 0.2, 0.2, 0.2, 0.9, 1.0]))
        self.assertEqual(spec.train_low_count + spec.train_high_count, 6)
        self.assertGreater(spec.threshold, 0.2)
        self.assertLess(spec.threshold, 0.9)

    def test_minimum_variance_threshold_finds_separated_cluster(self) -> None:
        spec = fit_minimum_within_variance_threshold(np.asarray([0.0, 0.1, 0.2, 0.8, 0.9, 1.0]))
        self.assertAlmostEqual(spec.threshold, 0.5)
        self.assertEqual((spec.train_low_count, spec.train_high_count), (3, 3))
        self.assertIsNotNone(spec.within_group_sse)

    def test_metrics_return_undefined_auc_for_one_class(self) -> None:
        metrics = binary_metrics(np.asarray([1, 1]), np.asarray([0.2, 0.8]))
        self.assertIsNone(metrics["auroc"])
        self.assertIsNone(metrics["average_precision"])
        self.assertAlmostEqual(float(metrics["brier"]), 0.34)

    def test_continuous_metrics_have_expected_perfect_values(self) -> None:
        metrics = continuous_metrics(np.asarray([0.1, 0.3, 0.6]), np.asarray([0.1, 0.3, 0.6]))
        self.assertAlmostEqual(float(metrics["mae"]), 0.0)
        self.assertAlmostEqual(float(metrics["rmse"]), 0.0)
        self.assertAlmostEqual(float(metrics["spearman"]), 1.0)


if __name__ == "__main__":
    unittest.main()
