from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "analysis" / "run_uq_feature_fusion.py"
SPEC = importlib.util.spec_from_file_location("uq_feature_fusion", SCRIPT_PATH)
assert SPEC and SPEC.loader
fusion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fusion
SPEC.loader.exec_module(fusion)


class UQFeatureFusionTests(unittest.TestCase):
    def test_fold_scaling_and_oof_completeness(self) -> None:
        y = np.asarray([0, 1] * 10)
        x = np.column_stack([np.arange(20, dtype=float), np.arange(20, dtype=float) ** 2])
        result = fusion.fit_repeated_oof(
            x, y, ("a", "b"), n_splits=5, n_repeats=3, random_state=11,
            seed_label="test", model_name="test_model",
        )
        self.assertTrue(np.all(result.test_counts == 3))
        self.assertTrue(np.all(np.isfinite(result.probabilities)))
        # The recorded scaler means come from the training subsets, not a global pre-fit scaler.
        self.assertTrue(any(not np.allclose(mean, x.mean(axis=0)) for mean in result.scaler_means))
        coverage, _, _ = fusion.coverage_risk_curve(y, result.probabilities)
        self.assertEqual(len(coverage), 11)
        self.assertLessEqual(float(coverage.max()), 1.0)

    def test_grouped_split_prevents_question_leakage(self) -> None:
        groups = np.repeat(np.asarray([f"q{index}" for index in range(12)]), 2)
        y = np.tile(np.asarray([0, 1]), 12)
        for _, _, train, test in fusion.make_grouped_splits(y, groups, n_splits=3, n_repeats=2, random_state=7):
            self.assertFalse(set(groups[train]) & set(groups[test]))

    def test_bootstrap_skips_single_label_resamples(self) -> None:
        y = np.asarray([0, 1])
        result = fusion.bootstrap_auc_difference(
            y, np.asarray([0.2, 0.8]), np.asarray([0.3, 0.7]), samples=100, random_state=3,
        )
        self.assertGreater(result["skipped_single_label_resamples"], 0)
        self.assertGreater(result["valid_resamples"], 0)


if __name__ == "__main__":
    unittest.main()
