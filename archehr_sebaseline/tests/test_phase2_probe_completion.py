from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "run_phase2_probe_completion.py"
)
SPEC = importlib.util.spec_from_file_location("phase2_probe_completion", SCRIPT_PATH)
assert SPEC and SPEC.loader
completion = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = completion
SPEC.loader.exec_module(completion)


class Phase2ProbeCompletionTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_candidate_minus_baseline_direction(self) -> None:
        labels = np.asarray([0, 0, 1, 1] * 10)
        baseline = np.asarray([0.4, 0.7, 0.6, 0.3] * 10)
        candidate = np.asarray([0.1, 0.2, 0.8, 0.9] * 10)
        result = completion.paired_auc_bootstrap(
            labels,
            baseline,
            candidate,
            samples=500,
            random_seed=7,
        )
        self.assertGreater(result["observed_difference"], 0.0)
        self.assertGreaterEqual(result["ci_lower_95"], 0.0)
        self.assertEqual(result["requested_resamples"], 500)

    def test_ranking_metrics_reports_probability_brier(self) -> None:
        result = completion.ranking_metrics(
            np.asarray([0, 1]),
            np.asarray([0.2, 0.8]),
        )
        self.assertEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["brier"], 0.04)


if __name__ == "__main__":
    unittest.main()
