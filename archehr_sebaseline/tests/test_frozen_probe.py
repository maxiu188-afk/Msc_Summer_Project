from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from archehr_sebaseline.frozen_probe import (
    load_frozen_probe_bundle,
    score_frozen_probe,
    write_frozen_probe_bundle,
)


class FrozenProbeTests(unittest.TestCase):
    def test_round_trip_and_scores_match_manual_logistic(self) -> None:
        pipeline = SimpleNamespace(
            named_steps={
                "standardize": SimpleNamespace(
                    mean_=np.asarray([1.0, 2.0]), scale_=np.asarray([2.0, 4.0])
                ),
                "model": SimpleNamespace(
                    coef_=np.asarray([[0.5, -1.0]]),
                    intercept_=np.asarray([0.25]),
                    classes_=np.asarray([0, 1]),
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = root / "bundle.json"
            arrays = root / "parameters.npz"
            write_frozen_probe_bundle(
                metadata,
                arrays,
                [
                    {
                        "name": "p_true_probe",
                        "track": "p_true",
                        "analysis": "hard_threshold_even",
                        "pipeline": pipeline,
                        "transformer_block": 24,
                        "token_position": "LT",
                        "target_threshold": 0.2,
                        "positive_class": "high_uncertainty",
                    }
                ],
                source={"dataset": "fixture"},
            )
            probe = load_frozen_probe_bundle(metadata)["p_true_probe"]
            scores = score_frozen_probe(probe, np.asarray([[1.0, 2.0], [3.0, 6.0]]))
        expected = 1.0 / (1.0 + np.exp(-np.asarray([0.25, -0.25])))
        np.testing.assert_allclose(scores, expected)
        self.assertEqual(probe.transformer_block, 24)
        self.assertEqual(probe.token_position, "LT")


if __name__ == "__main__":
    unittest.main()
