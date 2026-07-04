from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.entropy import (
    normalized_semantic_entropy,
    predictive_entropy_from_logprobs,
    semantic_entropy_from_ids_and_logprobs,
    semantic_entropy_from_cluster_sizes,
)


class EntropyTests(unittest.TestCase):
    def test_single_cluster_entropy_is_zero(self) -> None:
        self.assertAlmostEqual(semantic_entropy_from_cluster_sizes([3]), 0.0)

    def test_two_one_cluster_entropy(self) -> None:
        self.assertAlmostEqual(
            semantic_entropy_from_cluster_sizes([2, 1]),
            0.6365141683,
            places=4,
        )

    def test_two_one_normalized_entropy(self) -> None:
        self.assertAlmostEqual(
            normalized_semantic_entropy([2, 1], num_samples=3),
            0.5793801643,
            places=4,
        )

    def test_predictive_entropy_from_logprobs(self) -> None:
        self.assertAlmostEqual(predictive_entropy_from_logprobs([-0.5, -1.0]), 0.75)

    def test_likelihood_weighted_semantic_entropy_equal_probs(self) -> None:
        self.assertAlmostEqual(
            semantic_entropy_from_ids_and_logprobs([0, 0, 1], [-1.0, -1.0, -1.0]),
            0.6365141683,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
