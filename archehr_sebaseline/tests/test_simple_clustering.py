from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.simple_clustering import cluster_by_clean_answer_exact


class SimpleClusteringTests(unittest.TestCase):
    def test_exact_clean_answer_clusters(self) -> None:
        generations = [
            {"example_id": "ex1", "sample_id": 0, "clean_answer": "same"},
            {"example_id": "ex1", "sample_id": 1, "clean_answer": "same"},
            {"example_id": "ex1", "sample_id": 2, "clean_answer": "different"},
        ]
        clusters = cluster_by_clean_answer_exact(generations)
        self.assertEqual(clusters[0]["semantic_ids"], [0, 0, 1])
        self.assertEqual(clusters[0]["cluster_sizes"], [2, 1])


if __name__ == "__main__":
    unittest.main()
