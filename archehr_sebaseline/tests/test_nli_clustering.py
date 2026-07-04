from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.nli_clustering import (
    ENTAILMENT,
    NEUTRAL,
    are_bidirectionally_equivalent,
    cluster_by_bidirectional_entailment,
)


class FakeEntailmentScorer:
    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        *,
        question: str | None = None,
    ) -> str:
        del question
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        if "pneumonia" in premise_lower and "pneumonia" in hypothesis_lower:
            return ENTAILMENT
        if premise == hypothesis:
            return ENTAILMENT
        return NEUTRAL


class NLIClusteringTests(unittest.TestCase):
    def test_bidirectional_equivalence_uses_scorer(self) -> None:
        self.assertTrue(
            are_bidirectionally_equivalent(
                "Antibiotics treated pneumonia.",
                "Pneumonia was treated with antibiotics.",
                FakeEntailmentScorer(),
            )
        )

    def test_cluster_by_bidirectional_entailment(self) -> None:
        generations = [
            {"example_id": "ex1", "sample_id": 0, "clean_answer": "Pneumonia was treated."},
            {"example_id": "ex1", "sample_id": 1, "clean_answer": "Treatment was for pneumonia."},
            {"example_id": "ex1", "sample_id": 2, "clean_answer": "It was a heart attack."},
        ]
        clusters = cluster_by_bidirectional_entailment(
            generations,
            scorer=FakeEntailmentScorer(),
            examples_by_id={"ex1": {"question": "Why antibiotics?"}},
        )
        self.assertEqual(clusters[0]["clustering_method"], "nli_bidirectional_entailment")
        self.assertEqual(clusters[0]["semantic_ids"], [0, 0, 1])
        self.assertEqual(clusters[0]["cluster_sizes"], [2, 1])


if __name__ == "__main__":
    unittest.main()
