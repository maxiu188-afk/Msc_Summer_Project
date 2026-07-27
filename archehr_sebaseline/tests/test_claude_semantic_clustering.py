from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.claude_semantic_clustering import (
    build_equivalence_prompt,
    build_structured_equivalence_prompt,
    cluster_from_pair_decisions,
    extract_pair_bitstring,
    extract_structured_pair_bitstring,
    pair_indices,
    transitivity_violation_count,
)


class ClaudeSemanticClusteringTests(unittest.TestCase):
    def test_prompt_freezes_45_pair_order_and_correctness_boundary(self) -> None:
        prompt = build_equivalence_prompt(
            question="What is tested?",
            answers=[f"Answer {index}" for index in range(10)],
        )
        self.assertEqual(len(pair_indices(10)), 45)
        self.assertIn("(0,1), (0,2)", prompt)
        self.assertIn("(8,9)", prompt)
        self.assertIn("correctness against an external reference is irrelevant", prompt)
        self.assertIn("45-character string", prompt)

    def test_extract_pair_bitstring_allows_wrapping_but_only_one_match(self) -> None:
        bits = "1" * 45
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"`{bits}`")]
        )
        self.assertEqual(extract_pair_bitstring(message), bits)
        bad = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"{bits} {bits}")]
        )
        with self.assertRaises(ValueError):
            extract_pair_bitstring(bad)

    def test_structured_prompt_preserves_rubric_and_changes_only_serialization(self) -> None:
        answers = [f"Answer {index}" for index in range(10)]
        plain = build_equivalence_prompt(question="What is tested?", answers=answers)
        structured = build_structured_equivalence_prompt(
            question="What is tested?", answers=answers
        )
        plain_prefix, _ = plain.rsplit("\n", 1)
        structured_prefix, structured_instruction = structured.rsplit("\n", 1)
        self.assertEqual(plain_prefix, structured_prefix)
        self.assertIn("exactly 45 integers", structured_instruction)

    def test_extract_structured_pair_bitstring_requires_exact_integer_array(self) -> None:
        bits = [index % 2 for index in range(45)]
        message = SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="text",
                    text='{"decisions": [' + ",".join(map(str, bits)) + "]}",
                )
            ]
        )
        self.assertEqual(
            extract_structured_pair_bitstring(message),
            "".join(map(str, bits)),
        )
        for invalid_text in (
            '{"decisions": [0, 1]}',
            '{"decisions": [' + ",".join(["0"] * 44 + ["2"]) + "]}",
            '{"decisions": [' + ",".join(["0"] * 44 + ["true"]) + "]}",
            '{"decisions": [' + ",".join(["0"] * 45) + '], "note": "extra"}',
        ):
            with self.subTest(invalid_text=invalid_text):
                invalid = SimpleNamespace(
                    content=[SimpleNamespace(type="text", text=invalid_text)]
                )
                with self.assertRaises(ValueError):
                    extract_structured_pair_bitstring(invalid)

    def test_greedy_clustering_uses_first_representative(self) -> None:
        pairs = pair_indices(4)
        positives = {(0, 1), (1, 2)}
        bits = "".join("1" if pair in positives else "0" for pair in pairs)
        cluster = cluster_from_pair_decisions(
            example_id="q1",
            answers=["a", "b", "c", "d"],
            sample_ids=[0, 1, 2, 3],
            bitstring=bits,
        )
        self.assertEqual(cluster["semantic_ids"], [0, 0, 1, 2])
        self.assertEqual(cluster["cluster_sizes"], [2, 1, 1])

    def test_transitivity_violation_counts_two_positive_edges(self) -> None:
        pairs = pair_indices(3)
        bits = "".join("1" if pair in {(0, 1), (1, 2)} else "0" for pair in pairs)
        self.assertEqual(transitivity_violation_count(bits, num_answers=3), 1)


if __name__ == "__main__":
    unittest.main()
