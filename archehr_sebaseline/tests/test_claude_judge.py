from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.evaluation.claude_judge import (
    VALID_LABELS,
    build_semantic_quality_prompt,
)


class ClaudeJudgeTests(unittest.TestCase):
    def test_binary_prompt_requires_complete_factoid_or_list_set(self) -> None:
        prompt = build_semantic_quality_prompt(
            question="Which cytokines are requested?",
            references=["IL-6", "TNF-alpha"],
            candidate="IL-6; TNF-alpha",
            bioasq_type="list",
        )
        self.assertEqual(VALID_LABELS, frozenset({"correct", "incorrect"}))
        self.assertIn("missing required item", prompt)
        self.assertIn("BioASQ question type: list", prompt)
        self.assertIn("correct or incorrect", prompt)


if __name__ == "__main__":
    unittest.main()
