from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.cleaning import clean_answer


class CleaningTests(unittest.TestCase):
    def test_removes_sentence_and_numeric_citations(self) -> None:
        raw = "Treated for pneumonia. [S1][S2] Also improved. [1]  [2]"
        self.assertEqual(clean_answer(raw), "Treated for pneumonia. Also improved.")

    def test_normalizes_repeated_spaces(self) -> None:
        raw = "  Answer   with    extra spaces.  [S1] "
        self.assertEqual(clean_answer(raw), "Answer with extra spaces.")


if __name__ == "__main__":
    unittest.main()
