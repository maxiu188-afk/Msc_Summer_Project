from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_summary_clustering_extension import MAX_TOKENS, build_requests


class SummaryClusteringExtensionTests(unittest.TestCase):
    def test_build_requests_selects_only_added_ids(self) -> None:
        records = [
            {
                "example_id": f"q{index}",
                "incorrect": index % 2,
                "question": "What is tested?",
                "answers": [f"Answer {sample}" for sample in range(10)],
                "sample_ids": list(range(10)),
            }
            for index in range(3)
        ]
        requests, manifest = build_requests(
            records,
            added_ids={"q0", "q2"},
            model="claude-sonnet-5",
            effort="low",
        )
        self.assertEqual([row["example_id"] for row in manifest], ["q0", "q2"])
        self.assertEqual(len(requests), 2)
        for request in requests:
            params = request["params"]
            self.assertEqual(params["max_tokens"], MAX_TOKENS)
            self.assertEqual(params["output_config"]["effort"], "low")
            self.assertEqual(
                params["output_config"]["format"]["type"], "json_schema"
            )


if __name__ == "__main__":
    unittest.main()
