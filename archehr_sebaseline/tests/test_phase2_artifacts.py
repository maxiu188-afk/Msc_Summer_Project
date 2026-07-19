from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.phase2_artifacts import load_manifest_examples, validate_phase2_examples


class Phase2ArtifactTests(unittest.TestCase):
    def test_manifest_assignments_drive_prompt_splits_and_no_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "training13b.json"
            manifest_path = root / "manifest.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "questions": [
                            {"id": "f", "body": "Factoid?", "type": "factoid"},
                            {"id": "l", "body": "List?", "type": "list"},
                            {"id": "s", "body": "Summary?", "type": "summary"},
                            {"id": "y", "body": "Yes/no?", "type": "yesno"},
                        ]
                    }
                ),
                encoding="utf-8",
            )
            manifest_rows = [
                {"example_id": "f", "bioasq_type": "factoid", "split": "train", "phase1_reference": True, "question_group": "f"},
                {"example_id": "l", "bioasq_type": "list", "split": "validation", "phase1_reference": False, "question_group": "l"},
                {"example_id": "s", "bioasq_type": "summary", "split": "test", "phase1_reference": False, "question_group": "s"},
            ]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
            )

            examples, prompts = load_manifest_examples(data_path, manifest_path)

        self.assertEqual([row["id"] for row in examples], ["f", "l", "s"])
        self.assertEqual([row["split"] for row in examples], ["train", "validation", "test"])
        self.assertTrue(examples[0]["phase1_reference"])
        self.assertTrue(all(row["evidence_mode"] == "none" for row in prompts))
        self.assertEqual(
            validate_phase2_examples(examples),
            {
                "train": {"factoid": 1, "list": 0, "summary": 0},
                "validation": {"factoid": 0, "list": 1, "summary": 0},
                "test": {"factoid": 0, "list": 0, "summary": 1},
            },
        )

    def test_manifest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_path = root / "training13b.json"
            manifest_path = root / "manifest.jsonl"
            data_path.write_text(
                json.dumps({"questions": [{"id": "f", "body": "Factoid?", "type": "factoid"}]}),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps({"example_id": "other", "bioasq_type": "factoid", "split": "train"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_manifest_examples(data_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
