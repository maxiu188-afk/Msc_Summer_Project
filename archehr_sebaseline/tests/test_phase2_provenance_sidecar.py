from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "write_phase2_provenance_sidecar.py"
SPEC = importlib.util.spec_from_file_location("phase2_provenance_sidecar", SCRIPT_PATH)
assert SPEC and SPEC.loader
sidecar = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sidecar
SPEC.loader.exec_module(sidecar)


class Phase2ProvenanceSidecarTests(unittest.TestCase):
    def test_collect_artifact_hashes_covers_only_core_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            for relative_path in (
                "examples.jsonl",
                "prompts.jsonl",
                "best_generations.jsonl",
                "phase2_run_metadata.json",
                "run_timing.txt",
                "health_check.txt",
                "hidden_states/train.pt",
                "uq_baselines/p_true.csv",
            ):
                path = run_dir / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative_path, encoding="utf-8")
            post_processing = run_dir / "claude_binary_main_answer_judge" / "batch_metadata.json"
            post_processing.parent.mkdir(parents=True)
            post_processing.write_text("separate", encoding="utf-8")

            hashes = sidecar.collect_artifact_hashes(run_dir)

        self.assertEqual(set(hashes), {
            "examples.jsonl",
            "prompts.jsonl",
            "best_generations.jsonl",
            "phase2_run_metadata.json",
            "run_timing.txt",
            "health_check.txt",
            "hidden_states/train.pt",
            "uq_baselines/p_true.csv",
        })

    def test_runtime_bundle_hash_changes_with_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source.py"
            source.write_text("first", encoding="utf-8")
            first_bundle, first_files = sidecar.runtime_bundle_hash(root, ("source.py",))
            source.write_text("second", encoding="utf-8")
            second_bundle, second_files = sidecar.runtime_bundle_hash(root, ("source.py",))

        self.assertNotEqual(first_bundle, second_bundle)
        self.assertNotEqual(first_files, second_files)


if __name__ == "__main__":
    unittest.main()
