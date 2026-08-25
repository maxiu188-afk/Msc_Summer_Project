from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from archehr_sebaseline.generation import StaticGenerator
from archehr_sebaseline.nli_clustering import CONTRADICTION, ENTAILMENT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_pubmedqa_v2_semantic_entropy.py"
SPEC = importlib.util.spec_from_file_location("pubmedqa_v2_semantic_entropy", SCRIPT_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class ExactTextScorer:
    def check_implication(self, premise: str, hypothesis: str, *, question: str | None = None) -> str:
        del question
        return ENTAILMENT if premise == hypothesis else CONTRADICTION


def _write_source(root: Path, *, prompt_version: str = "pubmedqa_context_explanation_v2") -> None:
    run_dir = root / "source_run"
    analysis_dir = root / "source_analysis"
    run_dir.mkdir()
    analysis_dir.mkdir()
    examples = [
        {
            "id": "1",
            "question": "Does treatment help?",
            "gold_answer": "Reference one.",
            "label": "yes",
        },
        {
            "id": "2",
            "question": "Is the association absent?",
            "gold_answer": "Reference two.",
            "label": "no",
        },
    ]
    prompts = [
        {
            "example_id": row["id"],
            "dataset": "pubmedqa",
            "split": "test",
            "prompt": f"Context for {row['id']} without the hidden reference.",
            "prompt_version": prompt_version,
            "evidence_mode": "provided",
        }
        for row in examples
    ]
    (run_dir / "examples.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in examples), encoding="utf-8"
    )
    (run_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in prompts), encoding="utf-8"
    )
    (run_dir / "pubmedqa_transfer_run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_summary": {"prompt_version": prompt_version},
                "generation": {
                    "evidence_mode": "provided_pubmed_abstract_context",
                    "num_samples": 1,
                },
                "model_name": "google/gemma-3-12b-it",
            }
        ),
        encoding="utf-8",
    )
    with (analysis_dir / "transfer_predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as outfile:
        writer = csv.DictWriter(
            outfile,
            fieldnames=["example_id", "incorrect", "predicted_label", "gold_label"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"example_id": "1", "incorrect": 0, "predicted_label": "yes", "gold_label": "yes"},
                {"example_id": "2", "incorrect": 1, "predicted_label": "yes", "gold_label": "no"},
            ]
        )


def _args(root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        source_run_dir=root / "source_run",
        source_analysis_dir=root / "source_analysis",
        output_dir=root / "output",
        analysis_dir=root / "analysis",
        expected_examples=2,
        max_examples=None,
        selected_ids_path=None,
        temperature=1.0,
        model_name="google/gemma-3-12b-it",
        nli_model_name="pritamdeka/PubMedBERT-MNLI-MedNLI",
        max_new_tokens=192,
        max_input_tokens=4096,
        nli_max_input_tokens=512,
        device="cpu",
        torch_dtype="",
        nli_torch_dtype="",
        local_files_only=True,
        overwrite=False,
    )


class PubMedQAV2SemanticEntropyTests(unittest.TestCase):
    def test_run_reuses_v2_labels_and_writes_ten_samples_per_question(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(root)
            result = runner.run_experiment(
                _args(root),
                generator=StaticGenerator(["yes. Supported.", "maybe. Mixed."]),
                nli_scorer=ExactTextScorer(),
            )
            generations = runner.read_jsonl(root / "output" / "generations.jsonl")
            predictions = runner._read_csv(root / "analysis" / "predictions.csv")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["rows"]["examples"], 2)
        self.assertEqual(result["rows"]["generations"], 20)
        self.assertEqual({row["incorrect"] for row in predictions}, {"0", "1"})
        self.assertEqual(len(generations), 20)
        self.assertIn("calibration", result["excluded_work"])

    def test_rejects_non_v2_prompt_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(root, prompt_version="pubmedqa_context_explanation_v1")
            with self.assertRaisesRegex(ValueError, "context-v2"):
                runner.load_frozen_v2_source(_args(root))

    def test_manifest_selection_and_temperature_change_only_requested_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(root)
            manifest = root / "manifest.csv"
            with manifest.open("w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(outfile, fieldnames=["example_id"])
                writer.writeheader()
                writer.writerow({"example_id": "2"})
                writer.writerow({"example_id": "1"})
            args = _args(root)
            args.selected_ids_path = manifest
            args.temperature = 0.7
            result = runner.run_experiment(
                args,
                generator=StaticGenerator(["no. Unsupported.", "maybe. Mixed."]),
                nli_scorer=ExactTextScorer(),
            )
            generations = runner.read_jsonl(root / "output" / "generations.jsonl")

        self.assertEqual(result["rows"]["examples"], 2)
        self.assertEqual(result["generation"]["temperature"], 0.7)
        self.assertTrue(result["selection"]["selection_is_external_manifest"])
        self.assertEqual([row["example_id"] for row in generations[::10]], ["2", "1"])
        self.assertEqual({row["temperature"] for row in generations}, {0.7})


if __name__ == "__main__":
    unittest.main()
