from __future__ import annotations

import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_phase2_uq_efficiency.py"
SPEC = importlib.util.spec_from_file_location("phase2_uq_efficiency", SCRIPT_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


class Phase2UQEfficiencyTests(unittest.TestCase):
    def test_load_selection_uses_requested_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            examples = []
            prompts = []
            generations = []
            labels = []
            for split in ("validation", "test"):
                for question_type in ("factoid", "list", "summary"):
                    example_id = f"{split}-{question_type}"
                    examples.append(
                        {"id": example_id, "split": split, "bioasq_type": question_type}
                    )
                    prompts.append({"example_id": example_id, "prompt": "prompt"})
                    generations.append(
                        {
                            "example_id": example_id,
                            "sample_id": 0,
                            "generated_token_ids": [1],
                        }
                    )
                    labels.append(
                        {
                            "example_id": example_id,
                            "label": "incorrect",
                            "label_valid": "true",
                        }
                    )

            for name, rows in (
                ("examples.jsonl", examples),
                ("prompts.jsonl", prompts),
                ("best_generations.jsonl", generations),
            ):
                (run_dir / name).write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
            labels_path = run_dir / "labels.csv"
            with labels_path.open("w", encoding="utf-8", newline="") as outfile:
                writer = csv.DictWriter(
                    outfile, fieldnames=["example_id", "label", "label_valid"]
                )
                writer.writeheader()
                writer.writerows(labels)

            args = SimpleNamespace(
                run_dir=run_dir,
                labels_path=labels_path,
                split="validation",
                max_examples=None,
                expected_examples=3,
            )
            selected = benchmark.load_selection(args)
            self.assertEqual(
                {row["example_id"] for row in selected},
                {"validation-factoid", "validation-list", "validation-summary"},
            )

    def test_frozen_probe_scores_already_have_uncertainty_direction(self) -> None:
        scores = benchmark.frozen_probe_uncertainties(
            p_true_probe_score=0.8,
            accuracy_probe_score=0.9,
        )
        self.assertEqual(scores["p_true_probe_uncertainty"], 0.8)
        self.assertEqual(scores["accuracy_probe_uncertainty"], 0.9)

    def test_ranking_metrics_use_higher_probe_score_for_incorrect_answers(self) -> None:
        rows = [
            {
                "incorrect": 0,
                "p_true_probe_uncertainty": 0.1,
                "accuracy_probe_uncertainty": 0.2,
                "blind_p_true_uncertainty": 0.1,
                "normalized_nll_10_samples": 0.1,
                "discrete_semantic_entropy": 0.1,
                "num_clusters": 1,
            },
            {
                "incorrect": 1,
                "p_true_probe_uncertainty": 0.8,
                "accuracy_probe_uncertainty": 0.9,
                "blind_p_true_uncertainty": 0.8,
                "normalized_nll_10_samples": 0.8,
                "discrete_semantic_entropy": 0.8,
                "num_clusters": 2,
            },
        ]
        metrics = {row["method"]: row for row in benchmark.ranking_metrics(rows)}
        self.assertEqual(metrics["p_true_probe"]["auroc"], 1.0)
        self.assertEqual(metrics["accuracy_probe"]["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
