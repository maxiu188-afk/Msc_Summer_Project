from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pubmedqa_v2_se_temperature_sensitivity.py"
SPEC = importlib.util.spec_from_file_location("pubmedqa_v2_se_temperature_sensitivity", SCRIPT_PATH)
assert SPEC and SPEC.loader
study = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = study
SPEC.loader.exec_module(study)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_rows() -> list[dict[str, object]]:
    rows = []
    index = 0
    for incorrect in (0, 1):
        for gold_label in ("yes", "no", "maybe"):
            for _ in range(2):
                index += 1
                rows.append(
                    {
                        "example_id": str(index),
                        "incorrect": incorrect,
                        "gold_label": gold_label,
                        "predicted_label": "yes",
                    }
                )
    return rows


def _write_condition(
    root: Path,
    *,
    temperature: float,
    manifest_rows: list[dict[str, str]],
    score_offset: float,
    include_extra: bool = False,
    manifest_path: Path | None = None,
) -> None:
    prediction_rows = []
    for index, row in enumerate(manifest_rows):
        score = (index % 3) * 0.2 + score_offset
        prediction_rows.append(
            {
                "example_id": row["example_id"],
                "incorrect": row["incorrect"],
                "discrete_semantic_entropy": score,
                "num_clusters": 1 if score == 0 else 2,
                "source_v2_gold_label": row["gold_label"],
            }
        )
    if include_extra:
        prediction_rows.append(
            {
                "example_id": "extra",
                "incorrect": 0,
                "discrete_semantic_entropy": 0.0,
                "num_clusters": 1,
                "source_v2_gold_label": "yes",
            }
        )
    _write_csv(root / "predictions.csv", prediction_rows)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "prompt_version": "pubmedqa_context_explanation_v2",
                "model_name": "google/gemma-3-12b-it",
                "generation": {
                    "temperature": temperature,
                    "num_samples": 10,
                    "seed": 31,
                    "top_p": 0.9,
                    "top_k": 50,
                    "max_new_tokens": 192,
                },
                "clustering": {
                    "model_name": "pritamdeka/PubMedBERT-MNLI-MedNLI",
                    "method": "nli_bidirectional_entailment",
                    "condition_on_question": True,
                    "strict_entailment": False,
                },
                "source_sha256": (
                    {"selected_ids_manifest": study.sha256_file(manifest_path)}
                    if manifest_path is not None
                    else {}
                ),
            }
        ),
        encoding="utf-8",
    )


class PubMedQATemperatureSensitivityTests(unittest.TestCase):
    def test_prepare_manifest_is_deterministic_and_covers_six_strata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_csv(source / "transfer_predictions.csv", _source_rows())
            args = SimpleNamespace(
                source_analysis_dir=source,
                output_manifest=root / "manifest.csv",
                output_summary=root / "manifest_summary.json",
                sample_size=6,
                selection_seed=20260806,
            )
            result = study.prepare_manifest(args)
            manifest = study._read_csv(args.output_manifest)

        self.assertEqual(result["sample_size"], 6)
        self.assertFalse(result["selection_uses_existing_se"])
        self.assertEqual(len(manifest), 6)
        self.assertEqual(len({row["stratum"] for row in manifest}), 6)
        self.assertEqual({row["selected_stratum_size"] for row in manifest}, {"1"})

    def test_analyze_writes_three_conditions_and_paired_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            _write_csv(source / "transfer_predictions.csv", _source_rows())
            prepare_args = SimpleNamespace(
                source_analysis_dir=source,
                output_manifest=root / "manifest.csv",
                output_summary=root / "manifest_summary.json",
                sample_size=6,
                selection_seed=20260806,
            )
            study.prepare_manifest(prepare_args)
            manifest = study._read_csv(prepare_args.output_manifest)
            baseline = root / "baseline"
            low = root / "low"
            high = root / "high"
            _write_condition(
                baseline,
                temperature=1.0,
                manifest_rows=manifest,
                score_offset=0.0,
                include_extra=True,
            )
            _write_condition(
                low,
                temperature=0.7,
                manifest_rows=manifest,
                score_offset=0.1,
                manifest_path=prepare_args.output_manifest,
            )
            _write_condition(
                high,
                temperature=1.3,
                manifest_rows=manifest,
                score_offset=0.2,
                manifest_path=prepare_args.output_manifest,
            )
            args = SimpleNamespace(
                manifest=prepare_args.output_manifest,
                baseline_analysis_dir=baseline,
                low_analysis_dir=low,
                high_analysis_dir=high,
                output_dir=root / "comparison",
                bootstrap_samples=100,
                bootstrap_seed=20260806,
            )
            result = study.analyze(args)
            metrics = study._read_csv(args.output_dir / "temperature_metrics.csv")
            differences = study._read_csv(args.output_dir / "paired_differences.csv")

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["num_examples"], 6)
        self.assertEqual({row["condition"] for row in metrics}, {"low", "baseline", "high"})
        self.assertEqual(len(differences), 10)
        self.assertEqual(
            {row["comparison"] for row in differences},
            {"low_minus_baseline", "high_minus_baseline"},
        )


if __name__ == "__main__":
    unittest.main()
