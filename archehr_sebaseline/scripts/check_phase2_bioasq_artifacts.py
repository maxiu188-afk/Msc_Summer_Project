"""Validate required single-answer Phase-2 BioASQ artifacts after collection."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected_examples", type=int, required=True)
    parser.add_argument("--expected_hidden_dimension", type=int, default=3840)
    parser.add_argument("--write_report", type=Path, default=None)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    try:
        examples = read_jsonl(run_dir / "examples.jsonl")
        generations = read_jsonl(run_dir / "best_generations.jsonl")
        self_report = _read_csv(run_dir / "uq_baselines" / "self_report_examples.csv")
        metadata = json.loads((run_dir / "phase2_run_metadata.json").read_text(encoding="utf-8"))
        if len(examples) != args.expected_examples or len(generations) != args.expected_examples:
            raise ValueError("Example/generation count does not equal the requested Phase-2 workload.")
        if len(self_report) != args.expected_examples:
            raise ValueError("Blind P(True) output count does not equal the requested Phase-2 workload.")
        if any(int(row.get("sample_id", "-1")) != 0 for row in generations):
            raise ValueError("Phase-2 run contains a non-zero sample ID.")
        if any(row.get("num_generations") != "1" for row in self_report):
            raise ValueError("Self-report rows are not one-answer records.")
        if "p_true_blind_uncertainty" not in self_report[0]:
            raise ValueError("Blind P(True) uncertainty column is missing.")
        excluded = set(metadata.get("uq", {}).get("excluded") or [])
        if not {"semantic_entropy", "cluster_count", "sample_disagreement", "p_true_10"}.issubset(excluded):
            raise ValueError("Run metadata does not explicitly exclude high-temperature UQ.")

        import torch

        hidden_counts = {}
        for split in ("train", "validation", "test"):
            artifact = torch.load(
                run_dir / "hidden_states" / f"phase2_hidden_states_{split}.pt",
                map_location="cpu",
                weights_only=True,
            )
            vectors = artifact["vectors"]
            valid = artifact["position_valid"]
            if tuple(vectors.shape[1:]) != (4, 3, args.expected_hidden_dimension):
                raise ValueError(f"Unexpected hidden-state tensor shape for {split}: {tuple(vectors.shape)}.")
            if tuple(valid.shape) != (int(vectors.shape[0]), 3):
                raise ValueError(f"Unexpected hidden-state validity shape for {split}: {tuple(valid.shape)}.")
            hidden_counts[split] = int(vectors.shape[0])
        if sum(hidden_counts.values()) != args.expected_examples:
            raise ValueError("Hidden-state tensor rows do not equal the requested Phase-2 workload.")
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 3

    report = "\n".join(
        [
            "phase2_single_answer_health_check: PASS",
            f"examples: {len(examples)}",
            f"best_generations: {len(generations)}",
            f"blind_p_true_rows: {len(self_report)}",
            f"hidden_state_rows: {hidden_counts}",
            "hidden_state_layout: [example, 4 blocks, 3 token positions, 3840 dimensions]",
            "high_temperature_samples: not generated",
        ]
    ) + "\n"
    if args.write_report is not None:
        args.write_report.write_text(report, encoding="utf-8")
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
