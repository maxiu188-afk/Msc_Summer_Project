#!/usr/bin/env python3
"""Repair the reversed frozen-Probe scores in a completed efficiency run.

Job 5773786 was produced by a benchmark version that stored ``1 - score`` for
both frozen heads even though each head's positive class already represented
uncertainty.  This repair is deterministic and uses the saved per-example
scores; it does not regenerate answers, replay the causal LM, or rerun NLI.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = PROJECT_ROOT / "scripts" / "benchmark_phase2_uq_efficiency.py"
SPEC = importlib.util.spec_from_file_location("phase2_uq_efficiency", BENCHMARK_PATH)
assert SPEC and SPEC.loader
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)

CORRECTION_ID = "frozen_probe_positive_class_direction_20260725"
PROBE_FIELDS = ("p_true_probe_uncertainty", "accuracy_probe_uncertainty")
BACKUP_DIR_NAME = "archive_probe_direction_pre_fix"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as outfile:
        for row in rows:
            outfile.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def corrected_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    corrected = []
    for row in rows:
        repaired = dict(row)
        for field in PROBE_FIELDS:
            if field not in repaired:
                raise ValueError(f"Missing required field {field!r}")
            value = float(repaired[field])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field} is outside [0, 1]: {value}")
            repaired[field] = 1.0 - value
        corrected.append(repaired)
    return corrected


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    required = {
        "benchmark_status.json",
        "benchmark_summary.json",
        "uq_scores.csv",
        "uq_scores.jsonl",
        "uq_ranking_metrics.csv",
    }
    missing = sorted(name for name in required if not (output_dir / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing repair inputs: {missing}")

    status_path = output_dir / "benchmark_status.json"
    summary_path = output_dir / "benchmark_summary.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError(f"Run is not complete: {status.get('status')!r}")
    existing_corrections = summary.get("posthoc_corrections", [])
    if any(item.get("id") == CORRECTION_ID for item in existing_corrections):
        raise ValueError(f"Correction already applied: {CORRECTION_ID}")

    _, old_metrics = read_csv(output_dir / "uq_ranking_metrics.csv")
    old_by_method = {row["method"]: float(row["auroc"]) for row in old_metrics}
    if not (
        old_by_method.get("p_true_probe", 1.0) < 0.5
        and old_by_method.get("accuracy_probe", 1.0) < 0.5
    ):
        raise ValueError(
            "The saved Probe AUROCs do not match the known reversed-direction signature."
        )

    backup_dir = output_dir / BACKUP_DIR_NAME
    if backup_dir.exists():
        raise FileExistsError(f"Backup directory already exists: {backup_dir}")
    backup_dir.mkdir()
    for name in required:
        shutil.copy2(output_dir / name, backup_dir / name)

    csv_fields, csv_rows = read_csv(output_dir / "uq_scores.csv")
    jsonl_rows = read_jsonl(output_dir / "uq_scores.jsonl")
    if len(csv_rows) != len(jsonl_rows):
        raise ValueError(
            f"CSV/JSONL row count mismatch: {len(csv_rows)} versus {len(jsonl_rows)}"
        )
    corrected_csv_rows = corrected_rows(csv_rows)
    corrected_jsonl_rows = corrected_rows(jsonl_rows)
    ranking_rows = benchmark.ranking_metrics(corrected_csv_rows)

    write_csv(output_dir / "uq_scores.csv", csv_fields, corrected_csv_rows)
    write_jsonl(output_dir / "uq_scores.jsonl", corrected_jsonl_rows)
    write_csv(
        output_dir / "uq_ranking_metrics.csv",
        ["method", "examples", "positive_class", "auroc", "average_precision"],
        ranking_rows,
    )

    correction = {
        "id": CORRECTION_ID,
        "affected_fields": list(PROBE_FIELDS),
        "backup_directory": BACKUP_DIR_NAME,
        "operation": "replaced each saved value with 1 - value",
        "reason": (
            "Both frozen heads already output the probability of their "
            "uncertainty-positive class; benchmark job 5773786 inverted them."
        ),
        "requires_gpu_rerun": False,
    }
    summary["ranking_metrics"] = ranking_rows
    summary["posthoc_corrections"] = [*existing_corrections, correction]
    status["posthoc_corrections"] = [
        *status.get("posthoc_corrections", []),
        CORRECTION_ID,
    ]
    write_json(summary_path, summary)
    write_json(status_path, status)

    print(
        f"repair=PASS rows={len(corrected_csv_rows)} "
        f"backup={backup_dir} correction={CORRECTION_ID}"
    )
    for row in ranking_rows:
        print(
            f"{row['method']} auroc={float(row['auroc']):.6f} "
            f"ap={float(row['average_precision']):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
