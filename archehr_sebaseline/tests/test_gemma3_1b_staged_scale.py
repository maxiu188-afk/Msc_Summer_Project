from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "analysis"
    / "run_gemma3_1b_staged_scale_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("gemma3_1b_staged_scale", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_wilson_interval_contains_observed_accuracy() -> None:
    lower, upper = MODULE.wilson_interval(10, 50)

    assert lower == pytest.approx(0.112438, abs=1e-6)
    assert upper == pytest.approx(0.330372, abs=1e-6)
    assert lower < 0.2 < upper


def test_validate_alignment_accepts_exact_staged_subset() -> None:
    accepted_rows = [
        {"id": f"f{index}", "bioasq_type": "factoid"} for index in range(50)
    ]
    accepted_rows.extend(
        {"id": f"l{index}", "bioasq_type": "list"} for index in range(50)
    )
    accepted_rows.extend(
        {"id": f"s{index}", "bioasq_type": "summary"} for index in range(200)
    )
    accepted_rows.append({"id": "unused", "bioasq_type": "factoid"})
    staged_rows = accepted_rows[:-1]
    run1 = {
        "example_rows": staged_rows,
        "examples": {row["id"]: row for row in staged_rows},
    }
    accepted = {
        "example_rows": accepted_rows,
        "examples": {row["id"]: row for row in accepted_rows},
    }

    MODULE.validate_alignment(run1, accepted, accepted)


def test_validate_alignment_rejects_record_drift() -> None:
    accepted_rows = [
        {"id": f"f{index}", "bioasq_type": "factoid"} for index in range(50)
    ]
    accepted_rows.extend(
        {"id": f"l{index}", "bioasq_type": "list"} for index in range(50)
    )
    accepted_rows.extend(
        {"id": f"s{index}", "bioasq_type": "summary"} for index in range(200)
    )
    staged_rows = [dict(row) for row in accepted_rows]
    staged_rows[0]["question"] = "changed"
    run1 = {
        "example_rows": staged_rows,
        "examples": {row["id"]: row for row in staged_rows},
    }
    accepted = {
        "example_rows": accepted_rows,
        "examples": {row["id"]: row for row in accepted_rows},
    }

    with pytest.raises(ValueError, match="differ"):
        MODULE.validate_alignment(run1, accepted, accepted)
