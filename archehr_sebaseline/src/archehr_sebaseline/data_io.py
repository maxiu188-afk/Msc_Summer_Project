"""File IO and lightweight data normalization utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def project_root() -> Path:
    """Return the root directory of the standalone baseline project."""

    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    return project_root() / "data"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def _fail_if_exists(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Pass --overwrite if replacing it is intended."
        )


def write_jsonl(
    records: Iterable[dict[str, Any]],
    path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fail_if_exists(output_path, overwrite)
    with output_path.open("w", encoding="utf-8", newline="\n") as outfile:
        for record in records:
            outfile.write(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False))
            outfile.write("\n")


def write_csv(
    rows: Iterable[dict[str, Any]],
    path: str | Path,
    fieldnames: list[str],
    *,
    overwrite: bool = False,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _fail_if_exists(output_path, overwrite)
    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_fake_examples(path: str | Path) -> list[dict[str, Any]]:
    examples_path = Path(path)
    return read_jsonl(examples_path)


def normalize_archehr_example(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize one ArchEHR-QA-like record into the project schema.

    This supports future real-data loading without binding the maintained
    pipelines to any
    particular local dataset layout. The function expects the caller to supply
    already-loaded records.
    """

    example_id = record.get("id") or record.get("example_id") or record.get("qid")
    if example_id is None:
        raise ValueError("ArchEHR-QA record is missing an id-like field.")

    evidence = record.get("evidence") or record.get("sentences") or record.get("context")
    if isinstance(evidence, str):
        evidence = [{"sentence_id": "S1", "text": evidence}]
    if not isinstance(evidence, list):
        raise ValueError(f"Example {example_id} has unsupported evidence format.")

    normalized_evidence = []
    for idx, sentence in enumerate(evidence, start=1):
        if isinstance(sentence, str):
            normalized_evidence.append({"sentence_id": f"S{idx}", "text": sentence})
            continue
        if not isinstance(sentence, dict):
            raise ValueError(f"Example {example_id} has a non-string evidence item.")
        sentence_id = sentence.get("sentence_id") or sentence.get("id") or f"S{idx}"
        text = sentence.get("text") or sentence.get("sentence")
        if not text:
            raise ValueError(f"Example {example_id} has an evidence item without text.")
        normalized_evidence.append({"sentence_id": str(sentence_id), "text": str(text)})

    return {
        "id": str(example_id),
        "patient_question": str(record.get("patient_question") or record.get("question") or ""),
        "clinician_rewrite": str(
            record.get("clinician_rewrite")
            or record.get("clinician_question")
            or record.get("rewrite")
            or ""
        ),
        "evidence": normalized_evidence,
        "gold_answer": str(record.get("gold_answer") or record.get("answer") or ""),
    }


def load_archehr_examples(data_path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Load real ArchEHR-QA-like records from an explicit JSON or JSONL file.

    This function intentionally has no default path.
    """

    path = Path(data_path)
    if path.suffix.lower() == ".jsonl":
        raw_records = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as infile:
            loaded = json.load(infile)
        if isinstance(loaded, dict):
            raw_records = loaded.get("examples") or loaded.get("data") or loaded.get("records")
            if raw_records is None:
                raw_records = [loaded]
        elif isinstance(loaded, list):
            raw_records = loaded
        else:
            raise ValueError(f"Unsupported JSON root in {path}.")
    else:
        raise ValueError(f"Unsupported ArchEHR-QA file extension: {path.suffix}")

    examples = [normalize_archehr_example(record) for record in raw_records]
    return examples[:limit] if limit is not None else examples
