"""Dataset adapters that normalize public QA datasets to the common schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common_schema import make_common_example


SUPPORTED_DATASETS = ["fake", "pubmedqa"]


FAKE_COMMON_FIXTURES = [
    make_common_example(
        dataset="fake_archehr",
        example_id="fake_001",
        split="synthetic",
        question="Can the patient take the new medicine with dinner?",
        clinician_question="Is the medication compatible with evening meals?",
        evidence_sentences=[
            "The medication can be taken with or without food.",
            "The patient reported nausea when taking tablets before breakfast.",
        ],
        gold_answer="Yes. The medication can be taken with dinner because it can be taken with or without food.",
    ),
    make_common_example(
        dataset="fake_archehr",
        example_id="fake_002",
        split="synthetic",
        question="Should the patient stop the blood pressure medicine?",
        clinician_question="Is there evidence to discontinue the antihypertensive?",
        evidence_sentences=[
            "The discharge plan says to continue the blood pressure medicine daily.",
            "The patient should arrange primary care follow-up in two weeks.",
        ],
        gold_answer="No. The discharge plan says to continue the blood pressure medicine daily.",
    ),
]


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def fake_examples_to_common(
    examples: list[dict[str, Any]], *, split: str = "synthetic"
) -> list[dict[str, Any]]:
    """Normalize the project fake ArchEHR-like examples."""

    common_examples = []
    for record in examples:
        evidence_items = record.get("evidence") or []
        evidence_sentences = []
        for evidence in evidence_items:
            if isinstance(evidence, dict):
                evidence_sentences.append(str(evidence.get("text") or ""))
            else:
                evidence_sentences.append(str(evidence))

        common_examples.append(
            make_common_example(
                dataset="fake_archehr",
                example_id=str(record["id"]),
                split=split,
                question=str(record.get("patient_question") or ""),
                clinician_question=record.get("clinician_rewrite"),
                evidence_sentences=evidence_sentences,
                gold_answer=record.get("gold_answer"),
            )
        )
    return common_examples


def load_fake_common_examples(
    data_path: str | Path | None = None,
    *,
    split: str = "synthetic",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if data_path is None:
        examples = [dict(example, split=split) for example in FAKE_COMMON_FIXTURES]
    else:
        from .data_io import load_fake_examples

        examples = fake_examples_to_common(load_fake_examples(data_path), split=split)
    return examples[:limit] if limit is not None else examples


def pubmedqa_records_to_common(
    records: dict[str, dict[str, Any]], *, split: str = "pqal"
) -> list[dict[str, Any]]:
    """Normalize PubMedQA PQA-L records to the project schema."""

    common_examples = []
    for pmid in sorted(records):
        record = records[pmid]
        contexts = record.get("CONTEXTS") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = [str(context) for context in contexts if str(context).strip()]

        common_examples.append(
            make_common_example(
                dataset="pubmedqa",
                example_id=str(pmid),
                split=split,
                question=str(record.get("QUESTION") or ""),
                context="\n".join(contexts),
                evidence_sentences=contexts,
                gold_answer=record.get("LONG_ANSWER"),
                options=["yes", "no", "maybe"],
                label=record.get("final_decision"),
            )
        )
    return common_examples


def load_pubmedqa_common_examples(
    data_path: str | Path,
    *,
    split: str = "pqal",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    loaded = _load_json(data_path)
    if not isinstance(loaded, dict):
        raise ValueError("PubMedQA adapter expects the official JSON dict format.")
    examples = pubmedqa_records_to_common(loaded, split=split)
    return examples[:limit] if limit is not None else examples


def load_common_examples(
    *,
    dataset: str,
    data_path: str | Path | None = None,
    split: str = "dev",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load a supported dataset into the common schema."""

    dataset_name = dataset.lower()
    if dataset_name == "fake":
        return load_fake_common_examples(data_path, split=split, limit=limit)
    if dataset_name == "pubmedqa":
        if data_path is None:
            raise ValueError("--data_path is required for dataset=pubmedqa.")
        return load_pubmedqa_common_examples(data_path, split=split, limit=limit)
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}.")
