"""Dataset adapters that normalize public QA datasets to the common schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .common_schema import make_common_example


SUPPORTED_DATASETS = ["fake", "pubmedqa", "archehr_qa"]


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


def _load_json_or_jsonl(path: str | Path) -> Any:
    input_path = Path(path)
    if input_path.suffix.lower() == ".xml":
        return _load_archehr_xml_records(input_path)
    if input_path.suffix.lower() == ".jsonl":
        records = []
        with input_path.open("r", encoding="utf-8") as infile:
            for line_number, line in enumerate(infile, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {input_path}:{line_number}") from exc
        return records
    return _load_json(input_path)


def _element_text(parent: ET.Element, tag: str) -> str | None:
    child = parent.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _load_archehr_xml_records(path: str | Path) -> list[dict[str, Any]]:
    """Load the official ArchEHR-QA XML case format without answer keys."""

    tree = ET.parse(path)
    root = tree.getroot()
    records: list[dict[str, Any]] = []
    for case in root.findall(".//case"):
        case_id = case.attrib.get("id")
        if not case_id:
            continue
        sentences = []
        for index, sentence in enumerate(case.findall(".//note_excerpt_sentences/sentence"), start=1):
            text = (sentence.text or "").strip()
            if not text:
                continue
            sentences.append(
                {
                    "sentence_id": sentence.attrib.get("id") or f"S{index}",
                    "text": text,
                }
            )
        records.append(
            {
                "id": case_id,
                "patient_question": _element_text(case, "patient_question") or "",
                "clinician_question": _element_text(case, "clinician_question"),
                "clinical_specialty": _element_text(case, "clinical_specialty"),
                "evidence": sentences,
            }
        )
    if not records:
        raise ValueError(f"No ArchEHR-QA cases found in XML file: {path}")
    return records


def _records_from_loaded_archehr(loaded: Any) -> list[dict[str, Any]]:
    if isinstance(loaded, list):
        records = loaded
    elif isinstance(loaded, dict):
        records = (
            loaded.get("examples")
            or loaded.get("data")
            or loaded.get("records")
            or loaded.get("questions")
        )
        if records is None:
            records = [loaded]
    else:
        raise ValueError("ArchEHR-QA loader expects a JSON object/list or JSONL records.")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise ValueError("ArchEHR-QA records must be JSON objects.")
    return records


def _normalize_evidence_items(record: dict[str, Any], *, example_id: str) -> tuple[list[str], list[str]]:
    evidence = (
        record.get("evidence_sentences")
        or record.get("evidence")
        or record.get("clinical_note")
        or record.get("note_sentences")
        or record.get("sentences")
        or record.get("context")
        or record.get("note")
    )
    if isinstance(evidence, str):
        evidence = [{"sentence_id": "S1", "text": evidence}]
    if not isinstance(evidence, list):
        raise ValueError(f"ArchEHR-QA example {example_id} has unsupported evidence format.")

    sentence_ids: list[str] = []
    sentence_texts: list[str] = []
    for index, item in enumerate(evidence, start=1):
        if isinstance(item, str):
            sentence_id = f"S{index}"
            text = item
        elif isinstance(item, dict):
            sentence_id = (
                item.get("sentence_id")
                or item.get("id")
                or item.get("citation")
                or item.get("sentence_index")
                or index
            )
            text = item.get("text") or item.get("sentence") or item.get("content")
        else:
            raise ValueError(f"ArchEHR-QA example {example_id} has a non-string evidence item.")
        text = str(text or "").strip()
        if not text:
            continue
        sentence_ids.append(str(sentence_id))
        sentence_texts.append(text)
    if not sentence_texts:
        raise ValueError(f"ArchEHR-QA example {example_id} has no evidence sentences.")
    return sentence_ids, sentence_texts


def _normalize_relevant_sentence_ids(record: dict[str, Any]) -> list[str] | None:
    values = (
        record.get("gold_relevant_sentence_ids")
        or record.get("relevant_sentence_ids")
        or record.get("essential_sentence_ids")
        or record.get("citations")
        or record.get("gold_citations")
    )
    if values is None:
        return None
    if isinstance(values, (str, int)):
        values = [values]
    if not isinstance(values, list):
        return None
    return [str(value) for value in values]


def archehr_records_to_common(
    records: list[dict[str, Any]], *, split: str = "dev"
) -> list[dict[str, Any]]:
    """Normalize ArchEHR-QA-style local records to the project schema."""

    common_examples = []
    for index, record in enumerate(records, start=1):
        example_id = record.get("id") or record.get("example_id") or record.get("qid") or index
        example_id = str(example_id)
        sentence_ids, sentence_texts = _normalize_evidence_items(record, example_id=example_id)
        patient_question = (
            record.get("patient_question")
            or record.get("question")
            or record.get("patientQuestion")
            or ""
        )
        clinician_question = (
            record.get("clinician_question")
            or record.get("clinician_rewrite")
            or record.get("clinicianQuestion")
            or record.get("clinical_question")
        )
        example = make_common_example(
            dataset="archehr_qa",
            example_id=example_id,
            split=str(record.get("split") or split),
            question=str(patient_question),
            clinician_question=str(clinician_question) if clinician_question else None,
            context="\n".join(sentence_texts),
            evidence_sentences=sentence_texts,
            gold_answer=record.get("gold_answer")
            or record.get("clinician_answer")
            or record.get("answer")
            or record.get("reference_answer"),
            citations=_normalize_relevant_sentence_ids(record),
        )
        example["evidence_sentence_ids"] = sentence_ids
        example["gold_relevant_sentence_ids"] = _normalize_relevant_sentence_ids(record)
        common_examples.append(example)
    return common_examples


def load_archehr_common_examples(
    data_path: str | Path,
    *,
    split: str = "dev",
    limit: int | None = None,
) -> list[dict[str, Any]]:
    loaded = _load_json_or_jsonl(data_path)
    examples = archehr_records_to_common(_records_from_loaded_archehr(loaded), split=split)
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
    if dataset_name in {"archehr_qa", "archehr", "archehrqa"}:
        if data_path is None:
            raise ValueError("--data_path is required for dataset=archehr_qa.")
        return load_archehr_common_examples(data_path, split=split, limit=limit)
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}.")
