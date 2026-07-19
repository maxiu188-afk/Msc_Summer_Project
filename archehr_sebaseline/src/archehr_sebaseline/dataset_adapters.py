"""Dataset adapters that normalize public QA datasets to the common schema."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from .common_schema import make_common_example


SUPPORTED_DATASETS = [
    "fake",
    "pubmedqa",
    "bioasq",
    "bioasq_medical_uq",
    "bioasq_summary",
    "bioasq_factoid",
    "bioasq_list",
    "bioasq_yesno",
    "archehr_qa",
]


BIOASQ_MEDICAL_UQ_TYPES = ("factoid", "list", "summary")


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


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _flatten_exact_answers(value: Any) -> list[str]:
    answers: list[str] = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        text = str(item).strip()
        if text:
            answers.append(text)

    visit(value)
    return answers


def _load_bioasq_questions(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    paths = sorted(input_path.glob("*.json")) if input_path.is_dir() else [input_path]
    questions: list[dict[str, Any]] = []
    for json_path in paths:
        loaded = _load_json(json_path)
        records = loaded.get("questions") if isinstance(loaded, dict) else None
        if not isinstance(records, list):
            raise ValueError(f"BioASQ adapter expects a JSON object with questions: {json_path}")
        for record in records:
            if not isinstance(record, dict):
                raise ValueError(f"BioASQ question records must be objects: {json_path}")
            questions.append({**record, "_source_file": json_path.name})
    return questions


def bioasq_records_to_common(
    records: list[dict[str, Any]],
    *,
    split: str = "train",
    question_type: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize BioASQ Task B records to the project common schema."""

    wanted_type = question_type.lower() if question_type else None
    common_examples = []
    for index, record in enumerate(records, start=1):
        bioasq_type = str(record.get("type") or "").lower()
        if wanted_type and bioasq_type != wanted_type:
            continue

        snippets = record.get("snippets") or []
        if not isinstance(snippets, list):
            snippets = []
        evidence_sentences = []
        evidence_sentence_ids = []
        snippet_documents = []
        for snippet_index, snippet in enumerate(snippets, start=1):
            if isinstance(snippet, dict):
                text = str(snippet.get("text") or "").strip()
                document = snippet.get("document")
            else:
                text = str(snippet).strip()
                document = None
            if not text:
                continue
            evidence_sentences.append(text)
            evidence_sentence_ids.append(f"S{snippet_index}")
            snippet_documents.append(str(document) if document else "")

        ideal_answers = [
            str(answer).strip()
            for answer in _as_list(record.get("ideal_answer"))
            if str(answer).strip()
        ]
        exact_answers = _flatten_exact_answers(record.get("exact_answer"))
        label = None
        options = None
        if bioasq_type == "yesno":
            normalized_exact = [answer.lower() for answer in exact_answers]
            if "yes" in normalized_exact:
                label = "yes"
            elif "no" in normalized_exact:
                label = "no"
            options = ["yes", "no"]

        example = make_common_example(
            dataset="bioasq",
            example_id=str(record.get("id") or index),
            split=split,
            question=str(record.get("body") or ""),
            context="\n".join(evidence_sentences),
            evidence_sentences=evidence_sentences,
            gold_answer=ideal_answers[0] if ideal_answers else None,
            citations=evidence_sentence_ids if evidence_sentence_ids else None,
            options=options,
            label=label,
        )
        example["bioasq_type"] = bioasq_type
        example["source_file"] = record.get("_source_file")
        example["documents"] = [str(document) for document in record.get("documents") or []]
        example["concepts"] = [str(concept) for concept in record.get("concepts") or []]
        example["ideal_answers"] = ideal_answers
        example["exact_answers"] = exact_answers
        example["evidence_sentence_ids"] = evidence_sentence_ids
        example["snippet_documents"] = snippet_documents
        common_examples.append(example)
    return common_examples


def load_bioasq_common_examples(
    data_path: str | Path,
    *,
    split: str = "train",
    limit: int | None = None,
    question_type: str | None = None,
) -> list[dict[str, Any]]:
    examples = bioasq_records_to_common(
        _load_bioasq_questions(data_path),
        split=split,
        question_type=question_type,
    )
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


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower().replace("-", "_")


def _element_text(parent: ET.Element, tag: str) -> str | None:
    wanted = tag.lower().replace("-", "_")
    for child in parent.iter():
        if child is parent:
            continue
        if _local_name(child.tag) != wanted:
            continue
        text = " ".join(child.itertext()).strip()
        return text or None
    return None


def _question_text(case: ET.Element, role: str) -> str | None:
    role = role.lower()
    for child in case.iter():
        if child is case:
            continue
        name = _local_name(child.tag)
        attrs = {str(key).lower(): str(value).lower() for key, value in child.attrib.items()}
        attr_text = " ".join(attrs.values())
        if role in name and "question" in name:
            text = " ".join(child.itertext()).strip()
            if text:
                return text
        if name == "question" and role in attr_text:
            text = " ".join(child.itertext()).strip()
            if text:
                return text
    return None


def _case_sentence_elements(case: ET.Element) -> list[ET.Element]:
    sentence_elements = []
    for child in case.iter():
        name = _local_name(child.tag)
        if name in {"sentence", "note_sentence", "excerpt_sentence"}:
            sentence_elements.append(child)
    return sentence_elements


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
        for index, sentence in enumerate(_case_sentence_elements(case), start=1):
            text = " ".join(sentence.itertext()).strip()
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
                "patient_question": _question_text(case, "patient")
                or _element_text(case, "patient_question")
                or _element_text(case, "patientQuestion")
                or _element_text(case, "consumer_question")
                or "",
                "clinician_question": _question_text(case, "clinician")
                or _element_text(case, "clinician_question")
                or _element_text(case, "clinicianQuestion")
                or _element_text(case, "clinical_question"),
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
        if not patient_question and clinician_question:
            patient_question = clinician_question
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
    bioasq_type_limits: dict[str, int] | None = None,
    selection_seed: int = 20260718,
) -> list[dict[str, Any]]:
    """Load a supported dataset into the common schema."""

    dataset_name = dataset.lower()
    if dataset_name == "fake":
        return load_fake_common_examples(data_path, split=split, limit=limit)
    if dataset_name == "pubmedqa":
        if data_path is None:
            raise ValueError("--data_path is required for dataset=pubmedqa.")
        return load_pubmedqa_common_examples(data_path, split=split, limit=limit)
    if dataset_name == "bioasq" or dataset_name.startswith("bioasq_"):
        if data_path is None:
            raise ValueError("--data_path is required for dataset=bioasq.")
        question_type = None
        if dataset_name == "bioasq_medical_uq":
            examples = load_bioasq_common_examples(data_path, split=split, limit=None)
            selected = [
                example
                for example in examples
                if str(example.get("bioasq_type") or "").lower()
                in BIOASQ_MEDICAL_UQ_TYPES
            ]
            if bioasq_type_limits is not None:
                normalized_limits: dict[str, int] = {}
                for question_type, type_limit in bioasq_type_limits.items():
                    normalized_type = str(question_type).lower()
                    if normalized_type not in BIOASQ_MEDICAL_UQ_TYPES:
                        allowed = ", ".join(BIOASQ_MEDICAL_UQ_TYPES)
                        raise ValueError(
                            f"Unsupported BioASQ medical-UQ type limit: {question_type}. "
                            f"Allowed types: {allowed}."
                        )
                    if not isinstance(type_limit, int) or isinstance(type_limit, bool) or type_limit < 0:
                        raise ValueError(
                            f"BioASQ type limit for {normalized_type} must be a non-negative integer."
                        )
                    normalized_limits[normalized_type] = type_limit
                if limit is not None and sum(normalized_limits.values()) != limit:
                    raise ValueError(
                        "When --bioasq_type_limit is used, --max_examples must equal "
                        "the sum of the type limits."
                    )

                selected_by_type: dict[str, list[dict[str, Any]]] = {
                    question_type: [] for question_type in BIOASQ_MEDICAL_UQ_TYPES
                }
                for example in selected:
                    selected_by_type[str(example["bioasq_type"]).lower()].append(example)

                quota_selected: list[dict[str, Any]] = []
                for offset, question_type in enumerate(BIOASQ_MEDICAL_UQ_TYPES):
                    requested = normalized_limits.get(question_type, 0)
                    available = selected_by_type[question_type]
                    if requested > len(available):
                        raise ValueError(
                            f"Requested {requested} {question_type} BioASQ questions, "
                            f"but only {len(available)} are available."
                        )
                    sampler = random.Random(selection_seed + offset)
                    sampled = list(available)
                    sampler.shuffle(sampled)
                    quota_selected.extend(sampled[:requested])
                return sorted(quota_selected, key=lambda example: str(example["id"]))
            return selected[:limit] if limit is not None else selected
        if dataset_name.startswith("bioasq_"):
            question_type = dataset_name.removeprefix("bioasq_")
        return load_bioasq_common_examples(
            data_path,
            split=split,
            limit=limit,
            question_type=question_type,
        )
    if dataset_name in {"archehr_qa", "archehr", "archehrqa"}:
        if data_path is None:
            raise ValueError("--data_path is required for dataset=archehr_qa.")
        return load_archehr_common_examples(data_path, split=split, limit=limit)
    supported = ", ".join(SUPPORTED_DATASETS)
    raise ValueError(f"Unsupported dataset: {dataset}. Supported datasets: {supported}.")
