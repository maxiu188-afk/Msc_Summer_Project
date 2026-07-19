"""Prompt construction for grounded clinical QA."""

from __future__ import annotations

from typing import Any

from .common_schema import common_to_prompt_example, is_common_example


SYSTEM_INSTRUCTION = (
    "You are answering a patient question using only the provided clinical "
    "evidence. Be clear, concise, and cite supporting evidence sentence IDs."
)


BIOASQ_SYSTEM_INSTRUCTION = (
    "You are answering a BioASQ biomedical question using only the provided "
    "PubMed evidence snippets."
)

BIOASQ_DIRECT_SYSTEM_INSTRUCTION = (
    "You are answering a BioASQ biomedical question directly from your "
    "biomedical knowledge."
)


def bioasq_should_include_evidence(example: dict[str, Any], *, include_evidence: bool) -> bool:
    """Resolve prompt evidence mode while keeping summary QA deliberately unguided."""

    return include_evidence and str(example.get("bioasq_type") or "").lower() != "summary"


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    lines = []
    for sentence in evidence:
        sentence_id = sentence["sentence_id"]
        text = sentence["text"]
        lines.append(f"[{sentence_id}] {text}")
    return "\n".join(lines)


def build_prompt(example: dict[str, Any], *, include_evidence: bool = True) -> str:
    if str(example.get("dataset") or "").lower() == "bioasq":
        return build_bioasq_prompt(example, include_evidence=include_evidence)

    if is_common_example(example):
        example = common_to_prompt_example(example)

    patient_question = example.get("patient_question", "").strip()
    clinician_rewrite = example.get("clinician_rewrite", "").strip()
    options = example.get("options") or []
    question_for_answering = clinician_rewrite or patient_question

    prompt_parts = [
        f"System instruction: {SYSTEM_INSTRUCTION}",
        "",
        f"Patient question: {patient_question}",
    ]
    if clinician_rewrite:
        prompt_parts.append(f"Clinician rewrite: {clinician_rewrite}")

    prompt_parts.extend(
        [
            "",
            "Evidence sentences:",
            format_evidence(example["evidence"]),
            "",
            "Task:",
            f"Answer this question: {question_for_answering}",
            "Use only the evidence above.",
            "Cite every factual claim with sentence IDs like [S1].",
        ]
    )
    if options:
        prompt_parts.append(
            "If a decision label is appropriate, include one of these labels: "
            + ", ".join(str(option) for option in options)
            + "."
        )
    return "\n".join(prompt_parts)


def build_bioasq_prompt(example: dict[str, Any], *, include_evidence: bool = True) -> str:
    """Build a grounded or direct biomedical prompt for BioASQ Task B examples."""

    if is_common_example(example):
        prompt_example = common_to_prompt_example(example)
    else:
        prompt_example = example

    question = str(prompt_example.get("patient_question") or "").strip()
    bioasq_type = str(example.get("bioasq_type") or "").lower()
    type_instruction = {
        "summary": "Write one concise biomedical paragraph that directly answers the question.",
        "factoid": "Return only the answer entity or entities as semicolon-separated terms. Do not add an explanation.",
        "list": "Return only the complete requested set as semicolon-separated items. Do not add an explanation.",
        "yesno": "Start with exactly 'yes' or 'no', then add one brief supporting sentence.",
    }.get(bioasq_type, "Write a concise biomedical answer.")

    use_evidence = bioasq_should_include_evidence(example, include_evidence=include_evidence)
    prompt_parts = [
        "System instruction: "
        + (BIOASQ_SYSTEM_INSTRUCTION if use_evidence else BIOASQ_DIRECT_SYSTEM_INSTRUCTION),
        "",
        f"Question type: {bioasq_type or 'unknown'}",
        f"Question: {question}",
        "",
    ]
    if use_evidence:
        prompt_parts.extend(
            [
            "Evidence snippets:",
            format_evidence(prompt_example["evidence"]),
            "",
            ]
        )
    prompt_parts.extend(["Task:", type_instruction])
    if use_evidence:
        prompt_parts.extend(
            [
                "Use only the evidence snippets above.",
                "Cite every factual claim with snippet IDs like [S1].",
                "If the evidence is insufficient, say that the evidence is insufficient.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "Answer directly from your biomedical knowledge.",
                "Do not claim to have been given evidence or cite snippet IDs.",
                "If you are uncertain, state that uncertainty briefly.",
            ]
        )
    return "\n".join(prompt_parts)


def build_prompt_records(
    examples: list[dict[str, Any]], *, include_evidence: bool = True
) -> list[dict[str, Any]]:
    prompt_records = []
    for example in examples:
        dataset = str(example.get("dataset") or "").lower()
        prompt_version = "grounded_qa_common_v1"
        actual_include_evidence = include_evidence
        if dataset == "bioasq":
            actual_include_evidence = bioasq_should_include_evidence(
                example, include_evidence=include_evidence
            )
            question_type = str(example.get("bioasq_type") or "unknown").lower()
            prompt_version = (
                f"bioasq_{question_type}_grounded_v2"
                if actual_include_evidence
                else f"bioasq_{question_type}_direct_v2"
            )
        prompt_records.append(
            {
                "example_id": example["id"],
                "dataset": example.get("dataset"),
                "split": example.get("split"),
                "prompt": build_prompt(example, include_evidence=include_evidence),
                "prompt_version": prompt_version,
                "evidence_mode": "provided" if actual_include_evidence else "none",
            }
        )
    return prompt_records


ARCHEHR_SYSTEM_INSTRUCTION = (
    "You are a clinical assistant answering a patient question using only the "
    "provided electronic health record note sentences."
)


def _archehr_evidence_items(example: dict[str, Any]) -> list[dict[str, str]]:
    sentence_texts = example.get("evidence_sentences") or []
    sentence_ids = example.get("evidence_sentence_ids") or []
    evidence = []
    for index, text in enumerate(sentence_texts, start=1):
        sentence_id = str(sentence_ids[index - 1]) if index - 1 < len(sentence_ids) else f"S{index}"
        evidence.append({"sentence_id": sentence_id, "text": str(text)})
    if not evidence and example.get("context"):
        evidence.append({"sentence_id": "S1", "text": str(example["context"])})
    return evidence


def build_archehr_prompt(example: dict[str, Any]) -> str:
    """Build the simple structured ArchEHR-QA SE baseline prompt."""

    patient_question = str(example.get("question") or "").strip()
    clinician_question = str(example.get("clinician_question") or "").strip()
    evidence = _archehr_evidence_items(example)

    prompt_parts = [
        f"System instruction: {ARCHEHR_SYSTEM_INSTRUCTION}",
        "",
        f"Patient question: {patient_question}",
    ]
    if clinician_question:
        prompt_parts.append(f"Clinician question: {clinician_question}")
    prompt_parts.extend(
        [
            "",
            "Evidence sentences:",
            format_evidence(evidence),
            "",
            "Task:",
            f"Question to answer: {clinician_question or patient_question}",
            "Use only the evidence sentences above.",
            "If the evidence is insufficient, say that the evidence is insufficient.",
            "Do not copy the evidence sentences into the answer.",
            "Return only a valid JSON list. Each item must have exactly these keys:",
            '- "statement": one concise answer statement',
            '- "citation": the sentence ID or IDs that support the statement, such as "3" or "3, 4"',
            "Do not include unsupported statements.",
        ]
    )
    return "\n".join(prompt_parts)


def build_archehr_prompt_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_records = []
    for example in examples:
        prompt_records.append(
            {
                "example_id": example["id"],
                "dataset": example.get("dataset"),
                "split": example.get("split"),
                "prompt": build_archehr_prompt(example),
                "prompt_version": "archehr_structured_citations_v1",
            }
        )
    return prompt_records
