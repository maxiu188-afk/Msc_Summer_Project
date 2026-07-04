"""Common dataset schema for grounded QA experiments."""

from __future__ import annotations

from typing import Any


COMMON_FIELDS = [
    "dataset",
    "id",
    "split",
    "question",
    "clinician_question",
    "context",
    "evidence_sentences",
    "gold_answer",
    "citations",
    "options",
    "label",
]


def make_common_example(
    *,
    dataset: str,
    example_id: str,
    split: str,
    question: str,
    clinician_question: str | None = None,
    context: str | None = None,
    evidence_sentences: list[str] | None = None,
    gold_answer: str | None = None,
    citations: list[int] | None = None,
    options: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Create a normalized example with the project-wide schema."""

    if not dataset:
        raise ValueError("dataset must be non-empty.")
    if not example_id:
        raise ValueError("example_id must be non-empty.")
    if not question:
        raise ValueError(f"Example {example_id} is missing a question.")

    normalized_evidence = [str(item) for item in evidence_sentences or [] if str(item).strip()]
    normalized_context = context if context is not None else "\n".join(normalized_evidence)

    return {
        "dataset": str(dataset),
        "id": str(example_id),
        "split": str(split),
        "question": str(question),
        "clinician_question": str(clinician_question) if clinician_question else None,
        "context": str(normalized_context) if normalized_context else None,
        "evidence_sentences": normalized_evidence or None,
        "gold_answer": str(gold_answer) if gold_answer else None,
        "citations": [int(citation) for citation in citations] if citations else None,
        "options": [str(option) for option in options] if options else None,
        "label": str(label) if label else None,
    }


def common_to_prompt_example(example: dict[str, Any]) -> dict[str, Any]:
    """Convert a common-schema example to the prompt builder's legacy shape."""

    evidence_sentences = example.get("evidence_sentences")
    if evidence_sentences:
        evidence = [
            {"sentence_id": f"S{idx}", "text": str(sentence)}
            for idx, sentence in enumerate(evidence_sentences, start=1)
        ]
    elif example.get("context"):
        evidence = [{"sentence_id": "S1", "text": str(example["context"])}]
    else:
        evidence = []

    return {
        "id": str(example["id"]),
        "patient_question": str(example.get("question") or ""),
        "clinician_rewrite": str(example.get("clinician_question") or ""),
        "evidence": evidence,
        "gold_answer": str(example.get("gold_answer") or ""),
        "dataset": example.get("dataset"),
        "split": example.get("split"),
        "label": example.get("label"),
        "options": example.get("options"),
    }


def is_common_example(example: dict[str, Any]) -> bool:
    """Return whether a record looks like the project-wide common schema."""

    return "question" in example and ("evidence_sentences" in example or "context" in example)
