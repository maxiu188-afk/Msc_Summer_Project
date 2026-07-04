"""Prompt construction for grounded clinical QA."""

from __future__ import annotations

from typing import Any

from .common_schema import common_to_prompt_example, is_common_example


SYSTEM_INSTRUCTION = (
    "You are answering a patient question using only the provided clinical "
    "evidence. Be clear, concise, and cite supporting evidence sentence IDs."
)


def format_evidence(evidence: list[dict[str, Any]]) -> str:
    lines = []
    for sentence in evidence:
        sentence_id = sentence["sentence_id"]
        text = sentence["text"]
        lines.append(f"[{sentence_id}] {text}")
    return "\n".join(lines)


def build_prompt(example: dict[str, Any]) -> str:
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


def build_prompt_records(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prompt_records = []
    for example in examples:
        prompt_records.append(
            {
                "example_id": example["id"],
                "dataset": example.get("dataset"),
                "split": example.get("split"),
                "prompt": build_prompt(example),
                "prompt_version": "grounded_qa_common_v1",
            }
        )
    return prompt_records
