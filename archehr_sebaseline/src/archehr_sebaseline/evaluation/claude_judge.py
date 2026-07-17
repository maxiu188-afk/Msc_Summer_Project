"""Claude-based semantic answer-quality judging for BioASQ."""

from __future__ import annotations

from typing import Any


VALID_LABELS = frozenset({"good", "partial", "poor"})

CLAUDE_JUDGE_SYSTEM_PROMPT = (
    "You are performing an offline academic dataset annotation task. "
    "You are not responding to a patient, providing medical advice, making a diagnosis, "
    "or recommending treatment. Compare the supplied candidate text with the supplied "
    "reference text only, and return the requested classification."
)


def build_semantic_quality_prompt(
    *,
    question: str,
    references: list[str],
    candidate: str,
) -> str:
    """Build a three-level adaptation of the Semantic Entropy paper's judge prompt."""

    expected = "\n".join(f"- {reference}" for reference in references if reference.strip())
    if not expected:
        raise ValueError("At least one non-empty reference answer is required.")
    return "\n".join(
        [
            "We are assessing the quality of an answer to the following biomedical question.",
            "",
            f"Question: {question}",
            "",
            "Expected answer(s):",
            expected,
            "",
            f"Proposed answer: {candidate}",
            "",
            "Within the context of the question, classify the proposed answer against the expected answer(s):",
            "- good: semantically correct and adequately answers the central question, with no material error.",
            "- partial: the central answer is substantially correct but has a material omission or limitation, with no material contradiction.",
            "- poor: incorrect, contradictory, misleading, or does not substantively answer the central question.",
            "",
            "Ignore citation markers and wording differences. Think internally, then output exactly one token: good, partial, or poor.",
        ]
    )


def extract_label(message: Any) -> str:
    """Extract the single visible label from a Claude response."""

    text_blocks = [
        str(block.text).strip()
        for block in message.content
        if getattr(block, "type", None) == "text" and str(getattr(block, "text", "")).strip()
    ]
    text = " ".join(text_blocks)
    label = text.lower()
    if label not in VALID_LABELS:
        raise ValueError(f"Claude response did not contain a valid quality label: {text!r}")
    return label
