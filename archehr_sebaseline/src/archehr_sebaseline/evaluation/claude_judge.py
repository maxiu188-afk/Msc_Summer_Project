"""Claude-based semantic answer-quality judging for BioASQ."""

from __future__ import annotations

from typing import Any


VALID_LABELS = frozenset({"correct", "incorrect"})

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
    bioasq_type: str | None = None,
) -> str:
    """Build the binary correctness judge used for the current BioASQ protocol."""

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
            "- correct: fully answers the question, agrees with the reference answer(s), and has no material error or missing required answer item.",
            "- incorrect: any material error, contradiction, missing required item, wrong extra item, or non-answer. There is no partial-credit label.",
            "- For factoid and list questions, compare answer items as a set: accept harmless ordering or wording changes, but reject missing or incorrect items.",
            "- For summary questions, accept concise paraphrases that preserve the reference answer's material claims.",
            "",
            f"BioASQ question type: {(bioasq_type or 'unknown').lower()}",
            "Ignore citation markers and wording differences. Think internally, then output exactly one token: correct or incorrect.",
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
