"""Frozen-Probe PubMedQA transfer dataset and prompt contract."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .dataset_adapters import load_pubmedqa_common_examples


PUBMEDQA_LABELS = ("yes", "no", "maybe")
PUBMEDQA_TRANSFER_SPLIT = "test"
PUBMEDQA_DIRECT_PROMPT_VERSION = "pubmedqa_direct_explanation_v1"
PUBMEDQA_CONTEXT_PROMPT_VERSION = "pubmedqa_context_explanation_v2"


def pubmedqa_prompt_version(*, include_context: bool) -> str:
    return PUBMEDQA_CONTEXT_PROMPT_VERSION if include_context else PUBMEDQA_DIRECT_PROMPT_VERSION


def build_pubmedqa_transfer_prompt(
    example: dict[str, Any], *, include_context: bool = False
) -> str:
    """Build a direct or abstract-context decision-plus-explanation prompt."""

    question = str(example.get("question") or "").strip()
    if not question:
        raise ValueError("PubMedQA transfer example is missing its question.")
    prompt = [
        (
            "System instruction: Answer the biomedical research question using only the "
            "provided PubMed abstract context."
            if include_context
            else "System instruction: You are answering a biomedical research question directly from your biomedical knowledge."
        ),
        "",
        f"Question: {question}",
        "",
    ]
    if include_context:
        contexts = [str(value).strip() for value in example.get("evidence_sentences") or [] if str(value).strip()]
        if not contexts:
            raise ValueError("Context-conditioned PubMedQA example has no abstract context.")
        prompt.extend(
            [
                "PubMed abstract context:",
                "\n".join(f"[C{index}] {value}" for index, value in enumerate(contexts, start=1)),
                "",
            ]
        )
    prompt.extend(
        [
            "Task:",
            "Start with exactly one decision label: yes, no, or maybe.",
            "Then give a concise explanation of one to three sentences supporting that decision.",
        ]
    )
    if include_context:
        prompt.extend(
            [
                "Base the decision and explanation only on the PubMed abstract context above.",
                "Use the official PubMedQA annotation criteria for the decision label:",
                "Choose YES when the experiments and results reported in the abstract support the question's proposition in this study context, even if the conclusion is not universally true.",
                "Choose NO when the experiments and results reported in the abstract do not support the proposition.",
                "Choose MAYBE only when the abstract supports the proposition under some conditions but not others, or when a question asks about multiple interventions, observations, or groups and the answer is true for some but false for others.",
                "Do not use MAYBE to express your own uncertainty.",
                "Do not choose MAYBE merely because the study has limitations, cautious wording, a small sample size, non-significant findings, or recommends further research.",
                "Do not claim to have seen a reference answer or final decision label.",
            ]
        )
    else:
        prompt.extend(
            [
                "Do not claim to have been given an abstract, reference answer, or evidence snippets.",
                "If the available biomedical knowledge does not support yes or no, choose maybe and explain the uncertainty.",
            ]
        )
    return "\n".join(prompt)


def load_pubmedqa_transfer_examples(
    data_path: str | Path,
    ground_truth_path: str | Path,
    *,
    expected_examples: int | None = 500,
    limit: int | None = None,
    include_context: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load only the official PQA-L test IDs; never expose references in prompts."""

    with Path(ground_truth_path).open("r", encoding="utf-8") as infile:
        ground_truth = json.load(infile)
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError("PubMedQA ground truth must be a non-empty PMID-to-label mapping.")
    if expected_examples is not None and len(ground_truth) != expected_examples:
        raise ValueError(
            f"Expected {expected_examples} official PubMedQA test labels, got {len(ground_truth)}."
        )
    invalid = sorted({str(label).lower() for label in ground_truth.values()} - set(PUBMEDQA_LABELS))
    if invalid:
        raise ValueError(f"PubMedQA ground truth contains invalid labels: {invalid}.")

    all_examples = load_pubmedqa_common_examples(data_path, split=PUBMEDQA_TRANSFER_SPLIT)
    by_id = {str(example["id"]): example for example in all_examples}
    if len(by_id) != len(all_examples):
        raise ValueError("PubMedQA source contains duplicate PMIDs.")
    missing = sorted(set(ground_truth).difference(by_id))
    if missing:
        raise ValueError(f"Official PubMedQA test IDs are missing from PQA-L: {missing[:3]}.")

    examples = []
    for pmid in sorted(ground_truth):
        example = by_id[pmid]
        source_label = str(example.get("label") or "").lower()
        official_label = str(ground_truth[pmid]).lower()
        if source_label != official_label:
            raise ValueError(f"PubMedQA label mismatch for PMID {pmid}.")
        examples.append(
            {
                **example,
                "split": PUBMEDQA_TRANSFER_SPLIT,
                "label": official_label,
                "prompt_evidence_mode": "provided" if include_context else "none",
                "reference_answer_role": "evaluation_only_not_prompt_input",
            }
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("PubMedQA transfer limit must be positive.")
        examples = examples[:limit]
    prompts = [
        {
            "example_id": example["id"],
            "dataset": "pubmedqa",
            "split": PUBMEDQA_TRANSFER_SPLIT,
            "prompt": build_pubmedqa_transfer_prompt(example, include_context=include_context),
            "prompt_version": pubmedqa_prompt_version(include_context=include_context),
            "evidence_mode": "provided" if include_context else "none",
            "output_contract": "leading_yes_no_maybe_plus_1_to_3_sentence_explanation",
        }
        for example in examples
    ]
    return examples, prompts


_LEADING_LABEL = re.compile(r"^\s*(?:\[)?(yes|no|maybe)(?:\])?(?:\s|[.,:;\-])", re.IGNORECASE)


def extract_leading_pubmedqa_label(answer: str) -> str | None:
    """Read only the required leading label, avoiding labels mentioned in prose."""

    match = _LEADING_LABEL.search(str(answer or ""))
    return match.group(1).lower() if match else None


def pubmedqa_transfer_summary(
    examples: list[dict[str, Any]], *, include_context: bool = False
) -> dict[str, Any]:
    labels = Counter(str(example.get("label") or "").lower() for example in examples)
    return {
        "num_examples": len(examples),
        "label_counts": {label: labels[label] for label in PUBMEDQA_LABELS},
        "split": PUBMEDQA_TRANSFER_SPLIT,
        "prompt_version": pubmedqa_prompt_version(include_context=include_context),
        "evidence_mode": "provided" if include_context else "none",
    }
