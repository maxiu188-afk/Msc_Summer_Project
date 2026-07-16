"""Lightweight BioASQ Task B quality evaluation for Level 4 outputs.

This module is deliberately a transparent approximation, not a replacement for
the official BioASQ evaluation service.  It uses the same *families* of
automatic metrics where practical: ROUGE-2 and ROUGE-SU4 for ideal (summary)
answers, accuracy for yes/no, exact-answer matching for factoids, and set
precision/recall/F1 for lists.  When standard BioASQ document identifiers are
available, it also compares the documents behind generated snippet citations
with those identifiers.  The local implementation has no stemming, synonym
resource, manual readability assessment, or official ranked-answer format;
those limitations are recorded in the output report.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from ..data_io import read_jsonl, write_csv
from ..nli_clustering import CONTRADICTION, ENTAILMENT, EntailmentScorer
from .uncertainty_metrics import (
    AUROC_FIELDS,
    REJECTION_CURVE_FIELDS,
    auroc,
    fail_if_exists,
    finite_float,
    fmt,
)


DEFAULT_QUALITY_THRESHOLD = 0.15
DEFAULT_RELATIVE_RISK_FRACTION = 0.25
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 31
DEFAULT_UNCERTAINTY_FIELDS = [
    "normalized_discrete_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "avg_token_logprob_uncertainty",
    "mean_sequence_nll",
    "mean_normalized_nll",
    "mean_token_entropy",
    "verbalized_confidence_uncertainty",
    "p_true_uncertainty",
]

GENERATION_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "bioasq_type",
    "sample_id",
    "quality_metric",
    "quality_mode",
    "quality_score",
    "reference_quality_score",
    "reference_coverage_score",
    "reference_completeness_score",
    "reference_conciseness_score",
    "nli_reference_label",
    "nli_reference_score",
    "nli_reference_forward_label",
    "nli_reference_reverse_label",
    "nli_reference_reference_index",
    "citation_document_precision",
    "citation_document_recall",
    "citation_document_f1",
    "cited_document_count",
    "gold_document_count",
    "grounded_quality_score",
    "answer_token_count",
    "rouge_2_f1",
    "rouge_2_precision",
    "rouge_2_recall",
    "rouge_su4_f1",
    "rouge_su4_precision",
    "rouge_su4_recall",
    "yesno_accuracy",
    "factoid_strict_accuracy",
    "factoid_answer_first_accuracy",
    "factoid_lenient_accuracy",
    "list_precision",
    "list_recall",
    "list_f1",
    "has_snippet_citation",
    "snippet_citation_count",
    "valid_snippet_citation_count",
    "invalid_snippet_citation_count",
    "snippet_citation_validity",
    "evidence_claim_count",
    "evidence_entailment_rate",
    "evidence_contradiction_rate",
    "cited_claim_count",
    "cited_claim_entailment_rate",
    "reference_count",
    "answer_text",
]

EXAMPLE_QUALITY_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "bioasq_type",
    "num_generations",
    "quality_target",
    "quality_mode",
    "quality_threshold",
    "quality_metric",
    "quality_score",
    "sample0_quality_score",
    "mean_quality_score",
    "best_quality_score",
    "min_quality_score",
    "median_quality_score",
    "quality_score_stddev",
    "quality_pass_rate",
    "is_low_quality",
    "is_bottom_quantile_quality",
    "quality_rank_within_type",
    "quality_band",
    "mean_rouge_2_f1",
    "mean_rouge_su4_f1",
    "mean_reference_quality_score",
    "mean_reference_coverage_score",
    "mean_reference_completeness_score",
    "mean_nli_reference_score",
    "mean_grounded_quality_score",
    "mean_citation_document_precision",
    "mean_citation_document_recall",
    "mean_citation_document_f1",
    "mean_evidence_entailment_rate",
    "mean_evidence_contradiction_rate",
    "mean_cited_claim_entailment_rate",
    "mean_yesno_accuracy",
    "mean_factoid_answer_first_accuracy",
    "mean_factoid_lenient_accuracy",
    "mean_list_f1",
    "snippet_citation_rate",
    "mean_snippet_citation_validity",
    "reference_count",
]

SE_AUROC_SENSITIVITY_FIELDS = [
    "score_name",
    "target_name",
    "num_examples",
    "num_positive",
    "num_negative",
    "auroc",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
]

SE_ASSOCIATION_FIELDS = [
    "score_name",
    "num_examples",
    "risk_spearman_rho",
    "bootstrap_ci_lower",
    "bootstrap_ci_upper",
    "aurc_continuous_risk",
    "mean_quality_at_80_coverage",
]

SE_EVAL_FIELDS = [
    *EXAMPLE_QUALITY_FIELDS,
    "discrete_semantic_entropy",
    "normalized_discrete_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "avg_token_logprob_uncertainty",
    "mean_sequence_nll",
    "num_clusters",
    "mean_normalized_nll",
    "mean_token_entropy",
    "verbalized_confidence_uncertainty",
    "p_true_uncertainty",
]


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_CITATION_RE = re.compile(r"\[\s*S\d+(?:\s*[,;]\s*S?\d+)*\s*\]", re.IGNORECASE)
_CITATION_ID_RE = re.compile(r"\bS(\d+)\b", re.IGNORECASE)
_YESNO_RE = re.compile(r"^\s*(?:answer\s*(?:is|:)?\s*)?(yes|no)\b", re.IGNORECASE)
_FACTOID_PREFIX_RE = re.compile(r"^(?:the\s+answer\s+is|answer\s*(?:is|:)?|it\s+is)\s+", re.IGNORECASE)


def _answer_text(record: dict[str, Any]) -> str:
    return str(record.get("clean_answer") or record.get("raw_answer") or "").strip()


def strip_snippet_citations(text: str) -> str:
    """Remove the project's ``[S1]``-style citations before lexical scoring."""

    return _CITATION_RE.sub(" ", str(text or ""))


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(strip_snippet_citations(text).lower())


def _counter_prf(predicted: Counter[Any], reference: Counter[Any]) -> tuple[float, float, float]:
    predicted_total = sum(predicted.values())
    reference_total = sum(reference.values())
    if not predicted_total and not reference_total:
        return 1.0, 1.0, 1.0
    if not predicted_total or not reference_total:
        return 0.0, 0.0, 0.0
    overlap = sum((predicted & reference).values())
    precision = overlap / predicted_total
    recall = overlap / reference_total
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def rouge_n_f1(prediction: str, reference: str, *, n: int = 2) -> float:
    """Unstemmed ROUGE-N F1 approximation with multiset n-gram matching."""

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    predicted = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
    gold = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
    return _counter_prf(predicted, gold)[2]


def rouge_n_prf(prediction: str, reference: str, *, n: int = 2) -> tuple[float, float, float]:
    """Return unstemmed ROUGE-N precision, recall, and F1 for one reference."""

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    predicted = Counter(tuple(pred_tokens[i : i + n]) for i in range(max(0, len(pred_tokens) - n + 1)))
    gold = Counter(tuple(ref_tokens[i : i + n]) for i in range(max(0, len(ref_tokens) - n + 1)))
    return _counter_prf(predicted, gold)


def _su4_units(tokens: list[str]) -> Counter[tuple[str, ...]]:
    """Return unigrams plus skip-bigrams with at most four skipped positions."""

    units: Counter[tuple[str, ...]] = Counter((token,) for token in tokens)
    for start, left in enumerate(tokens):
        for end in range(start + 1, min(len(tokens), start + 6)):
            units[(left, tokens[end])] += 1
    return units


def rouge_su4_f1(prediction: str, reference: str) -> float:
    """Unstemmed ROUGE-SU4 F1 approximation used for BioASQ ideal answers."""

    return _counter_prf(_su4_units(tokenize(prediction)), _su4_units(tokenize(reference)))[2]


def rouge_su4_prf(prediction: str, reference: str) -> tuple[float, float, float]:
    """Return unstemmed ROUGE-SU4 precision, recall, and F1 for one reference."""

    return _counter_prf(_su4_units(tokenize(prediction)), _su4_units(tokenize(reference)))


def _best_reference_score(
    prediction: str,
    references: list[str],
    scorer: Callable[[str, str], float],
) -> float | None:
    return max((scorer(prediction, reference) for reference in references), default=None)


def _best_reference_prf(
    prediction: str,
    references: list[str],
    scorer: Callable[[str, str], tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    """Choose the reference with the highest F1 and retain its full PRF tuple."""

    scored = [scorer(prediction, reference) for reference in references]
    return max(scored, key=lambda item: item[2], default=None)


def _normalize_entity(text: str) -> str:
    return " ".join(tokenize(text))


def extract_yesno(answer: str) -> str:
    match = _YESNO_RE.match(strip_snippet_citations(answer))
    return match.group(1).lower() if match else "unknown"


def _first_phrase(text: str) -> str:
    cleaned = strip_snippet_citations(text).strip()
    return re.split(r"[.\n;:]", cleaned, maxsplit=1)[0].strip()


def _normalize_factoid_candidate(text: str) -> str:
    """Normalize an answer-first factoid phrase without retaining boilerplate."""

    candidate = _FACTOID_PREFIX_RE.sub("", _first_phrase(text)).strip(" -:;.")
    return _normalize_entity(candidate)


def _contains_alias(answer: str, aliases: set[str]) -> bool:
    normalized_answer = _normalize_entity(answer)
    return any(alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_answer) for alias in aliases)


def _starts_with_alias(answer: str, aliases: set[str]) -> bool:
    candidate = _normalize_entity(_FACTOID_PREFIX_RE.sub("", strip_snippet_citations(answer)).strip())
    return any(alias and re.match(rf"^{re.escape(alias)}(?:\s|$)", candidate) for alias in aliases)


def _list_items(answer: str) -> set[str]:
    first_line = strip_snippet_citations(answer).split("\n", maxsplit=1)[0]
    pieces = re.split(r"[,;•]|\band\b", first_line, flags=re.IGNORECASE)
    return {normalized for piece in pieces if (normalized := _normalize_entity(piece))}


def _set_prf(predicted: set[str], gold: set[str]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _robust_list_items(answer: str) -> set[str]:
    """Parse answer-first lists with commas, bullets, and conjunctions."""

    first_answer = re.split(r"[.]", strip_snippet_citations(answer), maxsplit=1)[0]
    first_answer = _FACTOID_PREFIX_RE.sub("", first_answer).strip()
    pieces = re.split(r"[,;\n]|\band\b|\bor\b|(?:^|\s)[-•](?:\s|$)", first_answer, flags=re.IGNORECASE)
    return {normalized for piece in pieces if (normalized := _normalize_entity(piece))}


def extract_snippet_citation_ids(text: str) -> list[str]:
    """Extract cited ``S`` identifiers, preserving repeats for diagnostics."""

    citation_ids: list[str] = []
    for match in _CITATION_RE.finditer(str(text or "")):
        citation_ids.extend(f"S{number}" for number in _CITATION_ID_RE.findall(match.group(0)))
    return citation_ids


def _document_set(values: Any) -> set[str]:
    """Return non-empty BioASQ document identifiers as a normalized set."""

    if not isinstance(values, list):
        return set()
    return {str(value).strip() for value in values if str(value).strip()}


def citation_document_overlap(
    citation_ids: list[str],
    example: dict[str, Any],
) -> tuple[float | None, float | None, float | None, int, int]:
    """Compare cited snippet documents with BioASQ's standard document set.

    BioASQ does not provide sentence-level gold citations for this setting.
    The metric is therefore document-level evidence-selection agreement, not a
    claim-entailment judgment.
    """

    if str(example.get("prompt_evidence_mode") or "provided").lower() == "none":
        return None, None, None, 0, 0

    snippet_ids = [str(value).strip() for value in example.get("evidence_sentence_ids") or []]
    snippet_documents = [str(value).strip() for value in example.get("snippet_documents") or []]
    snippet_to_document = {
        snippet_id: snippet_documents[index]
        for index, snippet_id in enumerate(snippet_ids)
        if index < len(snippet_documents) and snippet_documents[index]
    }
    cited_documents = {snippet_to_document[citation_id] for citation_id in citation_ids if citation_id in snippet_to_document}
    gold_documents = _document_set(example.get("documents"))
    if not gold_documents:
        return None, None, None, len(cited_documents), 0
    overlap = len(cited_documents & gold_documents)
    precision = overlap / len(cited_documents) if cited_documents else 0.0
    recall = overlap / len(gold_documents)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, len(cited_documents), len(gold_documents)


def _known_snippet_ids(example: dict[str, Any]) -> set[str]:
    explicit_ids = {str(value).strip() for value in example.get("evidence_sentence_ids") or [] if str(value).strip()}
    if explicit_ids:
        return explicit_ids
    return {f"S{index}" for index, _ in enumerate(example.get("evidence_sentences") or [], start=1)}


def _evidence_items(example: dict[str, Any], *, max_evidence_sentences: int) -> list[tuple[str, str]]:
    sentence_ids = [str(value).strip() for value in example.get("evidence_sentence_ids") or []]
    items = []
    for index, sentence in enumerate(example.get("evidence_sentences") or [], start=1):
        text = str(sentence).strip()
        if text:
            sentence_id = sentence_ids[index - 1] if index - 1 < len(sentence_ids) else f"S{index}"
            items.append((sentence_id, text))
    return items[:max_evidence_sentences]


def _answer_claims(answer: str, *, max_claims: int) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", str(answer or "").strip())
    claims = [part.strip() for part in parts if tokenize(part)]
    return claims[:max_claims]


def _score_reference_nli(
    answer: str,
    references: list[str],
    example: dict[str, Any],
    *,
    scorer: EntailmentScorer | None,
) -> dict[str, str | float | int | None]:
    """Assign a transparent three-class NLI score for ideal/exact-answer coverage."""

    unavailable = {
        "nli_reference_label": None,
        "nli_reference_score": None,
        "nli_reference_forward_label": None,
        "nli_reference_reverse_label": None,
        "nli_reference_reference_index": None,
    }
    if scorer is None or not references:
        return unavailable

    question = str(example.get("question") or "").strip() or None
    answer_without_citations = re.sub(r"\s+([.,;:!?])", r"\1", strip_snippet_citations(answer)).strip()
    best: dict[str, str | float | int] | None = None
    for index, reference in enumerate(references):
        forward = scorer.check_implication(answer_without_citations, reference, question=question)
        reverse = scorer.check_implication(reference, answer_without_citations, question=question)
        if forward == ENTAILMENT:
            label, score = "good", 1.0
        elif forward != CONTRADICTION and reverse == ENTAILMENT:
            label, score = "partial", 0.5
        else:
            label, score = "poor", 0.0
        candidate = {
            "nli_reference_label": label,
            "nli_reference_score": score,
            "nli_reference_forward_label": forward,
            "nli_reference_reverse_label": reverse,
            "nli_reference_reference_index": index,
        }
        if best is None or float(candidate["nli_reference_score"]) > float(best["nli_reference_score"]):
            best = candidate
    return best or unavailable


def _score_evidence_grounding(
    answer: str,
    example: dict[str, Any],
    *,
    scorer: EntailmentScorer | None,
    max_claims: int,
    max_evidence_sentences: int,
) -> dict[str, float | int | None]:
    """Score whether answer claims follow from supplied BioASQ snippets.

    Each claim is tested against every cited snippet when it has a valid
    citation; uncited claims are tested against the provided evidence set. This
    is an NLI diagnostic, not a substitute for expert clinical review.
    """

    unavailable = {
        "evidence_claim_count": None,
        "evidence_entailment_rate": None,
        "evidence_contradiction_rate": None,
        "cited_claim_count": None,
        "cited_claim_entailment_rate": None,
    }
    if scorer is None:
        return unavailable
    all_evidence = _evidence_items(example, max_evidence_sentences=10_000)
    evidence = all_evidence[:max_evidence_sentences]
    claims = _answer_claims(answer, max_claims=max_claims)
    if not evidence:
        return {**unavailable, "evidence_claim_count": len(claims), "cited_claim_count": 0}
    if not claims:
        return {
            "evidence_claim_count": 0,
            "evidence_entailment_rate": 0.0,
            "evidence_contradiction_rate": 0.0,
            "cited_claim_count": 0,
            "cited_claim_entailment_rate": None,
        }

    evidence_by_id = dict(all_evidence)
    question = str(example.get("question") or "").strip() or None
    entailed = contradicted = cited_claims = cited_entailed = 0
    for claim in claims:
        cited_ids = extract_snippet_citation_ids(claim)
        valid_cited_ids = [citation_id for citation_id in cited_ids if citation_id in evidence_by_id]
        if cited_ids:
            candidates = [(citation_id, evidence_by_id[citation_id]) for citation_id in valid_cited_ids]
            cited_claims += 1
        else:
            candidates = evidence
        labels = [
            scorer.check_implication(premise, strip_snippet_citations(claim), question=question)
            for _, premise in candidates
        ]
        if ENTAILMENT in labels:
            entailed += 1
            if cited_ids:
                cited_entailed += 1
        elif CONTRADICTION in labels:
            contradicted += 1
    return {
        "evidence_claim_count": len(claims),
        "evidence_entailment_rate": entailed / len(claims),
        "evidence_contradiction_rate": contradicted / len(claims),
        "cited_claim_count": cited_claims,
        "cited_claim_entailment_rate": cited_entailed / cited_claims if cited_claims else None,
    }


def _finalize_quality(
    row: dict[str, Any],
    *,
    answer: str,
    example: dict[str, Any],
    quality_mode: str,
    grounding_scorer: EntailmentScorer | None,
    reference_nli_scorer: EntailmentScorer | None,
    references: list[str],
    max_claims: int,
    max_evidence_sentences: int,
    citation_document_f1: float | None,
    reference_completeness: float | None = None,
    reference_conciseness: float | None = None,
) -> dict[str, Any]:
    """Attach quality axes and choose the requested composite target."""

    if quality_mode not in {"reference", "grounded", "three_axis"}:
        raise ValueError("quality_mode must be 'reference', 'grounded', or 'three_axis'.")
    coverage_score = float(row["quality_score"])
    citation_aware_score = (
        math.sqrt(coverage_score * citation_document_f1)
        if citation_document_f1 is not None
        else coverage_score
    )
    grounding = _score_evidence_grounding(
        answer,
        example,
        scorer=grounding_scorer,
        max_claims=max_claims,
        max_evidence_sentences=max_evidence_sentences,
    )
    entailment = grounding["evidence_entailment_rate"]
    contradiction = grounding["evidence_contradiction_rate"]
    grounded_score = None
    if entailment is not None and contradiction is not None:
        support = max(0.0, float(entailment) - float(contradiction))
        grounded_score = math.sqrt(citation_aware_score * support)
    if quality_mode == "grounded" and grounded_score is None:
        raise ValueError("quality_mode='grounded' requires an NLI grounding scorer and non-empty evidence.")
    reference_nli = _score_reference_nli(
        answer,
        references,
        example,
        scorer=reference_nli_scorer,
    )
    nli_reference_score = reference_nli["nli_reference_score"]
    if quality_mode == "three_axis" and nli_reference_score is None:
        raise ValueError("quality_mode='three_axis' requires an NLI reference scorer and at least one reference answer.")
    three_axis_score = (
        (coverage_score * citation_document_f1 * float(nli_reference_score)) ** (1.0 / 3.0)
        if nli_reference_score is not None and citation_document_f1 is not None
        else math.sqrt(coverage_score * float(nli_reference_score)) if nli_reference_score is not None
        else None
    )
    chosen_score = (
        grounded_score if quality_mode == "grounded"
        else three_axis_score if quality_mode == "three_axis"
        else citation_aware_score
    )
    coverage_metric = row["quality_metric"]
    citation_metric = f"citation_document_geometric_mean_{coverage_metric}"
    chosen_metric = (
        f"grounded_{citation_metric}" if quality_mode == "grounded"
        else f"three_axis_geometric_mean_{coverage_metric}" if quality_mode == "three_axis"
        else citation_metric
    )
    if citation_document_f1 is None:
        chosen_metric = (
            f"grounded_{coverage_metric}" if quality_mode == "grounded"
            else f"two_axis_reference_nli_geometric_mean_{coverage_metric}" if quality_mode == "three_axis"
            else coverage_metric
        )
    return {
        **row,
        "quality_metric": chosen_metric,
        "quality_mode": quality_mode,
        "quality_score": chosen_score,
        "reference_quality_score": coverage_score,
        "reference_coverage_score": coverage_score,
        "reference_completeness_score": reference_completeness,
        "reference_conciseness_score": reference_conciseness,
        **reference_nli,
        "grounded_quality_score": grounded_score,
        **grounding,
    }


def _quality_band(score: float, *, quality_threshold: float) -> str:
    if score >= 0.45:
        return "high"
    if score >= quality_threshold:
        return "medium"
    return "low"


def evaluate_generation_quality(
    generation: dict[str, Any],
    example: dict[str, Any],
    *,
    quality_mode: str = "reference",
    grounding_scorer: EntailmentScorer | None = None,
    reference_nli_scorer: EntailmentScorer | None = None,
    max_claims: int = 4,
    max_evidence_sentences: int = 10,
) -> dict[str, Any]:
    """Score one generated answer using the BioASQ type preserved in examples."""

    answer = _answer_text(generation)
    question_type = str(example.get("bioasq_type") or "summary").lower()
    ideal_answers = [str(item).strip() for item in example.get("ideal_answers") or [] if str(item).strip()]
    exact_answers = [str(item).strip() for item in example.get("exact_answers") or [] if str(item).strip()]
    citation_ids = extract_snippet_citation_ids(answer)
    known_snippet_ids = _known_snippet_ids(example)
    valid_citation_count = sum(citation_id in known_snippet_ids for citation_id in citation_ids)
    citation_validity = valid_citation_count / len(citation_ids) if citation_ids and known_snippet_ids else None
    citation_document_precision, citation_document_recall, citation_document_f1, cited_document_count, gold_document_count = citation_document_overlap(citation_ids, example)
    base = {
        "example_id": generation.get("example_id"),
        "dataset": generation.get("dataset") or example.get("dataset"),
        "split": generation.get("split") or example.get("split"),
        "bioasq_type": question_type,
        "sample_id": generation.get("sample_id"),
        "quality_mode": quality_mode,
        "answer_token_count": len(tokenize(answer)),
        "reference_quality_score": None,
        "reference_coverage_score": None,
        "reference_completeness_score": None,
        "reference_conciseness_score": None,
        "nli_reference_label": None,
        "nli_reference_score": None,
        "nli_reference_forward_label": None,
        "nli_reference_reverse_label": None,
        "nli_reference_reference_index": None,
        "citation_document_precision": citation_document_precision,
        "citation_document_recall": citation_document_recall,
        "citation_document_f1": citation_document_f1,
        "cited_document_count": cited_document_count,
        "gold_document_count": gold_document_count,
        "grounded_quality_score": None,
        "rouge_2_f1": None,
        "rouge_2_precision": None,
        "rouge_2_recall": None,
        "rouge_su4_f1": None,
        "rouge_su4_precision": None,
        "rouge_su4_recall": None,
        "yesno_accuracy": None,
        "factoid_strict_accuracy": None,
        "factoid_answer_first_accuracy": None,
        "factoid_lenient_accuracy": None,
        "list_precision": None,
        "list_recall": None,
        "list_f1": None,
        "has_snippet_citation": bool(citation_ids),
        "snippet_citation_count": len(citation_ids),
        "valid_snippet_citation_count": valid_citation_count if known_snippet_ids else None,
        "invalid_snippet_citation_count": len(citation_ids) - valid_citation_count if known_snippet_ids else None,
        "snippet_citation_validity": citation_validity,
        "evidence_claim_count": None,
        "evidence_entailment_rate": None,
        "evidence_contradiction_rate": None,
        "cited_claim_count": None,
        "cited_claim_entailment_rate": None,
        "reference_count": len(ideal_answers) if question_type == "summary" else len(exact_answers),
        "answer_text": answer,
    }
    if question_type == "summary":
        rouge2 = _best_reference_prf(answer, ideal_answers, rouge_n_prf)
        su4 = _best_reference_prf(answer, ideal_answers, rouge_su4_prf)
        quality = (rouge2[2] + su4[2]) / 2 if rouge2 is not None and su4 is not None else 0.0
        return _finalize_quality({
            **base,
            "quality_metric": "mean_rouge2_su4_f1",
            "quality_score": quality,
            "rouge_2_precision": rouge2[0] if rouge2 else None,
            "rouge_2_recall": rouge2[1] if rouge2 else None,
            "rouge_2_f1": rouge2[2] if rouge2 else None,
            "rouge_su4_precision": su4[0] if su4 else None,
            "rouge_su4_recall": su4[1] if su4 else None,
            "rouge_su4_f1": su4[2] if su4 else None,
        }, answer=answer, example=example, quality_mode=quality_mode, grounding_scorer=grounding_scorer, reference_nli_scorer=reference_nli_scorer, references=ideal_answers, citation_document_f1=citation_document_f1,
            max_claims=max_claims, max_evidence_sentences=max_evidence_sentences,
            reference_completeness=((rouge2[1] + su4[1]) / 2 if rouge2 and su4 else 0.0),
            reference_conciseness=((rouge2[0] + su4[0]) / 2 if rouge2 and su4 else 0.0))
    if question_type == "yesno":
        gold = _normalize_entity(exact_answers[0]) if exact_answers else ""
        accuracy = float(extract_yesno(answer) == gold) if gold else 0.0
        return _finalize_quality({**base, "quality_metric": "yesno_accuracy", "quality_score": accuracy, "yesno_accuracy": accuracy}, answer=answer, example=example, quality_mode=quality_mode, grounding_scorer=grounding_scorer, reference_nli_scorer=reference_nli_scorer, references=exact_answers, citation_document_f1=citation_document_f1, max_claims=max_claims, max_evidence_sentences=max_evidence_sentences, reference_completeness=accuracy, reference_conciseness=accuracy)
    if question_type == "factoid":
        aliases = {_normalize_entity(item) for item in exact_answers if _normalize_entity(item)}
        strict = float(_normalize_factoid_candidate(answer) in aliases) if aliases else 0.0
        answer_first = float(_starts_with_alias(answer, aliases)) if aliases else 0.0
        lenient = float(_contains_alias(answer, aliases)) if aliases else 0.0
        return _finalize_quality({
            **base,
            "quality_metric": "factoid_answer_first_accuracy",
            "quality_score": answer_first,
            "factoid_strict_accuracy": strict,
            "factoid_answer_first_accuracy": answer_first,
            "factoid_lenient_accuracy": lenient,
        }, answer=answer, example=example, quality_mode=quality_mode, grounding_scorer=grounding_scorer, reference_nli_scorer=reference_nli_scorer, references=exact_answers, citation_document_f1=citation_document_f1, max_claims=max_claims, max_evidence_sentences=max_evidence_sentences, reference_completeness=answer_first, reference_conciseness=answer_first)
    if question_type == "list":
        gold = {_normalize_entity(item) for item in exact_answers if _normalize_entity(item)}
        precision, recall, f1 = _set_prf(_robust_list_items(answer), gold)
        return _finalize_quality({**base, "quality_metric": "list_f1", "quality_score": f1, "list_precision": precision, "list_recall": recall, "list_f1": f1}, answer=answer, example=example, quality_mode=quality_mode, grounding_scorer=grounding_scorer, reference_nli_scorer=reference_nli_scorer, references=exact_answers, citation_document_f1=citation_document_f1, max_claims=max_claims, max_evidence_sentences=max_evidence_sentences, reference_completeness=recall, reference_conciseness=precision)
    return _finalize_quality({**base, "quality_metric": "unsupported", "quality_score": 0.0}, answer=answer, example=example, quality_mode=quality_mode, grounding_scorer=grounding_scorer, reference_nli_scorer=reference_nli_scorer, references=[], citation_document_f1=citation_document_f1, max_claims=max_claims, max_evidence_sentences=max_evidence_sentences)


def _mean(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [value for value in (finite_float(row.get(field)) for row in rows) if value is not None]
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _population_stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _assign_relative_quality_risk(
    rows: list[dict[str, str]],
    *,
    relative_risk_fraction: float,
) -> None:
    """Mark the lowest-quality fraction separately within each BioASQ type.

    This is a threshold-sensitivity target for ranking evaluation, not a gold
    correctness label.  All examples tied at the cutoff are included so that
    arbitrary example identifiers cannot decide a low-quality label.
    """

    if not 0.0 < relative_risk_fraction < 1.0:
        raise ValueError("relative_risk_fraction must be between 0 and 1.")
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("bioasq_type") or "unknown")].append(row)
    for group_rows in groups.values():
        ordered = sorted(group_rows, key=lambda row: finite_float(row.get("quality_score")) or 0.0)
        num_risk = max(1, math.ceil(len(ordered) * relative_risk_fraction))
        cutoff = finite_float(ordered[num_risk - 1].get("quality_score")) or 0.0
        denominator = max(len(ordered) - 1, 1)
        ranks = _average_ranks([finite_float(row.get("quality_score")) or 0.0 for row in ordered])
        for row, rank in zip(ordered, ranks):
            quality_score = finite_float(row.get("quality_score")) or 0.0
            row["quality_rank_within_type"] = fmt((rank - 1.0) / denominator)
            row["is_bottom_quantile_quality"] = fmt(quality_score <= cutoff)


def _quality_target(values: list[float], quality_target: str) -> float:
    if quality_target == "mean":
        return sum(values) / len(values)
    if quality_target == "sample0":
        return values[0]
    raise ValueError("quality_target must be 'mean' or 'sample0'.")


def build_quality_rows(
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    *,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_target: str = "mean",
    relative_risk_fraction: float = DEFAULT_RELATIVE_RISK_FRACTION,
    quality_mode: str = "reference",
    grounding_scorer: EntailmentScorer | None = None,
    reference_nli_scorer: EntailmentScorer | None = None,
    max_claims: int = 4,
    max_evidence_sentences: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    examples_by_id = {str(example["id"]): example for example in examples}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)
    generation_rows: list[dict[str, str]] = []
    example_rows: list[dict[str, str]] = []
    for example_id in sorted(grouped):
        example = examples_by_id.get(example_id, {"id": example_id})
        raw_rows = [
            evaluate_generation_quality(
                record,
                example,
                quality_mode=quality_mode,
                grounding_scorer=grounding_scorer,
                reference_nli_scorer=reference_nli_scorer,
                max_claims=max_claims,
                max_evidence_sentences=max_evidence_sentences,
            )
            for record in sorted(grouped[example_id], key=lambda item: int(item.get("sample_id", 0)))
        ]
        generation_rows.extend({field: fmt(row.get(field)) for field in GENERATION_QUALITY_FIELDS} for row in raw_rows)
        values = [float(row["quality_score"]) for row in raw_rows]
        target = _quality_target(values, quality_target)
        row = {
            "example_id": example_id,
            "dataset": example.get("dataset"),
            "split": example.get("split"),
            "bioasq_type": example.get("bioasq_type"),
            "num_generations": len(raw_rows),
            "quality_target": quality_target,
            "quality_mode": quality_mode,
            "quality_threshold": quality_threshold,
            "quality_metric": raw_rows[0]["quality_metric"],
            "quality_score": target,
            "sample0_quality_score": values[0],
            "mean_quality_score": sum(values) / len(values),
            "best_quality_score": max(values),
            "min_quality_score": min(values),
            "median_quality_score": _median(values),
            "quality_score_stddev": _population_stddev(values),
            "quality_pass_rate": sum(value >= quality_threshold for value in values) / len(values),
            "is_low_quality": target < quality_threshold,
            "is_bottom_quantile_quality": None,
            "quality_rank_within_type": None,
            "quality_band": _quality_band(target, quality_threshold=quality_threshold),
            "mean_rouge_2_f1": _mean(raw_rows, "rouge_2_f1"),
            "mean_rouge_su4_f1": _mean(raw_rows, "rouge_su4_f1"),
            "mean_reference_quality_score": _mean(raw_rows, "reference_quality_score"),
            "mean_reference_coverage_score": _mean(raw_rows, "reference_coverage_score"),
            "mean_reference_completeness_score": _mean(raw_rows, "reference_completeness_score"),
            "mean_nli_reference_score": _mean(raw_rows, "nli_reference_score"),
            "mean_grounded_quality_score": _mean(raw_rows, "grounded_quality_score"),
            "mean_citation_document_precision": _mean(raw_rows, "citation_document_precision"),
            "mean_citation_document_recall": _mean(raw_rows, "citation_document_recall"),
            "mean_citation_document_f1": _mean(raw_rows, "citation_document_f1"),
            "mean_evidence_entailment_rate": _mean(raw_rows, "evidence_entailment_rate"),
            "mean_evidence_contradiction_rate": _mean(raw_rows, "evidence_contradiction_rate"),
            "mean_cited_claim_entailment_rate": _mean(raw_rows, "cited_claim_entailment_rate"),
            "mean_yesno_accuracy": _mean(raw_rows, "yesno_accuracy"),
            "mean_factoid_answer_first_accuracy": _mean(raw_rows, "factoid_answer_first_accuracy"),
            "mean_factoid_lenient_accuracy": _mean(raw_rows, "factoid_lenient_accuracy"),
            "mean_list_f1": _mean(raw_rows, "list_f1"),
            "snippet_citation_rate": sum(bool(record["has_snippet_citation"]) for record in raw_rows) / len(raw_rows),
            "mean_snippet_citation_validity": _mean(raw_rows, "snippet_citation_validity"),
            "reference_count": raw_rows[0]["reference_count"],
        }
        example_rows.append({field: fmt(row.get(field)) for field in EXAMPLE_QUALITY_FIELDS})
    _assign_relative_quality_risk(example_rows, relative_risk_fraction=relative_risk_fraction)
    return generation_rows, example_rows


def _merge_se_rows(
    example_rows: list[dict[str, str]],
    score_rows: list[dict[str, str]],
    generation_uq_rows: list[dict[str, str]],
    self_report_rows: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    scores_by_id = {str(row.get("example_id")): row for row in score_rows}
    self_report_by_id = {str(row.get("example_id")): row for row in self_report_rows or []}
    grouped_uq: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in generation_uq_rows:
        grouped_uq[str(row.get("example_id"))].append(row)
    output = []
    for row in example_rows:
        example_id = str(row["example_id"])
        merged = {**row, **scores_by_id.get(example_id, {}), **self_report_by_id.get(example_id, {})}
        for output_field, generation_field in (
            ("avg_token_logprob_uncertainty", "mean_token_logprob"),
            ("mean_sequence_nll", "sequence_nll"),
            ("mean_normalized_nll", "normalized_nll"),
            ("mean_token_entropy", "mean_token_entropy"),
        ):
            values = [
                value
                for value in (
                    finite_float(item.get(generation_field))
                    for item in grouped_uq.get(example_id, [])
                )
                if value is not None
            ]
            if values:
                average = sum(values) / len(values)
                merged[output_field] = -average if output_field == "avg_token_logprob_uncertainty" else average
        output.append({field: fmt(merged.get(field)) for field in SE_EVAL_FIELDS})
    return output


def _build_auroc_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for score_name in DEFAULT_UNCERTAINTY_FIELDS:
        scored = [(finite_float(row.get(score_name)), str(row.get("is_low_quality")).lower() == "true") for row in rows]
        scored = [(score, risk) for score, risk in scored if score is not None]
        labels = [int(risk) for _, risk in scored]
        output.append({field: fmt({"score_name": score_name, "num_examples": len(labels), "num_positive": sum(labels), "num_negative": len(labels) - sum(labels), "auroc": auroc(labels, [score for score, _ in scored]) if labels else None}.get(field)) for field in AUROC_FIELDS})
    return output


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _average_ranks(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index, _ in ordered[start:end]:
            ranks[index] = average_rank
        start = end
    return ranks


def _spearman_rho(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left_ranks, right_ranks))
    left_scale = math.sqrt(sum((x - left_mean) ** 2 for x in left_ranks))
    right_scale = math.sqrt(sum((y - right_mean) ** 2 for y in right_ranks))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else None


def _bootstrap_ci(
    left: list[float],
    right: list[float],
    metric: Callable[[list[float], list[float]], float | None],
    *,
    num_samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if num_samples <= 0 or len(left) < 2:
        return None, None
    rng = random.Random(seed)
    values = []
    for _ in range(num_samples):
        indexes = [rng.randrange(len(left)) for _ in left]
        value = metric([left[index] for index in indexes], [right[index] for index in indexes])
        if value is not None:
            values.append(value)
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _continuous_risk_summary(scores: list[float], quality: list[float]) -> tuple[float | None, float | None]:
    """Return discrete AURC and retained mean quality at 80% coverage.

    Higher uncertainty is rejected first.  Risk is ``1 - quality`` and AURC is
    the mean risk over every non-empty retained prefix, so lower is better.
    """

    if not scores:
        return None, None
    ordered = sorted(zip(scores, quality), key=lambda item: item[0])
    risks = []
    for retained_count in range(1, len(ordered) + 1):
        retained = ordered[:retained_count]
        risks.append(sum(1.0 - item[1] for item in retained) / retained_count)
    retained_80 = ordered[:max(1, math.ceil(len(ordered) * 0.8))]
    mean_quality_80 = sum(item[1] for item in retained_80) / len(retained_80)
    return sum(risks) / len(risks), mean_quality_80


def _build_auroc_sensitivity_rows(
    rows: list[dict[str, str]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, str]]:
    output = []
    targets = (
        ("fixed_quality_threshold", "is_low_quality"),
        ("bottom_quality_quantile", "is_bottom_quantile_quality"),
    )
    for score_index, score_name in enumerate(DEFAULT_UNCERTAINTY_FIELDS):
        for target_index, (target_name, target_field) in enumerate(targets):
            scored = [
                (finite_float(row.get(score_name)), str(row.get(target_field)).lower() == "true")
                for row in rows
            ]
            scored = [(score, risk) for score, risk in scored if score is not None]
            scores = [score for score, _ in scored]
            labels = [int(risk) for _, risk in scored]
            value = auroc(labels, scores) if labels else None
            lower, upper = _bootstrap_ci(
                scores,
                [float(label) for label in labels],
                lambda sampled_scores, sampled_labels: auroc(
                    [int(label) for label in sampled_labels], sampled_scores
                ),
                num_samples=bootstrap_samples,
                seed=bootstrap_seed + score_index * 10 + target_index,
            )
            data = {
                "score_name": score_name,
                "target_name": target_name,
                "num_examples": len(labels),
                "num_positive": sum(labels),
                "num_negative": len(labels) - sum(labels),
                "auroc": value,
                "bootstrap_ci_lower": lower,
                "bootstrap_ci_upper": upper,
            }
            output.append({field: fmt(data.get(field)) for field in SE_AUROC_SENSITIVITY_FIELDS})
    return output


def _build_association_rows(
    rows: list[dict[str, str]],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[dict[str, str]]:
    output = []
    for score_index, score_name in enumerate(DEFAULT_UNCERTAINTY_FIELDS):
        pairs = [
            (finite_float(row.get(score_name)), finite_float(row.get("quality_score")))
            for row in rows
        ]
        pairs = [(score, quality) for score, quality in pairs if score is not None and quality is not None]
        scores = [score for score, _ in pairs]
        quality = [value for _, value in pairs]
        risk = [1.0 - value for value in quality]
        rho = _spearman_rho(scores, risk)
        lower, upper = _bootstrap_ci(
            scores,
            risk,
            _spearman_rho,
            num_samples=bootstrap_samples,
            seed=bootstrap_seed + 100 + score_index,
        )
        aurc, quality_80 = _continuous_risk_summary(scores, quality)
        data = {
            "score_name": score_name,
            "num_examples": len(scores),
            "risk_spearman_rho": rho,
            "bootstrap_ci_lower": lower,
            "bootstrap_ci_upper": upper,
            "aurc_continuous_risk": aurc,
            "mean_quality_at_80_coverage": quality_80,
        }
        output.append({field: fmt(data.get(field)) for field in SE_ASSOCIATION_FIELDS})
    return output


def _build_rejection_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    output = []
    for score_name in DEFAULT_UNCERTAINTY_FIELDS:
        scored = [(finite_float(row.get(score_name)), str(row.get("is_low_quality")).lower() == "true", finite_float(row.get("quality_score"))) for row in rows]
        scored = sorted((item for item in scored if item[0] is not None), key=lambda item: item[0])
        for fraction in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            retained = scored[:max(1, round(len(scored) * (1 - fraction)))] if scored else []
            quality = [value for _, _, value in retained if value is not None]
            data = {"score_name": score_name, "rejection_fraction": fraction, "coverage": len(retained) / len(scored) if scored else None, "num_retained": len(retained), "accuracy": sum(not risk for _, risk, _ in retained) / len(retained) if retained else None, "mean_quality_score": sum(quality) / len(quality) if quality else None, "threshold": retained[-1][0] if retained else None}
            output.append({field: fmt(data.get(field)) for field in REJECTION_CURVE_FIELDS})
    return output


def _summary(
    generation_rows: list[dict[str, str]],
    example_rows: list[dict[str, str]],
    auroc_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
    association_rows: list[dict[str, str]],
    *,
    relative_risk_fraction: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    quality = [value for value in (finite_float(row.get("quality_score")) for row in example_rows) if value is not None]
    type_counts = Counter(row.get("bioasq_type") or "unknown" for row in example_rows)
    return {
        "dataset": "bioasq",
        "evaluator": "lightweight_bioasq_v2",
        "official_metric_status": "local approximation; not the official BioASQ evaluation service",
        "num_examples": len(example_rows),
        "num_generations": len(generation_rows),
        "question_type_counts": dict(sorted(type_counts.items())),
        "quality_mode": example_rows[0].get("quality_mode") if example_rows else None,
        "quality_threshold": finite_float(example_rows[0].get("quality_threshold")) if example_rows else None,
        "relative_risk_fraction": relative_risk_fraction,
        "bootstrap_samples": bootstrap_samples,
        "low_quality_examples": sum(str(row.get("is_low_quality")).lower() == "true" for row in example_rows),
        "low_quality_rate": sum(str(row.get("is_low_quality")).lower() == "true" for row in example_rows) / len(example_rows) if example_rows else None,
        "mean_example_quality_score": sum(quality) / len(quality) if quality else None,
        "median_example_quality_score": _median(quality) if quality else None,
        "auroc_low_quality_by_score": {row["score_name"]: finite_float(row.get("auroc")) for row in auroc_rows},
        "auroc_sensitivity_by_target": {
            target_name: {
                row["score_name"]: finite_float(row.get("auroc"))
                for row in sensitivity_rows
                if row.get("target_name") == target_name
            }
            for target_name in ("fixed_quality_threshold", "bottom_quality_quantile")
        },
        "risk_spearman_by_score": {
            row["score_name"]: finite_float(row.get("risk_spearman_rho")) for row in association_rows
        },
    }


def _write_report(summary: dict[str, Any], paths: dict[str, str], output_path: Path, *, overwrite: bool) -> None:
    fail_if_exists(output_path, overwrite)
    lines = [
        "# Lightweight BioASQ Quality Evaluation",
        "",
        "## Scope and limits",
        "",
        "This local evaluator follows BioASQ metric families but is not the official service: it uses unstemmed lexical ROUGE-2 and ROUGE-SU4 approximations, no manual ideal-answer review, no synonym resource, and no official ranked-answer protocol.",
        "When BioASQ standard documents and snippet-document mappings are available, reference mode combines answer-reference coverage with document-level cited-evidence overlap using their geometric mean. If those gold document fields are absent, it falls back to coverage alone.",
        "The fixed `quality_threshold` and the per-type bottom-quality quantile are SE sensitivity targets, not BioASQ pass marks or gold correctness labels. Recalibrate the threshold against reviewed examples whenever the quality composition changes. Continuous quality/risk associations are reported alongside AUROC to avoid treating one threshold as the full result.",
        "Grounded mode additionally combines this citation-aware score with NLI evidence support, while retaining contradiction and cited-claim diagnostics. Document overlap and NLI grounding are diagnostics, not expert clinical review or sentence-level gold citation labels.",
        "",
        "## Summary",
        "",
    ]
    for key in ("num_examples", "num_generations", "question_type_counts", "quality_mode", "quality_threshold", "relative_risk_fraction", "bootstrap_samples", "low_quality_examples", "low_quality_rate", "mean_example_quality_score", "median_example_quality_score"):
        lines.append(f"- {key}: {summary.get(key)}")
    lines.extend(["", "## Artifacts", ""])
    for key, value in paths.items():
        lines.append(f"- {key}: `{value}`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_level4_bioasq(
    run_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD,
    quality_target: str = "mean",
    relative_risk_fraction: float = DEFAULT_RELATIVE_RISK_FRACTION,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    quality_mode: str = "reference",
    grounding_scorer: EntailmentScorer | None = None,
    reference_nli_scorer: EntailmentScorer | None = None,
    max_claims: int = 4,
    max_evidence_sentences: int = 10,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Evaluate a BioASQ Level 4 run and write quality/SE analysis artifacts."""

    run_path = Path(run_dir)
    output_path = Path(output_dir) if output_dir else run_path / "bioasq_eval"
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "generation_quality": output_path / "bioasq_quality_generations.csv",
        "example_quality": output_path / "bioasq_quality_examples.csv",
        "se_eval_examples": output_path / "bioasq_se_eval_examples.csv",
        "auroc": output_path / "bioasq_se_auroc.csv",
        "auroc_sensitivity": output_path / "bioasq_se_auroc_sensitivity.csv",
        "association": output_path / "bioasq_se_association.csv",
        "rejection_curve": output_path / "bioasq_se_rejection_curve.csv",
        "summary": output_path / "bioasq_eval_summary.json",
        "report": output_path / "bioasq_eval_report.md",
    }
    examples = read_jsonl(run_path / "examples.jsonl")
    generations = read_jsonl(run_path / "cleaned_generations.jsonl")
    with (run_path / "se_scores.csv").open("r", encoding="utf-8", newline="") as infile:
        score_rows = list(csv.DictReader(infile))
    generation_uq_path = run_path / "generation_uq.csv"
    if generation_uq_path.exists():
        with generation_uq_path.open("r", encoding="utf-8", newline="") as infile:
            generation_uq_rows = list(csv.DictReader(infile))
    else:
        generation_uq_rows = []
    self_report_path = run_path / "uq_baselines" / "self_report_examples.csv"
    if self_report_path.exists():
        with self_report_path.open("r", encoding="utf-8", newline="") as infile:
            self_report_rows = list(csv.DictReader(infile))
    else:
        self_report_rows = []
    generation_rows, example_rows = build_quality_rows(
        examples,
        generations,
        quality_threshold=quality_threshold,
        quality_target=quality_target,
        relative_risk_fraction=relative_risk_fraction,
        quality_mode=quality_mode,
        grounding_scorer=grounding_scorer,
        reference_nli_scorer=reference_nli_scorer,
        max_claims=max_claims,
        max_evidence_sentences=max_evidence_sentences,
    )
    se_rows = _merge_se_rows(example_rows, score_rows, generation_uq_rows, self_report_rows)
    auroc_rows = _build_auroc_rows(se_rows)
    sensitivity_rows = _build_auroc_sensitivity_rows(
        se_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    association_rows = _build_association_rows(
        se_rows,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    rejection_rows = _build_rejection_rows(se_rows)
    summary = _summary(
        generation_rows,
        example_rows,
        auroc_rows,
        sensitivity_rows,
        association_rows,
        relative_risk_fraction=relative_risk_fraction,
        bootstrap_samples=bootstrap_samples,
    )
    write_csv(generation_rows, paths["generation_quality"], GENERATION_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(example_rows, paths["example_quality"], EXAMPLE_QUALITY_FIELDS, overwrite=overwrite)
    write_csv(se_rows, paths["se_eval_examples"], SE_EVAL_FIELDS, overwrite=overwrite)
    write_csv(auroc_rows, paths["auroc"], AUROC_FIELDS, overwrite=overwrite)
    write_csv(sensitivity_rows, paths["auroc_sensitivity"], SE_AUROC_SENSITIVITY_FIELDS, overwrite=overwrite)
    write_csv(association_rows, paths["association"], SE_ASSOCIATION_FIELDS, overwrite=overwrite)
    write_csv(rejection_rows, paths["rejection_curve"], REJECTION_CURVE_FIELDS, overwrite=overwrite)
    fail_if_exists(paths["summary"], overwrite)
    paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    string_paths = {key: str(value) for key, value in paths.items()}
    _write_report(summary, string_paths, paths["report"], overwrite=overwrite)
    return {"summary": summary, "paths": string_paths}
