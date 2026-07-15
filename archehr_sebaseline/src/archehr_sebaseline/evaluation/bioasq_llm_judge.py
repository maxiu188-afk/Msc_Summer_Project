"""Independent, reproducible LLM-judge validation for BioASQ summary runs."""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from ..data_io import read_jsonl, write_csv, write_jsonl
from .uncertainty_metrics import auroc, fail_if_exists, finite_float, fmt


RUBRIC_MAXIMUM = 14
DEFAULT_JUDGE_LOW_QUALITY_THRESHOLD = 0.5
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

GENERATION_FIELDS = [
    "example_id",
    "sample_id",
    "stratum",
    "deterministic_generation_quality",
    "correctness",
    "completeness",
    "evidence_support",
    "relevance",
    "judge_score",
    "is_judge_low_quality",
    "parse_ok",
    "rationale",
]

EXAMPLE_FIELDS = [
    "example_id",
    "stratum",
    "num_generations",
    "num_valid_judgments",
    "deterministic_quality_score",
    "mean_judge_score",
    "judge_low_quality_rate",
    "is_judge_low_quality",
    "normalized_discrete_semantic_entropy",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def select_stratified_examples(
    quality_rows: list[dict[str, Any]],
    *,
    sample_size: int = 30,
    seed: int = 20260715,
) -> list[dict[str, Any]]:
    """Select equal-sized fixed samples from bottom, middle, and top thirds."""

    if sample_size <= 0 or sample_size % 3:
        raise ValueError("sample_size must be positive and divisible by 3.")
    if len(quality_rows) < sample_size:
        raise ValueError("sample_size cannot exceed the available examples.")

    scored = []
    for row in quality_rows:
        score = finite_float(row.get("quality_score"))
        if score is None:
            raise ValueError(f"Missing quality_score for example {row.get('example_id')}.")
        scored.append((score, str(row["example_id"]), row))
    scored.sort(key=lambda item: (item[0], item[1]))

    n = len(scored)
    boundaries = (n // 3, (2 * n) // 3)
    groups = {
        "bottom_third": scored[: boundaries[0]],
        "middle_third": scored[boundaries[0] : boundaries[1]],
        "top_third": scored[boundaries[1] :],
    }
    per_stratum = sample_size // 3
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for stratum, candidates in groups.items():
        if len(candidates) < per_stratum:
            raise ValueError(f"Not enough examples in {stratum}.")
        for score, example_id, row in rng.sample(candidates, per_stratum):
            selected.append(
                {
                    "example_id": example_id,
                    "stratum": stratum,
                    "deterministic_quality_score": score,
                    "quality_rank_within_type": finite_float(row.get("quality_rank_within_type")),
                }
            )
    return sorted(selected, key=lambda row: (row["stratum"], row["example_id"]))


def build_judge_prompt(example: dict[str, Any], answer: str) -> str:
    references = [str(value) for value in example.get("ideal_answers", []) if str(value).strip()]
    if not references and str(example.get("gold_answer") or "").strip():
        references = [str(example["gold_answer"])]
    evidence = []
    for index, sentence in enumerate(example.get("evidence_sentences", [])[:10], start=1):
        evidence.append(f"S{index}: {str(sentence)[:500]}")
    return "\n".join(
        [
            "You are an independent evaluator of a biomedical question-answering system.",
            "Score the candidate using only the question, reference answer(s), and evidence below.",
            "Rubric: correctness 0-4; completeness 0-4; evidence_support 0-4; relevance 0-2.",
            "Use integer scores. A fully correct, complete, evidence-supported, direct answer scores 4,4,4,2.",
            "Return exactly one JSON object and no markdown:",
            '{"correctness":0,"completeness":0,"evidence_support":0,"relevance":0,"rationale":"brief reason"}',
            "",
            f"QUESTION:\n{example.get('question', '')}",
            "",
            "REFERENCE ANSWER(S):\n" + "\n".join(f"- {value}" for value in references),
            "",
            "EVIDENCE:\n" + "\n".join(evidence),
            "",
            f"CANDIDATE ANSWER:\n{answer}",
        ]
    )


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    match = _JSON_OBJECT_RE.search(raw_response.strip())
    if match is None:
        raise ValueError("Judge response does not contain a JSON object.")
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("Judge response contains invalid JSON.") from exc
    limits = {"correctness": 4, "completeness": 4, "evidence_support": 4, "relevance": 2}
    parsed: dict[str, Any] = {}
    for field, maximum in limits.items():
        value = payload.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ValueError(f"Judge field {field} must be an integer.")
        value = int(value)
        if not 0 <= value <= maximum:
            raise ValueError(f"Judge field {field} must be in [0, {maximum}].")
        parsed[field] = value
    parsed["rationale"] = str(payload.get("rationale", "")).strip()
    parsed["judge_score"] = sum(parsed[field] for field in limits) / RUBRIC_MAXIMUM
    return parsed


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(values_a: list[float], values_b: list[float]) -> float | None:
    if len(values_a) != len(values_b):
        raise ValueError("Spearman inputs must have the same length.")
    if len(values_a) < 2:
        return None
    ranks_a = _average_ranks(values_a)
    ranks_b = _average_ranks(values_b)
    mean_a = sum(ranks_a) / len(ranks_a)
    mean_b = sum(ranks_b) / len(ranks_b)
    covariance = sum((a - mean_a) * (b - mean_b) for a, b in zip(ranks_a, ranks_b))
    variance_a = sum((a - mean_a) ** 2 for a in ranks_a)
    variance_b = sum((b - mean_b) ** 2 for b in ranks_b)
    denominator = math.sqrt(variance_a * variance_b)
    return covariance / denominator if denominator else None


def run_llm_judge_validation(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    judge: Callable[[str], str],
    judge_model_name: str,
    sample_size: int = 30,
    selection_seed: int = 20260715,
    low_quality_threshold: float = DEFAULT_JUDGE_LOW_QUALITY_THRESHOLD,
    overwrite: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Judge a fixed subset and compare the results with lexical quality and SE."""

    run_path = Path(run_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "selection": output_path / "selected_examples.jsonl",
        "prompts": output_path / "judge_prompts.jsonl",
        "raw": output_path / "judge_raw_responses.jsonl",
        "generation_scores": output_path / "judge_generation_scores.csv",
        "example_scores": output_path / "judge_example_scores.csv",
        "summary": output_path / "judge_validation_summary.json",
        "report": output_path / "judge_validation_report.md",
    }
    for path in paths.values():
        fail_if_exists(path, overwrite)

    examples = {str(row["id"]): row for row in read_jsonl(run_path / "examples.jsonl")}
    generations = read_jsonl(run_path / "cleaned_generations.jsonl")
    quality_rows = _read_csv(run_path / "bioasq_eval" / "bioasq_quality_examples.csv")
    generation_quality = {
        (str(row["example_id"]), int(row["sample_id"])): finite_float(row.get("quality_score"))
        for row in _read_csv(run_path / "bioasq_eval" / "bioasq_quality_generations.csv")
    }
    se_rows = {
        str(row["example_id"]): row
        for row in _read_csv(run_path / "bioasq_eval" / "bioasq_se_eval_examples.csv")
    }
    selected = select_stratified_examples(quality_rows, sample_size=sample_size, seed=selection_seed)
    selected_by_id = {row["example_id"]: row for row in selected}
    chosen_generations = [row for row in generations if str(row["example_id"]) in selected_by_id]
    chosen_generations.sort(key=lambda row: (str(row["example_id"]), int(row["sample_id"])))

    prompt_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    total = len(chosen_generations)
    for completed, generation in enumerate(chosen_generations, start=1):
        example_id = str(generation["example_id"])
        sample_id = int(generation["sample_id"])
        prompt = build_judge_prompt(examples[example_id], str(generation.get("clean_answer") or generation.get("raw_answer") or ""))
        raw_response = judge(prompt)
        prompt_rows.append({"example_id": example_id, "sample_id": sample_id, "prompt": prompt})
        raw_record: dict[str, Any] = {"example_id": example_id, "sample_id": sample_id, "raw_response": raw_response}
        base = {
            "example_id": example_id,
            "sample_id": sample_id,
            "stratum": selected_by_id[example_id]["stratum"],
            "deterministic_generation_quality": generation_quality.get((example_id, sample_id)),
        }
        try:
            parsed = parse_judge_response(raw_response)
            raw_record.update({"parse_ok": True, "parsed": parsed})
            score_rows.append(
                {
                    **base,
                    **parsed,
                    "is_judge_low_quality": parsed["judge_score"] < low_quality_threshold,
                    "parse_ok": True,
                }
            )
        except ValueError as exc:
            raw_record.update({"parse_ok": False, "parse_error": str(exc)})
            score_rows.append(
                {
                    **base,
                    "correctness": None,
                    "completeness": None,
                    "evidence_support": None,
                    "relevance": None,
                    "judge_score": None,
                    "is_judge_low_quality": None,
                    "parse_ok": False,
                    "rationale": "",
                }
            )
        raw_rows.append(raw_record)
        if progress_callback is not None:
            progress_callback(completed, total)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        grouped[str(row["example_id"])].append(row)
    example_rows: list[dict[str, Any]] = []
    for selection in selected:
        example_id = str(selection["example_id"])
        valid = [row for row in grouped[example_id] if row["judge_score"] is not None]
        mean_judge = sum(float(row["judge_score"]) for row in valid) / len(valid) if valid else None
        low_rate = sum(bool(row["is_judge_low_quality"]) for row in valid) / len(valid) if valid else None
        example_rows.append(
            {
                "example_id": example_id,
                "stratum": selection["stratum"],
                "num_generations": len(grouped[example_id]),
                "num_valid_judgments": len(valid),
                "deterministic_quality_score": selection["deterministic_quality_score"],
                "mean_judge_score": mean_judge,
                "judge_low_quality_rate": low_rate,
                "is_judge_low_quality": mean_judge < low_quality_threshold if mean_judge is not None else None,
                "normalized_discrete_semantic_entropy": finite_float(
                    se_rows.get(example_id, {}).get("normalized_discrete_semantic_entropy")
                ),
            }
        )

    valid_examples = [row for row in example_rows if row["mean_judge_score"] is not None]
    deterministic_scores = [float(row["deterministic_quality_score"]) for row in valid_examples]
    judge_scores = [float(row["mean_judge_score"]) for row in valid_examples]
    labels = [1 if row["is_judge_low_quality"] else 0 for row in valid_examples]
    se_valid = [row for row in valid_examples if row["normalized_discrete_semantic_entropy"] is not None]
    summary = {
        "judge_model_name": judge_model_name,
        "decoding": "greedy",
        "rubric_maximum": RUBRIC_MAXIMUM,
        "judge_low_quality_threshold": low_quality_threshold,
        "selection_method": "fixed equal sample from lexical-quality bottom, middle, and top thirds",
        "selection_seed": selection_seed,
        "sample_size": sample_size,
        "num_generations": total,
        "valid_judgments": sum(bool(row["parse_ok"]) for row in score_rows),
        "parse_failures": sum(not bool(row["parse_ok"]) for row in score_rows),
        "lexical_vs_judge_spearman": spearman(deterministic_scores, judge_scores),
        "lexical_risk_vs_judge_low_quality_auroc": auroc(labels, [-value for value in deterministic_scores]),
        "semantic_entropy_vs_judge_low_quality_auroc": auroc(
            [1 if row["is_judge_low_quality"] else 0 for row in se_valid],
            [float(row["normalized_discrete_semantic_entropy"]) for row in se_valid],
        ),
        "judge_low_quality_examples": sum(labels),
        "mean_judge_score": sum(judge_scores) / len(judge_scores) if judge_scores else None,
    }

    write_jsonl(selected, paths["selection"], overwrite=overwrite)
    write_jsonl(prompt_rows, paths["prompts"], overwrite=overwrite)
    write_jsonl(raw_rows, paths["raw"], overwrite=overwrite)
    write_csv(
        ({field: fmt(row.get(field)) for field in GENERATION_FIELDS} for row in score_rows),
        paths["generation_scores"],
        GENERATION_FIELDS,
        overwrite=overwrite,
    )
    write_csv(
        ({field: fmt(row.get(field)) for field in EXAMPLE_FIELDS} for row in example_rows),
        paths["example_scores"],
        EXAMPLE_FIELDS,
        overwrite=overwrite,
    )
    paths["summary"].write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    report_lines = [
        "# BioASQ Independent LLM-Judge Validation",
        "",
        "The judge is an independent model with greedy decoding. This is a reproducible model-based validation, not clinician review.",
        "",
        "## Configuration and results",
        "",
        *[f"- {key}: {value}" for key, value in summary.items()],
        "",
        "Raw prompts and responses are retained alongside generation- and example-level scores for audit.",
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return {"summary": summary, "paths": {key: str(value) for key, value in paths.items()}}
