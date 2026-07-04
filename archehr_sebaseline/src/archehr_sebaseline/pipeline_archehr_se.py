"""ArchEHR-QA Semantic Entropy baseline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .answer_parsing import parse_generation_records
from .baseline_uq import GENERATION_UQ_FIELDS, generation_uq_rows
from .citation_uq import CITATION_UQ_FIELDS, citation_uq_rows
from .cleaning import clean_answer
from .dataset_adapters import load_common_examples
from .data_io import project_root, write_csv, write_jsonl
from .generation import (
    GenerationConfig,
    HuggingFaceCausalLMGenerator,
    MissingGenerationDependency,
    TextGenerator,
    generate_answer_records,
)
from .nli_clustering import (
    EntailmentScorer,
    HuggingFaceNLIScorer,
    NLIConfig,
    cluster_by_bidirectional_entailment,
)
from .prompting import build_archehr_prompt_records
from .semantic_scores import SCORE_FIELDS, score_clusters
from .simple_clustering import cluster_by_clean_answer_exact


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    return {
        "examples": output_path / "examples.jsonl",
        "prompts": output_path / "prompts.jsonl",
        "generations": output_path / "generations.jsonl",
        "parsed_generations": output_path / "parsed_generations.jsonl",
        "cleaned_generations": output_path / "cleaned_generations.jsonl",
        "answer_clusters": output_path / "answer_clusters.jsonl",
        "answer_scores": output_path / "answer_se_scores.csv",
        "generation_uq": output_path / "generation_uq.csv",
        "citation_uq": output_path / "citation_uq.csv",
        "analysis_report": output_path / "analysis_report.md",
        "summary": output_path / "summary.txt",
    }


def _clean_parsed_generation_records(parsed_generations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for record in parsed_generations:
        answer_text = str(record.get("answer_text") or record.get("raw_answer") or "")
        cleaned.append({**record, "clean_answer": clean_answer(answer_text)})
    return cleaned


def _validate_counts(
    *,
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    parsed_generations: list[dict[str, Any]],
    clusters: list[dict[str, Any]],
    num_samples: int,
) -> None:
    expected_generations = len(examples) * num_samples
    if len(generations) != expected_generations:
        raise RuntimeError(f"Expected {expected_generations} generations, got {len(generations)}.")
    if len(parsed_generations) != expected_generations:
        raise RuntimeError("Parsed generation count does not match raw generation count.")
    if len(clusters) != len(examples):
        raise RuntimeError("Answer cluster row count does not match example count.")


def run_archehr_se(
    *,
    data_path: str | Path,
    split: str = "dev",
    output_dir: str | Path | None = None,
    config: GenerationConfig | None = None,
    limit_examples: int | None = None,
    overwrite: bool = False,
    generator: TextGenerator | None = None,
    clustering_method: str = "nli",
    nli_config: NLIConfig | None = None,
    nli_scorer: EntailmentScorer | None = None,
) -> dict[str, Any]:
    """Run a simple ArchEHR-QA answer-level SE and citation-UQ baseline."""

    config = config or GenerationConfig(num_samples=10, max_new_tokens=256)
    config.validate()
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else project_root() / "outputs" / "archehr_se"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)

    examples = load_common_examples(
        dataset="archehr_qa",
        data_path=data_path,
        split=split,
        limit=limit_examples,
    )
    prompt_records = build_archehr_prompt_records(examples)

    if generator is None:
        try:
            generator = HuggingFaceCausalLMGenerator(config)
        except MissingGenerationDependency:
            raise
        except OSError as exc:
            raise RuntimeError(
                "Could not load the ArchEHR SE model. Check --model_name, cache access, "
                "--local_files_only, and server model path permissions."
            ) from exc

    generations = generate_answer_records(
        prompt_records,
        generator,
        num_samples=config.num_samples,
        model_name=config.model_name,
        generation_level="archehr_se",
        include_token_scores=True,
    )
    parsed_generations = parse_generation_records(generations)
    cleaned_generations = _clean_parsed_generation_records(parsed_generations)

    if clustering_method == "exact":
        answer_clusters = cluster_by_clean_answer_exact(cleaned_generations)
        for row in answer_clusters:
            row["clustering_method"] = "exact_clean_answer"
    elif clustering_method == "nli":
        nli_config = nli_config or NLIConfig(device=config.device)
        if nli_scorer is None:
            nli_scorer = HuggingFaceNLIScorer(nli_config)
        examples_by_id = {str(example["id"]): example for example in examples}
        answer_clusters = cluster_by_bidirectional_entailment(
            cleaned_generations,
            scorer=nli_scorer,
            examples_by_id=examples_by_id,
            strict_entailment=nli_config.strict_entailment,
        )
    else:
        raise ValueError(f"Unsupported clustering_method: {clustering_method}")

    answer_scores = score_clusters(answer_clusters)
    gen_uq = generation_uq_rows(cleaned_generations)
    cit_uq = citation_uq_rows(cleaned_generations)

    _validate_counts(
        examples=examples,
        generations=generations,
        parsed_generations=parsed_generations,
        clusters=answer_clusters,
        num_samples=config.num_samples,
    )

    write_jsonl(examples, paths["examples"], overwrite=overwrite)
    write_jsonl(prompt_records, paths["prompts"], overwrite=overwrite)
    write_jsonl(generations, paths["generations"], overwrite=overwrite)
    write_jsonl(parsed_generations, paths["parsed_generations"], overwrite=overwrite)
    write_jsonl(cleaned_generations, paths["cleaned_generations"], overwrite=overwrite)
    write_jsonl(answer_clusters, paths["answer_clusters"], overwrite=overwrite)
    write_csv(answer_scores, paths["answer_scores"], SCORE_FIELDS, overwrite=overwrite)
    write_csv(gen_uq, paths["generation_uq"], GENERATION_UQ_FIELDS, overwrite=overwrite)
    write_csv(cit_uq, paths["citation_uq"], CITATION_UQ_FIELDS, overwrite=overwrite)

    result = {
        "level": "archehr_se",
        "dataset": "archehr_qa",
        "split": split,
        "data_path": str(data_path),
        "num_examples": len(examples),
        "num_generations": len(generations),
        "model_name": config.model_name,
        "num_samples": config.num_samples,
        "device": config.device,
        "torch_dtype": config.torch_dtype or "",
        "local_files_only": config.local_files_only,
        "requested_clustering_method": clustering_method,
        "clustering_method": answer_clusters[0]["clustering_method"] if answer_clusters else clustering_method,
        "nli_model_name": nli_config.model_name if nli_config and clustering_method == "nli" else "",
        "output_dir": str(output_dir),
        "paths": {name: str(path) for name, path in paths.items()},
    }
    summary = format_summary(result, answer_scores, cit_uq)
    if paths["summary"].exists() and not overwrite:
        raise FileExistsError(
            f"{paths['summary']} already exists. Pass --overwrite if replacing it is intended."
        )
    paths["summary"].write_text(summary + "\n", encoding="utf-8")
    if paths["analysis_report"].exists() and not overwrite:
        raise FileExistsError(
            f"{paths['analysis_report']} already exists. Pass --overwrite if replacing it is intended."
        )
    paths["analysis_report"].write_text(
        format_analysis_report(result, examples, answer_scores, cit_uq) + "\n",
        encoding="utf-8",
    )
    return result


def format_summary(
    result: dict[str, Any],
    answer_scores: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "ArchEHR-QA Semantic Entropy baseline complete",
        f"dataset: {result['dataset']}",
        f"split: {result['split']}",
        f"data_path: {result['data_path']}",
        f"examples: {result['num_examples']}",
        f"generations: {result['num_generations']}",
        f"model_name: {result['model_name']}",
        f"num_samples: {result['num_samples']}",
        f"device: {result['device']}",
        f"torch_dtype: {result['torch_dtype']}",
        f"local_files_only: {result['local_files_only']}",
        f"requested_clustering_method: {result['requested_clustering_method']}",
        f"clustering_method: {result['clustering_method']}",
        f"nli_model_name: {result['nli_model_name']}",
        f"output_dir: {result['output_dir']}",
        "answer_se_scores:",
    ]
    for row in answer_scores:
        lines.append(
            "- {example_id}: clusters={cluster_sizes}, H={semantic_entropy}, H_norm={normalized_semantic_entropy}".format(
                **row
            )
        )
    lines.append("citation_uq:")
    for row in citation_rows:
        lines.append(
            "- {example_id}: unique_sets={num_unique_citation_sets}, H_cite_norm={normalized_citation_set_entropy}, mean_jaccard={mean_pairwise_citation_jaccard}".format(
                **row
            )
        )
    return "\n".join(lines)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _top_rows(rows: list[dict[str, Any]], field: str, *, reverse: bool, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _as_float(row.get(field)), reverse=reverse)[:limit]


def format_analysis_report(
    result: dict[str, Any],
    examples: list[dict[str, Any]],
    answer_scores: list[dict[str, Any]],
    citation_rows: list[dict[str, Any]],
) -> str:
    examples_by_id = {str(example["id"]): example for example in examples}
    citation_by_id = {str(row["example_id"]): row for row in citation_rows}

    lines = [
        "# ArchEHR-QA Semantic Entropy Baseline Report",
        "",
        "## Run",
        "",
        f"- dataset: `{result['dataset']}`",
        f"- split: `{result['split']}`",
        f"- examples: `{result['num_examples']}`",
        f"- generations: `{result['num_generations']}`",
        f"- model: `{result['model_name']}`",
        f"- samples per example: `{result['num_samples']}`",
        f"- clustering: `{result['clustering_method']}`",
        "",
        "## Highest Answer SE",
        "",
    ]
    for row in _top_rows(answer_scores, "normalized_semantic_entropy", reverse=True):
        example = examples_by_id.get(str(row["example_id"]), {})
        citation_row = citation_by_id.get(str(row["example_id"]), {})
        lines.extend(
            [
                f"- `{row['example_id']}`: H_norm={row['normalized_semantic_entropy']}, "
                f"clusters={row['cluster_sizes']}, citation_H_norm={citation_row.get('normalized_citation_set_entropy', '')}",
                f"  - question: {example.get('question', '')}",
            ]
        )

    lines.extend(["", "## Lowest Answer SE", ""])
    for row in _top_rows(answer_scores, "normalized_semantic_entropy", reverse=False):
        example = examples_by_id.get(str(row["example_id"]), {})
        citation_row = citation_by_id.get(str(row["example_id"]), {})
        lines.extend(
            [
                f"- `{row['example_id']}`: H_norm={row['normalized_semantic_entropy']}, "
                f"clusters={row['cluster_sizes']}, citation_H_norm={citation_row.get('normalized_citation_set_entropy', '')}",
                f"  - question: {example.get('question', '')}",
            ]
        )

    lines.extend(["", "## Highest Citation Entropy", ""])
    for row in _top_rows(citation_rows, "normalized_citation_set_entropy", reverse=True):
        example = examples_by_id.get(str(row["example_id"]), {})
        lines.extend(
            [
                f"- `{row['example_id']}`: citation_H_norm={row['normalized_citation_set_entropy']}, "
                f"unique_sets={row['num_unique_citation_sets']}, mean_jaccard={row['mean_pairwise_citation_jaccard']}",
                f"  - question: {example.get('question', '')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This report describes uncertainty structure only.",
            "- It does not evaluate answer correctness unless separate gold-quality annotations are added.",
            "- High answer SE means sampled answers formed multiple semantic clusters.",
            "- High citation entropy means sampled answers cited unstable evidence sentence sets.",
        ]
    )
    return "\n".join(lines)
