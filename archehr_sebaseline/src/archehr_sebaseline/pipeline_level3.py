"""Level 3 pipeline: server stability run with common-schema datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cleaning import clean_generation_records
from .dataset_adapters import load_common_examples
from .data_io import project_root, write_csv, write_jsonl
from .generation import (
    GenerationConfig,
    HuggingFaceCausalLMGenerator,
    MissingGenerationDependency,
    TextGenerator,
    generate_answer_records,
)
from .prompting import build_prompt_records
from .semantic_scores import SCORE_FIELDS, score_clusters
from .simple_clustering import cluster_by_clean_answer_exact


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    return {
        "examples": output_path / "examples.jsonl",
        "prompts": output_path / "prompts.jsonl",
        "generations": output_path / "generations.jsonl",
        "cleaned_generations": output_path / "cleaned_generations.jsonl",
        "clusters": output_path / "clusters.jsonl",
        "scores": output_path / "se_scores.csv",
        "summary": output_path / "summary.txt",
    }


def _validate_output_counts(
    *,
    examples: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    cleaned_generations: list[dict[str, Any]],
    cluster_records: list[dict[str, Any]],
    num_samples: int,
) -> None:
    expected_generations = len(examples) * num_samples
    if len(generations) != expected_generations:
        raise RuntimeError(
            f"Expected {expected_generations} generations, got {len(generations)}."
        )
    if len(cleaned_generations) != len(generations):
        raise RuntimeError("Cleaned generation count does not match raw generation count.")
    if len(cluster_records) != len(examples):
        raise RuntimeError("Cluster row count does not match example count.")


def run_level3(
    *,
    dataset: str = "fake",
    data_path: str | Path | None = None,
    split: str = "dev",
    output_dir: str | Path | None = None,
    config: GenerationConfig | None = None,
    limit_examples: int = 10,
    overwrite: bool = False,
    generator: TextGenerator | None = None,
) -> dict[str, Any]:
    """Run a small server stability check using the common dataset schema."""

    config = config or GenerationConfig(num_samples=3, max_new_tokens=96)
    config.validate()
    output_dir = Path(output_dir) if output_dir is not None else project_root() / "outputs" / "level3"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)

    examples = load_common_examples(
        dataset=dataset,
        data_path=data_path,
        split=split,
        limit=limit_examples,
    )
    prompt_records = build_prompt_records(examples)

    if generator is None:
        try:
            generator = HuggingFaceCausalLMGenerator(config)
        except MissingGenerationDependency:
            raise
        except OSError as exc:
            raise RuntimeError(
                "Could not load the Level 3 model. Check --model_name, cache access, "
                "--local_files_only, and server model path permissions."
            ) from exc

    generations = generate_answer_records(
        prompt_records,
        generator,
        num_samples=config.num_samples,
        model_name=config.model_name,
        generation_level="level3",
    )
    cleaned_generations = clean_generation_records(generations)
    cluster_records = cluster_by_clean_answer_exact(cleaned_generations)
    score_rows = score_clusters(cluster_records)

    _validate_output_counts(
        examples=examples,
        generations=generations,
        cleaned_generations=cleaned_generations,
        cluster_records=cluster_records,
        num_samples=config.num_samples,
    )

    write_jsonl(examples, paths["examples"], overwrite=overwrite)
    write_jsonl(prompt_records, paths["prompts"], overwrite=overwrite)
    write_jsonl(generations, paths["generations"], overwrite=overwrite)
    write_jsonl(cleaned_generations, paths["cleaned_generations"], overwrite=overwrite)
    write_jsonl(cluster_records, paths["clusters"], overwrite=overwrite)
    write_csv(score_rows, paths["scores"], SCORE_FIELDS, overwrite=overwrite)

    result = {
        "level": "level3",
        "dataset": dataset,
        "split": split,
        "data_path": str(data_path) if data_path is not None else "",
        "num_examples": len(examples),
        "num_generations": len(generations),
        "output_dir": str(output_dir),
        "model_name": config.model_name,
        "num_samples": config.num_samples,
        "device": config.device,
        "torch_dtype": config.torch_dtype or "",
        "local_files_only": config.local_files_only,
        "clustering_method": "exact_clean_answer",
        "score_rows": score_rows,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    summary = format_summary(result)
    if paths["summary"].exists() and not overwrite:
        raise FileExistsError(
            f"{paths['summary']} already exists. Pass --overwrite if replacing it is intended."
        )
    paths["summary"].write_text(summary + "\n", encoding="utf-8")
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        "Level 3 common-schema Semantic Entropy stability run complete",
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
        f"clustering_method: {result['clustering_method']}",
        f"output_dir: {result['output_dir']}",
        "scores:",
    ]
    for row in result["score_rows"]:
        lines.append(
            "- {example_id}: clusters={cluster_sizes}, H={semantic_entropy}, "
            "H_norm={normalized_semantic_entropy}".format(**row)
        )
    return "\n".join(lines)
