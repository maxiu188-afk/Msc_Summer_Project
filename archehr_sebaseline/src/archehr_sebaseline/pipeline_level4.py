"""Level 4 pipeline: pilot run with token-score UQ baselines."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from .baseline_uq import (
    EXAMPLE_UQ_FIELDS,
    GENERATION_UQ_FIELDS,
    example_uq_rows,
    generation_uq_rows,
)
from .cleaning import clean_generation_records
from .dataset_adapters import load_common_examples
from .data_io import project_root, write_csv, write_jsonl
from .entropy import (
    normalize_entropy_value,
    normalized_semantic_entropy,
    predictive_entropy_from_logprobs,
    semantic_entropy_from_cluster_sizes,
    semantic_entropy_from_ids_and_logprobs,
)
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
from .prompting import build_prompt_records
from .simple_clustering import cluster_by_clean_answer_exact


LEVEL4_SCORE_FIELDS = [
    "example_id",
    "dataset",
    "split",
    "num_samples",
    "num_clusters",
    "cluster_sizes",
    "discrete_semantic_entropy",
    "normalized_discrete_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
    "predictive_entropy",
    "mean_token_logprob",
    "mean_sequence_nll",
    "mean_normalized_nll",
    "mean_token_entropy",
    "clustering_method",
]


def _output_paths(output_dir: str | Path) -> dict[str, Path]:
    output_path = Path(output_dir)
    return {
        "examples": output_path / "examples.jsonl",
        "prompts": output_path / "prompts.jsonl",
        "generations": output_path / "generations.jsonl",
        "cleaned_generations": output_path / "cleaned_generations.jsonl",
        "best_generations": output_path / "best_generations.jsonl",
        "clusters": output_path / "clusters.jsonl",
        "scores": output_path / "se_scores.csv",
        "generation_uq": output_path / "generation_uq.csv",
        "example_uq": output_path / "example_uq.csv",
        "summary": output_path / "summary.txt",
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _numeric_values(records: list[dict[str, Any]], field: str) -> list[float]:
    values = []
    for record in records:
        value = record.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _print_progress(
    stage: str,
    completed: int,
    total: int,
    started_at: float,
    detail: str,
) -> None:
    fraction = completed / total if total else 1.0
    bar_width = 24
    filled = min(bar_width, int(bar_width * fraction))
    bar = "#" * filled + "-" * (bar_width - filled)
    elapsed = time.monotonic() - started_at
    remaining = elapsed * (total - completed) / completed if completed else 0.0
    message = (
        f"[{stage}] [{bar}] {completed}/{total} ({fraction:.1%}) "
        f"elapsed={_format_duration(elapsed)} eta={_format_duration(remaining)} {detail}"
    )
    print("\r" + message.ljust(150), end="\n" if completed == total else "", flush=True)


def score_level4_clusters(
    cluster_records: list[dict[str, Any]],
    cleaned_generations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    generations_by_example: dict[str, list[dict[str, Any]]] = {}
    for generation in cleaned_generations:
        generations_by_example.setdefault(str(generation["example_id"]), []).append(generation)

    score_rows = []
    for cluster_record in cluster_records:
        example_id = str(cluster_record["example_id"])
        records = sorted(
            generations_by_example[example_id],
            key=lambda generation: generation["sample_id"],
        )
        cluster_sizes = cluster_record["cluster_sizes"]
        discrete_entropy = semantic_entropy_from_cluster_sizes(cluster_sizes)
        normalized_discrete = normalized_semantic_entropy(
            cluster_sizes,
            num_samples=cluster_record["num_samples"],
        )

        mean_logprobs = _numeric_values(records, "mean_token_logprob")
        if len(mean_logprobs) == len(records):
            predictive_entropy = predictive_entropy_from_logprobs(mean_logprobs)
            weighted_entropy = semantic_entropy_from_ids_and_logprobs(
                cluster_record["semantic_ids"],
                mean_logprobs,
            )
            normalized_weighted = normalize_entropy_value(
                weighted_entropy,
                num_samples=cluster_record["num_samples"],
            )
        else:
            predictive_entropy = None
            weighted_entropy = None
            normalized_weighted = None

        row = {
            "example_id": example_id,
            "dataset": records[0].get("dataset"),
            "split": records[0].get("split"),
            "num_samples": cluster_record["num_samples"],
            "num_clusters": len(cluster_sizes),
            "cluster_sizes": str(cluster_sizes),
            "discrete_semantic_entropy": discrete_entropy,
            "normalized_discrete_semantic_entropy": normalized_discrete,
            "likelihood_weighted_semantic_entropy": weighted_entropy,
            "normalized_likelihood_weighted_semantic_entropy": normalized_weighted,
            "predictive_entropy": predictive_entropy,
            "mean_token_logprob": _mean(mean_logprobs),
            "mean_sequence_nll": _mean(_numeric_values(records, "sequence_nll")),
            "mean_normalized_nll": _mean(_numeric_values(records, "normalized_nll")),
            "mean_token_entropy": _mean(_numeric_values(records, "mean_token_entropy")),
            "clustering_method": cluster_record["clustering_method"],
        }
        score_rows.append({field: _fmt(row.get(field)) for field in LEVEL4_SCORE_FIELDS})
    return score_rows


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


def run_level4(
    *,
    dataset: str = "pubmedqa",
    data_path: str | Path | None = None,
    split: str = "dev",
    output_dir: str | Path | None = None,
    config: GenerationConfig | None = None,
    limit_examples: int = 50,
    overwrite: bool = False,
    generator: TextGenerator | None = None,
    clustering_method: str = "nli",
    nli_config: NLIConfig | None = None,
    nli_scorer: EntailmentScorer | None = None,
    show_progress: bool = False,
    include_evidence: bool = True,
    best_generation_temperature: float = 0.1,
    best_generation_top_p: float | None = None,
    best_generation_top_k: int | None = None,
) -> dict[str, Any]:
    """Run a Level 4 pilot with token-level baseline uncertainty outputs."""

    config = config or GenerationConfig(num_samples=5, max_new_tokens=128)
    config.validate()
    output_dir = Path(output_dir) if output_dir is not None else project_root() / "outputs" / "level4"
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _output_paths(output_dir)

    examples = load_common_examples(
        dataset=dataset,
        data_path=data_path,
        split=split,
        limit=limit_examples,
    )
    # Store the effective mode per example: a mixed BioASQ run deliberately
    # keeps summary questions evidence-free even if factoid/list use snippets.
    from .prompting import bioasq_should_include_evidence

    examples = [
        {
            **example,
            "prompt_evidence_mode": (
                "provided"
                if (
                    include_evidence
                    and (
                        str(example.get("dataset") or "").lower() != "bioasq"
                        or bioasq_should_include_evidence(example, include_evidence=True)
                    )
                )
                else "none"
            ),
        }
        for example in examples
    ]
    prompt_records = build_prompt_records(examples, include_evidence=include_evidence)
    if best_generation_temperature <= 0:
        raise ValueError("best_generation_temperature must be positive.")

    if generator is None:
        try:
            if show_progress:
                print("[model] Loading generation model...", flush=True)
            generator = HuggingFaceCausalLMGenerator(config)
            if show_progress:
                print("[model] Generation model loaded; starting sampled generation.", flush=True)
        except MissingGenerationDependency:
            raise
        except OSError as exc:
            raise RuntimeError(
                "Could not load the Level 4 model. Check --model_name, cache access, "
                "--local_files_only, and server model path permissions."
            ) from exc

    generation_started_at = time.monotonic()
    generation_progress_callback = None
    if show_progress:
        def generation_progress_callback(
            completed: int,
            total: int,
            example_id: str,
            sample_id: int,
        ) -> None:
            _print_progress(
                "generation",
                completed,
                total,
                generation_started_at,
                f"example={example_id} sample={sample_id + 1}/{config.num_samples}",
            )

    # The low-temperature answer is the accuracy target.  High-temperature
    # samples below remain exclusively for semantic-uncertainty estimation.
    best_raw_generations = generate_answer_records(
        prompt_records,
        generator,
        num_samples=1,
        model_name=config.model_name,
        generation_level="best_generation",
        include_token_scores=False,
        temperature=best_generation_temperature,
        top_p=config.top_p if best_generation_top_p is None else best_generation_top_p,
        top_k=config.top_k if best_generation_top_k is None else best_generation_top_k,
    )
    best_generations = clean_generation_records(best_raw_generations)

    generations = generate_answer_records(
        prompt_records,
        generator,
        num_samples=config.num_samples,
        model_name=config.model_name,
        generation_level="level4",
        include_token_scores=True,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        progress_callback=generation_progress_callback,
    )
    cleaned_generations = clean_generation_records(generations)
    if clustering_method == "exact":
        cluster_records = cluster_by_clean_answer_exact(cleaned_generations)
    elif clustering_method == "nli":
        nli_config = nli_config or NLIConfig(device=config.device)
        if nli_scorer is None:
            if show_progress:
                print("[nli] Loading NLI model...", flush=True)
            nli_scorer = HuggingFaceNLIScorer(nli_config)
            if show_progress:
                print("[nli] NLI model loaded; starting answer clustering.", flush=True)
        examples_by_id = {str(example["id"]): example for example in examples}
        nli_started_at = time.monotonic()
        nli_progress_callback = None
        if show_progress:
            def nli_progress_callback(completed: int, total: int, example_id: str) -> None:
                _print_progress(
                    "nli",
                    completed,
                    total,
                    nli_started_at,
                    f"example={example_id}",
                )

        cluster_records = cluster_by_bidirectional_entailment(
            cleaned_generations,
            scorer=nli_scorer,
            examples_by_id=examples_by_id,
            strict_entailment=nli_config.strict_entailment,
            progress_callback=nli_progress_callback,
        )
    else:
        raise ValueError(f"Unsupported clustering_method: {clustering_method}")

    if show_progress:
        print("[finalize] Computing uncertainty scores and writing artifacts...", flush=True)
    score_rows = score_level4_clusters(cluster_records, cleaned_generations)
    generation_uq = generation_uq_rows(cleaned_generations)
    example_uq = example_uq_rows(cleaned_generations)

    _validate_output_counts(
        examples=examples,
        generations=generations,
        cleaned_generations=cleaned_generations,
        cluster_records=cluster_records,
        num_samples=config.num_samples,
    )
    if len(best_generations) != len(examples):
        raise RuntimeError("Best-generation count does not match example count.")

    write_jsonl(examples, paths["examples"], overwrite=overwrite)
    write_jsonl(prompt_records, paths["prompts"], overwrite=overwrite)
    write_jsonl(generations, paths["generations"], overwrite=overwrite)
    write_jsonl(cleaned_generations, paths["cleaned_generations"], overwrite=overwrite)
    write_jsonl(best_generations, paths["best_generations"], overwrite=overwrite)
    write_jsonl(cluster_records, paths["clusters"], overwrite=overwrite)
    write_csv(score_rows, paths["scores"], LEVEL4_SCORE_FIELDS, overwrite=overwrite)
    write_csv(generation_uq, paths["generation_uq"], GENERATION_UQ_FIELDS, overwrite=overwrite)
    write_csv(example_uq, paths["example_uq"], EXAMPLE_UQ_FIELDS, overwrite=overwrite)

    result = {
        "level": "level4",
        "dataset": dataset,
        "split": split,
        "data_path": str(data_path) if data_path is not None else "",
        "num_examples": len(examples),
        "num_generations": len(generations),
        "num_best_generations": len(best_generations),
        "output_dir": str(output_dir),
        "model_name": config.model_name,
        "num_samples": config.num_samples,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "best_generation_temperature": best_generation_temperature,
        "best_generation_top_p": config.top_p if best_generation_top_p is None else best_generation_top_p,
        "best_generation_top_k": config.top_k if best_generation_top_k is None else best_generation_top_k,
        "device": config.device,
        "torch_dtype": config.torch_dtype or "",
        "local_files_only": config.local_files_only,
        "prompt_evidence_mode": "provided" if include_evidence else "none",
        "clustering_method": cluster_records[0]["clustering_method"] if cluster_records else clustering_method,
        "requested_clustering_method": clustering_method,
        "nli_model_name": nli_config.model_name if nli_config and clustering_method == "nli" else "",
        "token_scores": True,
        "score_rows": score_rows,
        "paths": {name: str(path) for name, path in paths.items()},
    }
    summary = format_summary(result)
    if paths["summary"].exists() and not overwrite:
        raise FileExistsError(
            f"{paths['summary']} already exists. Pass --overwrite if replacing it is intended."
        )
    paths["summary"].write_text(summary + "\n", encoding="utf-8")
    if show_progress:
        print("[complete] All artifacts and summary.txt were written.", flush=True)
    return result


def format_summary(result: dict[str, Any]) -> str:
    lines = [
        "Level 4 token-score Semantic Entropy pilot run complete",
        f"dataset: {result['dataset']}",
        f"split: {result['split']}",
        f"data_path: {result['data_path']}",
        f"examples: {result['num_examples']}",
        f"generations: {result['num_generations']}",
        f"best_generations: {result['num_best_generations']}",
        f"model_name: {result['model_name']}",
        f"num_samples: {result['num_samples']}",
        f"sampling_temperature: {result['temperature']}",
        f"sampling_top_p: {result['top_p']}",
        f"sampling_top_k: {result['top_k']}",
        f"best_generation_temperature: {result['best_generation_temperature']}",
        f"device: {result['device']}",
        f"torch_dtype: {result['torch_dtype']}",
        f"local_files_only: {result['local_files_only']}",
        f"prompt_evidence_mode: {result['prompt_evidence_mode']}",
        f"requested_clustering_method: {result['requested_clustering_method']}",
        f"clustering_method: {result['clustering_method']}",
        f"nli_model_name: {result['nli_model_name']}",
        f"token_scores: {result['token_scores']}",
        f"output_dir: {result['output_dir']}",
        "scores:",
    ]
    for row in result["score_rows"]:
        lines.append(
            "- {example_id}: clusters={cluster_sizes}, H_discrete={discrete_semantic_entropy}, "
            "H_weighted={likelihood_weighted_semantic_entropy}, PE={predictive_entropy}".format(
                **row
            )
        )
    return "\n".join(lines)
