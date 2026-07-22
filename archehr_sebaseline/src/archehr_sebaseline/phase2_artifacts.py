"""Single-answer Phase-2 BioASQ artifact collection.

This module intentionally excludes semantic entropy, NLI clustering, and any
high-temperature alternative answers.  It creates the shared source artifacts
for the P(True)-Probe and Accuracy-Probe experiments.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .baseline_uq import EXAMPLE_UQ_FIELDS, GENERATION_UQ_FIELDS, example_uq_rows, generation_uq_rows
from .cleaning import clean_generation_records
from .data_io import read_jsonl, write_csv, write_jsonl
from .dataset_adapters import load_common_examples
from .generation import HuggingFaceCausalLMGenerator, generate_answer_records
from .phase2_splits import PHASE2_BIOASQ_TYPES, SPLITS
from .prompting import build_prompt_records
from .self_report_uq import (
    SELF_REPORT_EXAMPLE_FIELDS,
    SELF_REPORT_GENERATION_FIELDS,
    score_self_report_best_answers,
)


FEATURE_BLOCKS = (24, 32, 40, 48)
FEATURE_POSITIONS = ("TBG", "SLT", "LT")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest_examples(
    data_path: str | Path,
    split_manifest_path: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load all eligible questions and attach only their frozen split metadata."""

    examples = load_common_examples(
        dataset="bioasq_medical_uq",
        data_path=data_path,
        split="train13b",
    )
    examples_by_id = {str(example["id"]): example for example in examples}
    if len(examples_by_id) != len(examples):
        raise ValueError("Eligible BioASQ source contains duplicate IDs.")
    manifest_rows = read_jsonl(split_manifest_path)
    manifest_by_id = {str(row.get("example_id")): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("Phase-2 manifest contains duplicate example IDs.")
    if set(manifest_by_id) != set(examples_by_id):
        missing = sorted(set(examples_by_id).difference(manifest_by_id))
        extra = sorted(set(manifest_by_id).difference(examples_by_id))
        raise ValueError(
            "Phase-2 manifest and eligible BioASQ source disagree: "
            f"missing={missing[:3]}, extra={extra[:3]}."
        )

    assigned_examples = []
    for example_id, manifest_row in manifest_by_id.items():
        assigned_split = str(manifest_row.get("split") or "")
        source_type = str(examples_by_id[example_id].get("bioasq_type") or "").lower()
        manifest_type = str(manifest_row.get("bioasq_type") or "").lower()
        if assigned_split not in SPLITS:
            raise ValueError(f"Manifest example {example_id} has invalid split {assigned_split!r}.")
        if source_type not in PHASE2_BIOASQ_TYPES or manifest_type != source_type:
            raise ValueError(f"Manifest BioASQ type mismatch for {example_id}.")
        assigned_examples.append(
            {
                **examples_by_id[example_id],
                "split": assigned_split,
                "phase1_reference": bool(manifest_row.get("phase1_reference")),
                "question_group": str(manifest_row.get("question_group") or ""),
                "prompt_evidence_mode": "none",
            }
        )
    assigned_examples.sort(key=lambda row: (SPLITS.index(str(row["split"])), str(row["id"])))
    prompts = build_prompt_records(assigned_examples, include_evidence=False)
    return assigned_examples, prompts


def validate_phase2_examples(examples: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Check manifest shape before model work begins."""

    if not examples:
        raise ValueError("Phase-2 collection has no examples.")
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for example in examples:
        split = str(example.get("split"))
        question_type = str(example.get("bioasq_type") or "").lower()
        if split not in SPLITS or question_type not in PHASE2_BIOASQ_TYPES:
            raise ValueError(f"Unexpected Phase-2 example assignment: {example.get('id')!r}.")
        counts[split][question_type] += 1
    return {
        split: {question_type: counts[split][question_type] for question_type in PHASE2_BIOASQ_TYPES}
        for split in SPLITS
    }


def write_phase2_core_artifacts(
    *,
    output_dir: str | Path,
    examples: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    generator: HuggingFaceCausalLMGenerator,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    generation_level: str = "phase2_low_temperature_main_answer",
    progress_callback: Callable[[int, int, str, int], None] | None = None,
    overwrite: bool = False,
) -> list[dict[str, Any]]:
    """Generate exactly one low-temperature answer and its token-level UQ."""

    output_dir = Path(output_dir)
    raw_generations = generate_answer_records(
        prompts,
        generator,
        num_samples=1,
        model_name=model_name,
        generation_level=generation_level,
        include_token_scores=True,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        progress_callback=progress_callback,
    )
    generations = clean_generation_records(raw_generations)
    if len(generations) != len(examples):
        raise RuntimeError("Expected exactly one low-temperature generation per Phase-2 question.")
    write_jsonl(examples, output_dir / "examples.jsonl", overwrite=overwrite)
    write_jsonl(prompts, output_dir / "prompts.jsonl", overwrite=overwrite)
    write_jsonl(generations, output_dir / "best_generations.jsonl", overwrite=overwrite)
    write_csv(
        generation_uq_rows(generations),
        output_dir / "uq_baselines" / "single_answer_generation_uq.csv",
        GENERATION_UQ_FIELDS,
        overwrite=overwrite,
    )
    write_csv(
        example_uq_rows(generations),
        output_dir / "uq_baselines" / "single_answer_example_uq.csv",
        EXAMPLE_UQ_FIELDS,
        overwrite=overwrite,
    )
    return generations


def write_phase2_self_report_artifacts(
    *,
    output_dir: str | Path,
    examples: list[dict[str, Any]],
    best_generations: list[dict[str, Any]],
    generator: HuggingFaceCausalLMGenerator,
    overwrite: bool = False,
) -> None:
    """Score original blind P(True) and verbalized confidence on main answers."""

    generation_rows, example_rows = score_self_report_best_answers(examples, best_generations, generator)
    output_dir = Path(output_dir) / "uq_baselines"
    write_csv(
        generation_rows,
        output_dir / "self_report_generations.csv",
        SELF_REPORT_GENERATION_FIELDS,
        overwrite=overwrite,
    )
    write_csv(
        example_rows,
        output_dir / "self_report_examples.csv",
        SELF_REPORT_EXAMPLE_FIELDS,
        overwrite=overwrite,
    )


def write_phase2_hidden_state_artifacts(
    *,
    output_dir: str | Path,
    prompts: list[dict[str, Any]],
    best_generations: list[dict[str, Any]],
    generator: HuggingFaceCausalLMGenerator,
    transformer_blocks: tuple[int, ...] = FEATURE_BLOCKS,
    splits: tuple[str, ...] = SPLITS,
    progress_callback: Callable[[int, int, str], None] | None = None,
    overwrite: bool = False,
) -> None:
    """Extract the fixed 24/32/40/48 x TBG/SLT/LT feature grid by split."""

    prompts_by_id = {str(row["example_id"]): row for row in prompts}
    generations_by_id = {str(row["example_id"]): row for row in best_generations}
    if set(prompts_by_id) != set(generations_by_id):
        raise ValueError("Prompt and low-temperature generation IDs do not match.")
    torch = generator._torch  # Kept on the generator to guarantee matching runtime/model dtype.
    output_dir = Path(output_dir) / "hidden_states"
    if not splits or len(set(splits)) != len(splits):
        raise ValueError("Hidden-state splits must be a non-empty unique sequence.")
    vectors_by_split: dict[str, list[Any]] = {split: [] for split in splits}
    validity_by_split: dict[str, list[list[bool]]] = {split: [] for split in splits}
    metadata_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in splits}

    for index, example_id in enumerate(sorted(prompts_by_id), start=1):
        prompt_row = prompts_by_id[example_id]
        generation = generations_by_id[example_id]
        split = str(prompt_row.get("split"))
        if split not in splits:
            raise ValueError(f"Prompt {example_id} has invalid split {split!r}.")
        feature = generator.extract_answer_hidden_states(
            str(prompt_row["prompt"]),
            generated_token_ids=[int(token_id) for token_id in generation.get("generated_token_ids") or []],
            transformer_blocks=transformer_blocks,
        )
        vectors_by_split[split].append(feature.pop("vectors"))
        position_valid = feature.pop("position_valid")
        validity_by_split[split].append([bool(position_valid[name]) for name in FEATURE_POSITIONS])
        metadata_by_split[split].append(
            {
                "example_id": example_id,
                "split": split,
                "tensor_row": len(vectors_by_split[split]) - 1,
                "prompt_version": prompt_row.get("prompt_version"),
                **feature,
            }
        )
        if progress_callback is not None:
            progress_callback(index, len(prompts_by_id), example_id)

    for split in splits:
        if not vectors_by_split[split]:
            raise RuntimeError(f"No hidden-state vectors collected for split {split}.")
        tensor = torch.stack(vectors_by_split[split])
        valid = torch.tensor(validity_by_split[split], dtype=torch.bool)
        if list(tensor.shape[1:3]) != [len(transformer_blocks), len(FEATURE_POSITIONS)]:
            raise RuntimeError(f"Unexpected hidden-state feature shape for {split}: {list(tensor.shape)}.")
        artifact = {
            "schema_version": "phase2_hidden_states_v1",
            "tensor_layout": "[example, transformer_block, token_position, hidden_dimension]",
            "transformer_blocks": list(transformer_blocks),
            "token_positions": list(FEATURE_POSITIONS),
            "vectors": tensor,
            "position_valid": valid,
        }
        path = output_dir / f"phase2_hidden_states_{split}.pt"
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, path)
        write_jsonl(
            metadata_by_split[split],
            output_dir / f"phase2_hidden_states_{split}_index.jsonl",
            overwrite=overwrite,
        )


def write_run_metadata(
    *,
    output_dir: str | Path,
    data_path: str | Path,
    split_manifest_path: str | Path,
    model_name: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    max_input_tokens: int,
    counts: dict[str, dict[str, int]],
    overwrite: bool = False,
) -> None:
    """Persist the protocol contract and input hashes alongside every run."""

    path = Path(output_dir) / "phase2_run_metadata.json"
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite.")
    payload = {
        "schema_version": "phase2_single_answer_artifacts_v1",
        "data_path": str(Path(data_path).resolve()),
        "data_sha256": sha256_file(data_path),
        "split_manifest_path": str(Path(split_manifest_path).resolve()),
        "split_manifest_sha256": sha256_file(split_manifest_path),
        "model_name": model_name,
        "generation": {
            "num_samples": 1,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_new_tokens": max_new_tokens,
            "max_input_tokens": max_input_tokens,
            "evidence_mode": "none",
        },
        "uq": {
            "direct_p_true": "blind_question_plus_main_answer",
            "included": [
                "p_true_blind_uncertainty",
                "verbalized_confidence_uncertainty",
                "sequence_nll",
                "normalized_nll",
                "mean_token_entropy",
                "max_token_entropy",
            ],
            "excluded": ["semantic_entropy", "cluster_count", "sample_disagreement", "p_true_10"],
        },
        "hidden_states": {
            "transformer_blocks": list(FEATURE_BLOCKS),
            "token_positions": list(FEATURE_POSITIONS),
            "dtype": "bfloat16",
            "layout": "[example, transformer_block, token_position, hidden_dimension]",
            "forward_pass": "stored_prompt_plus_saved_generated_token_ids_no_cache",
        },
        "counts_by_split_and_type": counts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
