"""Deterministic, leakage-aware BioASQ splits for the Phase-2 probe."""

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from typing import Any, Iterable


PHASE2_BIOASQ_TYPES = ("factoid", "list", "summary")
SPLITS = ("train", "validation", "test")
_QUESTION_NORMALIZE_RE = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalized_question_key(example: dict[str, Any]) -> str:
    """Return an exact-question grouping key, scoped to the BioASQ type."""

    question_type = str(example.get("bioasq_type") or "").strip().lower()
    question = str(example.get("question") or "").casefold()
    normalized = " ".join(_QUESTION_NORMALIZE_RE.sub(" ", question).split())
    if question_type not in PHASE2_BIOASQ_TYPES:
        raise ValueError(f"Unsupported Phase-2 BioASQ type: {question_type or '<missing>'}.")
    if not normalized:
        raise ValueError(f"BioASQ example {example.get('id')!r} has an empty question.")
    return f"{question_type}:{normalized}"


def _integer_targets(total: int, fractions: tuple[float, float, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative.")
    if len(fractions) != len(SPLITS) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12):
        raise ValueError("Split fractions must contain train, validation, and test values summing to one.")
    raw = [total * fraction for fraction in fractions]
    counts = [math.floor(value) for value in raw]
    for _, index in sorted(
        ((raw[index] - counts[index], index) for index in range(len(counts))),
        reverse=True,
    )[: total - sum(counts)]:
        counts[index] += 1
    # Keep compatibility with the project's macOS Python 3.9 smoke runtime.
    return dict(zip(SPLITS, counts))


def _allocate_groups(
    groups: list[tuple[str, list[dict[str, Any]]]],
    *,
    targets: dict[str, int],
) -> dict[str, str]:
    """Allocate complete exact-question groups as close as possible to quotas."""

    counts = {split: 0 for split in SPLITS}
    assignments: dict[str, str] = {}
    for group_key, examples in groups:
        group_size = len(examples)
        # Largest remaining quota keeps ordinary one-question groups exactly at
        # their requested counts and gives duplicate groups one shared split.
        split = max(
            SPLITS,
            key=lambda candidate: (targets[candidate] - counts[candidate], -SPLITS.index(candidate)),
        )
        assignments[group_key] = split
        counts[split] += group_size
    return assignments


def build_phase2_bioasq_split_manifest(
    examples: Iterable[dict[str, Any]],
    *,
    phase1_reference_ids: Iterable[str],
    seed: int = 20260719,
    validation_fraction: float = 0.10,
    test_fraction: float = 0.10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split all eligible training13b questions without test-set leakage.

    Phase-1 questions are already observed during baseline design, so they are
    train-side examples and can never enter validation or final test. The 80/10/10
    quota is calculated over the full count of each question type; Phase-1 IDs
    first consume that type's train quota and unobserved questions fill the
    remaining quota. Exact duplicate questions are grouped before assignment, so
    they cannot cross a split.
    """

    if not 0.0 <= validation_fraction < 1.0 or not 0.0 <= test_fraction < 1.0:
        raise ValueError("validation_fraction and test_fraction must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be below one.")

    phase1_ids = {str(example_id) for example_id in phase1_reference_ids}
    eligible = list(examples)
    ids = [str(example.get("id")) for example in eligible]
    if len(ids) != len(set(ids)):
        duplicates = sorted(example_id for example_id, count in Counter(ids).items() if count > 1)
        raise ValueError(f"BioASQ IDs must be unique; duplicates include {duplicates[:3]}.")
    unknown_phase1_ids = sorted(phase1_ids.difference(ids))
    if unknown_phase1_ids:
        raise ValueError(
            "Phase-1 reference IDs are absent from the supplied BioASQ dataset; "
            f"examples include {unknown_phase1_ids[:3]}."
        )

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in eligible:
        groups[normalized_question_key(example)].append(example)

    assigned: dict[str, str] = {}
    for group_key, group_examples in groups.items():
        if any(str(example["id"]) in phase1_ids for example in group_examples):
            assigned[group_key] = "train"

    fractions = (1.0 - validation_fraction - test_fraction, validation_fraction, test_fraction)
    for type_offset, question_type in enumerate(PHASE2_BIOASQ_TYPES):
        type_groups = [
            (group_key, group_examples)
            for group_key, group_examples in groups.items()
            if group_key.startswith(f"{question_type}:")
        ]
        unseen_groups = sorted(
            (
                (group_key, group_examples) for group_key, group_examples in type_groups if group_key not in assigned
            ),
            key=lambda item: item[0],
        )
        random.Random(seed + type_offset).shuffle(unseen_groups)
        total_targets = _integer_targets(sum(len(group_examples) for _, group_examples in type_groups), fractions)
        fixed_train_count = sum(
            len(group_examples) for group_key, group_examples in type_groups if group_key in assigned
        )
        if fixed_train_count > total_targets["train"]:
            raise ValueError(
                f"Phase-1 references consume {fixed_train_count} {question_type} examples, "
                f"above its {total_targets['train']} train quota. Increase the train fraction."
            )
        residual_targets = dict(total_targets)
        residual_targets["train"] -= fixed_train_count
        assigned.update(_allocate_groups(unseen_groups, targets=residual_targets))

    manifest = []
    for group_key, group_examples in groups.items():
        split = assigned[group_key]
        for example in group_examples:
            example_id = str(example["id"])
            manifest.append(
                {
                    "example_id": example_id,
                    "bioasq_type": str(example["bioasq_type"]).lower(),
                    "split": split,
                    "question_group": group_key,
                    "phase1_reference": example_id in phase1_ids,
                }
            )
    manifest.sort(key=lambda row: row["example_id"])

    by_split_type: dict[str, dict[str, int]] = {
        split: {question_type: 0 for question_type in PHASE2_BIOASQ_TYPES} for split in SPLITS
    }
    for row in manifest:
        by_split_type[row["split"]][row["bioasq_type"]] += 1
    summary = {
        "seed": seed,
        "eligible_types": list(PHASE2_BIOASQ_TYPES),
        "phase1_reference_count": len(phase1_ids),
        "target_split_fractions": {"train": fractions[0], "validation": validation_fraction, "test": test_fraction},
        "counts_by_split_and_type": by_split_type,
        "total_counts_by_split": {
            split: sum(by_split_type[split].values()) for split in SPLITS
        },
        "exact_question_groups": len(groups),
        "duplicate_question_groups": sum(1 for group_examples in groups.values() if len(group_examples) > 1),
    }
    validate_phase2_bioasq_split_manifest(manifest, phase1_reference_ids=phase1_ids)
    return manifest, summary


def validate_phase2_bioasq_split_manifest(
    manifest: Iterable[dict[str, Any]], *, phase1_reference_ids: Iterable[str]
) -> None:
    """Raise when a split manifest violates question-level isolation."""

    rows = list(manifest)
    ids = [str(row.get("example_id")) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Split manifest contains duplicate example IDs.")
    phase1_ids = {str(example_id) for example_id in phase1_reference_ids}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split = str(row.get("split"))
        if split not in SPLITS:
            raise ValueError(f"Unexpected split {split!r}.")
        group_splits[str(row.get("question_group"))].add(split)
        if str(row.get("example_id")) in phase1_ids and split != "train":
            raise ValueError("A Phase-1 reference question was assigned outside train.")
    leaked_groups = [group for group, splits in group_splits.items() if len(splits) != 1]
    if leaked_groups:
        raise ValueError(f"Exact duplicate questions cross splits; examples include {leaked_groups[:3]}.")
