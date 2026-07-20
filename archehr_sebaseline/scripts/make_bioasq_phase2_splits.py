"""Create the fixed BioASQ train/validation/test manifest for Phase 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from archehr_sebaseline.data_io import read_jsonl, write_jsonl
from archehr_sebaseline.dataset_adapters import load_common_examples
from archehr_sebaseline.phase2_splits import build_phase2_bioasq_split_manifest


PHASE1_TYPE_LIMITS = {"factoid": 480, "list": 320, "summary": 200}
PHASE1_SELECTION_SEED = 20260718


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True, help="Official BioASQ training13b JSON.")
    parser.add_argument(
        "--phase1_examples",
        type=Path,
        help="Phase-1 1,000-question examples.jsonl; its IDs are forced into train when available.",
    )
    parser.add_argument(
        "--reconstruct_phase1_reference",
        action="store_true",
        help=(
            "Reconstruct Phase-1 IDs from this pinned raw dataset using the recorded "
            "20260718 selection seed and 480/320/200 type quotas."
        ),
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--test_fraction", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.phase1_examples is None and not args.reconstruct_phase1_reference:
        print(
            "Provide --phase1_examples or explicitly use --reconstruct_phase1_reference.",
            file=sys.stderr,
        )
        return 2
    if args.phase1_examples is not None and args.reconstruct_phase1_reference:
        print("Use either --phase1_examples or --reconstruct_phase1_reference, not both.", file=sys.stderr)
        return 2
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        print(f"Refusing to overwrite non-empty output directory: {args.output_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = load_common_examples(dataset="bioasq_medical_uq", data_path=args.data_path, split="train13b")
    phase1_provenance: dict[str, object]
    if args.phase1_examples is not None:
        phase1_ids = [str(row["id"]) for row in read_jsonl(args.phase1_examples)]
        phase1_provenance = {
            "method": "phase1_examples_jsonl",
            "examples_sha256": _sha256(args.phase1_examples),
        }
    else:
        phase1_examples = load_common_examples(
            dataset="bioasq_medical_uq",
            data_path=args.data_path,
            split="train13b",
            limit=sum(PHASE1_TYPE_LIMITS.values()),
            bioasq_type_limits=PHASE1_TYPE_LIMITS,
            selection_seed=PHASE1_SELECTION_SEED,
        )
        phase1_ids = [str(row["id"]) for row in phase1_examples]
        phase1_provenance = {
            "method": "reconstructed_from_pinned_training13b",
            "selection_seed": PHASE1_SELECTION_SEED,
            "type_limits": PHASE1_TYPE_LIMITS,
        }
    manifest, summary = build_phase2_bioasq_split_manifest(
        examples,
        phase1_reference_ids=phase1_ids,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    manifest_path = args.output_dir / "phase2_bioasq_split_manifest.jsonl"
    summary_path = args.output_dir / "phase2_bioasq_split_summary.json"
    write_jsonl(manifest, manifest_path, overwrite=args.overwrite)
    summary.update(
        {
            "dataset": "BioASQ training13b",
            "data_path_sha256": _sha256(args.data_path),
            "phase1_reference_provenance": phase1_provenance,
        }
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(manifest)} assignments to {manifest_path}")
    print(json.dumps(summary["counts_by_split_and_type"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
