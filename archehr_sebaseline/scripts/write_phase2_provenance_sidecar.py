"""Append immutable reproducibility provenance to a completed Phase-2 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


SIDECAR_NAME = "phase2_provenance_sidecar.json"
CORE_ARTIFACT_DIRS = ("hidden_states", "uq_baselines")
DEFAULT_RUNTIME_FILES = (
    "src/archehr_sebaseline/generation.py",
    "src/archehr_sebaseline/phase2_artifacts.py",
    "scripts/run_phase2_bioasq_artifacts.py",
    "scripts/check_phase2_bioasq_artifacts.py",
    "scripts/run_phase2_bioasq_isambard.sbatch",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def collect_artifact_hashes(run_dir: Path) -> dict[str, str]:
    """Hash every core artifact while excluding later post-processing outputs."""

    paths = [
        run_dir / "examples.jsonl",
        run_dir / "prompts.jsonl",
        run_dir / "best_generations.jsonl",
        run_dir / "phase2_run_metadata.json",
        run_dir / "run_timing.txt",
        run_dir / "health_check.txt",
    ]
    for directory_name in CORE_ARTIFACT_DIRS:
        directory = run_dir / directory_name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Core Phase-2 artifacts are missing: {missing[:3]}")
    return {
        str(path.relative_to(run_dir)): sha256_file(path)
        for path in sorted(paths, key=lambda item: str(item))
    }


def runtime_bundle_hash(project_root: Path, relative_paths: Iterable[str]) -> tuple[str, dict[str, str]]:
    file_hashes = {}
    for relative_path in relative_paths:
        path = project_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Runtime source file is missing: {path}")
        file_hashes[relative_path] = sha256_file(path)
    return canonical_json_sha256(file_hashes), file_hashes


def _run_optional(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def model_and_tokenizer_provenance(model_name: str, hf_cache_dir: Path | None) -> dict[str, Any]:
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("This command requires the collection Transformers/Torch environment.") from exc

    kwargs = {"local_files_only": True}
    if hf_cache_dir is not None:
        # ``HF_HOME`` is the parent that contains ``hub/``. Passing it as
        # ``cache_dir`` would instead make Transformers search
        # ``$HF_HOME/models--...`` and miss the actual snapshot.
        os.environ["HF_HOME"] = str(hf_cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(hf_cache_dir)
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name, **kwargs)
    tokenizer = processor.tokenizer
    vocab = tokenizer.get_vocab()
    special_tokens = getattr(tokenizer, "special_tokens_map", {})
    return {
        "model": {
            "repository": model_name,
            "resolved_revision": getattr(config, "_commit_hash", None),
            "config_sha256": canonical_json_sha256(config.to_dict()),
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "repository": model_name,
            "snapshot_revision": getattr(config, "_commit_hash", None),
            "vocab_size": len(vocab),
            "vocab_sha256": canonical_json_sha256(vocab),
            "special_tokens_sha256": canonical_json_sha256(special_tokens),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--model_name", default="google/gemma-3-12b-it")
    parser.add_argument("--hf_cache_dir", type=Path, default=None)
    parser.add_argument("--source_git_commit", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_path = run_dir / SIDECAR_NAME
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite.")
    metadata_path = run_dir / "phase2_run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Phase-2 metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    project_root = PROJECT_ROOT
    bundle_sha256, source_hashes = runtime_bundle_hash(project_root, DEFAULT_RUNTIME_FILES)
    provenance = model_and_tokenizer_provenance(args.model_name, args.hf_cache_dir)
    source_commit = args.source_git_commit or _run_optional(["git", "rev-parse", "HEAD"])
    payload = {
        "schema_version": "phase2_provenance_sidecar_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "original_metadata": {
            "path": "phase2_run_metadata.json",
            "sha256": sha256_file(metadata_path),
            "data_sha256": metadata.get("data_sha256"),
            "split_manifest_sha256": metadata.get("split_manifest_sha256"),
        },
        "collection_contract": metadata.get("hidden_states"),
        "collection_uq": metadata.get("uq"),
        **provenance,
        "collector_source": {
            "git_commit_recorded_after_run": source_commit,
            "runtime_bundle_sha256": bundle_sha256,
            "files": source_hashes,
        },
        "core_artifact_sha256": collect_artifact_hashes(run_dir),
        "scope": "Core generation, single-answer UQ, and hidden-state artifacts only; Claude post-processing is separate.",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output_path}")
    print(f"model_revision: {payload['model']['resolved_revision']}")
    print(f"runtime_bundle_sha256: {bundle_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
