"""Portable export and scoring for validation-selected frozen linear Probes."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenProbe:
    name: str
    track: str
    analysis: str
    transformer_block: int
    token_position: str
    target_threshold: float | None
    standardizer_mean: np.ndarray
    standardizer_scale: np.ndarray
    coefficient: np.ndarray
    intercept: float


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as infile:
        for chunk in iter(lambda: infile.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logistic_pipeline_parameters(pipeline: Any) -> dict[str, np.ndarray | float]:
    """Extract the fitted StandardScaler + binary logistic-regression state."""

    scaler = pipeline.named_steps.get("standardize")
    estimator = pipeline.named_steps.get("model")
    if scaler is None or estimator is None:
        raise ValueError("Frozen Probe must be a standardize/model pipeline.")
    mean = np.asarray(getattr(scaler, "mean_", None), dtype=np.float64).reshape(-1)
    scale = np.asarray(getattr(scaler, "scale_", None), dtype=np.float64).reshape(-1)
    coefficient = np.asarray(getattr(estimator, "coef_", None), dtype=np.float64)
    intercept = np.asarray(getattr(estimator, "intercept_", None), dtype=np.float64).reshape(-1)
    classes = np.asarray(getattr(estimator, "classes_", None))
    if coefficient.shape != (1, mean.size) or scale.shape != mean.shape or intercept.shape != (1,):
        raise ValueError("Frozen Probe has an unexpected fitted parameter shape.")
    if classes.tolist() != [0, 1]:
        raise ValueError(f"Frozen Probe classes must be [0, 1], got {classes.tolist()}.")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not np.isfinite(coefficient).all() or not np.isfinite(intercept).all():
        raise ValueError("Frozen Probe parameters contain a non-finite value.")
    if np.any(scale <= 0):
        raise ValueError("Frozen Probe standardizer scale must be positive.")
    return {
        "standardizer_mean": mean,
        "standardizer_scale": scale,
        "coefficient": coefficient.reshape(-1),
        "intercept": float(intercept[0]),
    }


def write_frozen_probe_bundle(
    metadata_path: str | Path,
    array_path: str | Path,
    probes: list[dict[str, Any]],
    *,
    source: dict[str, Any],
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write selected Probe parameters without serializing executable objects."""

    metadata_path = Path(metadata_path)
    array_path = Path(array_path)
    if not probes:
        raise ValueError("At least one frozen Probe is required.")
    if not overwrite and (metadata_path.exists() or array_path.exists()):
        raise FileExistsError("Frozen Probe bundle already exists; pass overwrite=True.")
    arrays: dict[str, np.ndarray] = {}
    rows = []
    seen: set[str] = set()
    for probe in probes:
        name = str(probe["name"])
        if not name or name in seen:
            raise ValueError(f"Frozen Probe name is empty or duplicated: {name!r}.")
        seen.add(name)
        parameters = logistic_pipeline_parameters(probe["pipeline"])
        prefix = name.replace("-", "_")
        keys = {}
        for field in ("standardizer_mean", "standardizer_scale", "coefficient"):
            key = f"{prefix}_{field}"
            arrays[key] = np.asarray(parameters[field], dtype=np.float64)
            keys[field] = key
        rows.append(
            {
                "name": name,
                "track": str(probe["track"]),
                "analysis": str(probe["analysis"]),
                "model": "standard_scaler_plus_binary_logistic_regression",
                "transformer_block": int(probe["transformer_block"]),
                "token_position": str(probe["token_position"]),
                "target_threshold": (
                    float(probe["target_threshold"])
                    if probe.get("target_threshold") is not None
                    else None
                ),
                "positive_class": str(probe["positive_class"]),
                "hidden_dimension": int(arrays[keys["coefficient"]].size),
                "array_keys": keys,
                "intercept": float(parameters["intercept"]),
            }
        )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(array_path, **arrays)
    payload = {
        "schema_version": "frozen_linear_probe_bundle_v1",
        "source": source,
        "array_file": array_path.name,
        "array_sha256": sha256_file(array_path),
        "probes": rows,
        "target_dataset_policy": "score only; never fit, recalibrate, select, or tune",
    }
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_frozen_probe_bundle(metadata_path: str | Path) -> dict[str, FrozenProbe]:
    metadata_path = Path(metadata_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "frozen_linear_probe_bundle_v1":
        raise ValueError("Unsupported frozen Probe bundle schema.")
    array_path = metadata_path.parent / str(payload.get("array_file") or "")
    if sha256_file(array_path) != payload.get("array_sha256"):
        raise ValueError("Frozen Probe parameter-array SHA-256 mismatch.")
    result: dict[str, FrozenProbe] = {}
    with np.load(array_path, allow_pickle=False) as arrays:
        for row in payload.get("probes") or []:
            keys = row.get("array_keys") or {}
            probe = FrozenProbe(
                name=str(row["name"]),
                track=str(row["track"]),
                analysis=str(row["analysis"]),
                transformer_block=int(row["transformer_block"]),
                token_position=str(row["token_position"]),
                target_threshold=(
                    float(row["target_threshold"])
                    if row.get("target_threshold") is not None
                    else None
                ),
                standardizer_mean=np.asarray(arrays[str(keys["standardizer_mean"])], dtype=np.float64),
                standardizer_scale=np.asarray(arrays[str(keys["standardizer_scale"])], dtype=np.float64),
                coefficient=np.asarray(arrays[str(keys["coefficient"])], dtype=np.float64),
                intercept=float(row["intercept"]),
            )
            if probe.name in result:
                raise ValueError(f"Duplicate frozen Probe name: {probe.name}.")
            if not (
                probe.standardizer_mean.shape
                == probe.standardizer_scale.shape
                == probe.coefficient.shape
                == (int(row["hidden_dimension"]),)
            ):
                raise ValueError(f"Frozen Probe {probe.name} parameter shapes disagree.")
            result[probe.name] = probe
    return result


def score_frozen_probe(probe: FrozenProbe, features: np.ndarray) -> np.ndarray:
    """Apply a frozen Probe exactly; no target-dataset fitting is possible here."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[1] != probe.coefficient.size:
        raise ValueError(
            f"Frozen Probe {probe.name} expects [N,{probe.coefficient.size}] features, "
            f"got {features.shape}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Target hidden states contain a non-finite value.")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"(divide by zero|overflow|invalid value) encountered in matmul",
            category=RuntimeWarning,
        )
        logits = ((features - probe.standardizer_mean) / probe.standardizer_scale) @ probe.coefficient
    logits = logits + probe.intercept
    scores = np.empty_like(logits, dtype=np.float64)
    positive = logits >= 0
    scores[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    scores[~positive] = exp_logits / (1.0 + exp_logits)
    if not np.isfinite(scores).all():
        raise RuntimeError("Frozen Probe produced a non-finite score.")
    return scores
