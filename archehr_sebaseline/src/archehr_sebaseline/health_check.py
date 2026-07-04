"""Health checks for Level 4 output artifacts."""

from __future__ import annotations

import ast
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LEVEL4_REQUIRED_FILES = {
    "examples": "examples.jsonl",
    "prompts": "prompts.jsonl",
    "generations": "generations.jsonl",
    "cleaned_generations": "cleaned_generations.jsonl",
    "clusters": "clusters.jsonl",
    "scores": "se_scores.csv",
    "generation_uq": "generation_uq.csv",
    "example_uq": "example_uq.csv",
    "summary": "summary.txt",
}

NORMALIZED_ENTROPY_FIELDS = [
    "normalized_discrete_semantic_entropy",
    "normalized_likelihood_weighted_semantic_entropy",
]

NONNEGATIVE_SCORE_FIELDS = [
    "discrete_semantic_entropy",
    "likelihood_weighted_semantic_entropy",
    "predictive_entropy",
]

TOKEN_SCORE_FIELDS = [
    "sequence_logprob",
    "mean_token_logprob",
    "sequence_nll",
    "normalized_nll",
    "mean_token_entropy",
    "max_token_entropy",
]


@dataclass
class HealthCheckResult:
    output_dir: Path
    passed: bool = True
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    summary: dict[str, str] = field(default_factory=dict)

    def ok(self, message: str) -> None:
        self.checks.append(message)

    def fail(self, message: str) -> None:
        self.passed = False
        self.failures.append(message)

    def format_report(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Level 4 output health check: {status}",
            f"output_dir: {self.output_dir}",
        ]
        if self.counts:
            lines.append("counts:")
            for key in sorted(self.counts):
                lines.append(f"- {key}: {self.counts[key]}")
        if self.summary:
            lines.append("summary:")
            for key in sorted(self.summary):
                lines.append(f"- {key}: {self.summary[key]}")
        if self.checks:
            lines.append("checks:")
            for check in self.checks:
                lines.append(f"- OK: {check}")
        if self.failures:
            lines.append("failures:")
            for failure in self.failures:
                lines.append(f"- FAIL: {failure}")
        return "\n".join(lines)


def parse_summary(path: Path) -> dict[str, str]:
    summary: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key and not key.startswith("- "):
            summary[key.strip()] = value.strip()
    return summary


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as infile:
        for line_number, line in enumerate(infile, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(value)
    return records


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as infile:
        return list(csv.DictReader(infile))


def _as_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _record_count(result: HealthCheckResult, name: str, count: int) -> None:
    result.counts[name] = count


def _check_equal(result: HealthCheckResult, label: str, actual: int, expected: int) -> None:
    if actual == expected:
        result.ok(f"{label} count is {actual}")
    else:
        result.fail(f"{label} count is {actual}, expected {expected}")


def _check_no_nan_text(result: HealthCheckResult, paths: dict[str, Path]) -> None:
    bad_paths = []
    for path in paths.values():
        text = path.read_text(encoding="utf-8", errors="replace")
        if "NaN" in text or "Infinity" in text or "-Infinity" in text:
            bad_paths.append(path.name)
    if bad_paths:
        result.fail(f"Non-finite text markers found in: {', '.join(sorted(bad_paths))}")
    else:
        result.ok("no NaN or Infinity text markers found")


def _check_clusters(result: HealthCheckResult, cluster_rows: list[dict[str, Any]]) -> None:
    for index, row in enumerate(cluster_rows, start=1):
        example_id = str(row.get("example_id", f"row-{index}"))
        semantic_ids = row.get("semantic_ids")
        cluster_sizes = row.get("cluster_sizes")
        num_samples = row.get("num_samples")
        if not isinstance(semantic_ids, list):
            result.fail(f"cluster {example_id} has non-list semantic_ids")
            continue
        if not isinstance(cluster_sizes, list):
            result.fail(f"cluster {example_id} has non-list cluster_sizes")
            continue
        if num_samples != len(semantic_ids):
            result.fail(
                f"cluster {example_id} has num_samples={num_samples}, "
                f"semantic_ids={len(semantic_ids)}"
            )
        if sum(int(size) for size in cluster_sizes) != len(semantic_ids):
            result.fail(f"cluster {example_id} cluster_sizes do not sum to num_samples")
        if cluster_sizes and (min(cluster_sizes) < 1 or max(cluster_sizes) > len(semantic_ids)):
            result.fail(f"cluster {example_id} has invalid cluster_sizes={cluster_sizes}")
    if not result.failures:
        result.ok("cluster rows have consistent semantic_ids and cluster_sizes")


def _parse_cluster_sizes(value: str) -> list[int] | None:
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, int) for item in parsed):
        return None
    return parsed


def _check_score_rows(result: HealthCheckResult, score_rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(score_rows, start=1):
        example_id = row.get("example_id") or f"row-{index}"
        num_samples = _as_int(row.get("num_samples"))
        num_clusters = _as_int(row.get("num_clusters"))
        cluster_sizes = _parse_cluster_sizes(row.get("cluster_sizes", ""))
        if num_samples is None or num_samples < 1:
            result.fail(f"score {example_id} has invalid num_samples={row.get('num_samples')}")
        if num_clusters is None or num_clusters < 1:
            result.fail(f"score {example_id} has invalid num_clusters={row.get('num_clusters')}")
        if cluster_sizes is None:
            result.fail(f"score {example_id} has invalid cluster_sizes={row.get('cluster_sizes')}")
        elif num_clusters is not None and len(cluster_sizes) != num_clusters:
            result.fail(f"score {example_id} cluster size count does not match num_clusters")
        if num_samples is not None and num_clusters is not None and num_clusters > num_samples:
            result.fail(f"score {example_id} has more clusters than samples")

        for field_name in NONNEGATIVE_SCORE_FIELDS:
            value = _as_float(row.get(field_name))
            if row.get(field_name) and value is None:
                result.fail(f"score {example_id} has non-finite {field_name}")
            elif value is not None and value < -1e-9:
                result.fail(f"score {example_id} has negative {field_name}={value}")

        for field_name in NORMALIZED_ENTROPY_FIELDS:
            value = _as_float(row.get(field_name))
            if row.get(field_name) and value is None:
                result.fail(f"score {example_id} has non-finite {field_name}")
            elif value is not None and not -1e-9 <= value <= 1.000000001:
                result.fail(f"score {example_id} has out-of-range {field_name}={value}")

    if not result.failures:
        result.ok("SE score rows have valid counts and entropy ranges")


def _check_token_scores(result: HealthCheckResult, generation_uq_rows: list[dict[str, str]]) -> None:
    missing = []
    for index, row in enumerate(generation_uq_rows, start=1):
        example_id = row.get("example_id") or f"row-{index}"
        sample_id = row.get("sample_id") or "?"
        for field_name in TOKEN_SCORE_FIELDS:
            if _as_float(row.get(field_name)) is None:
                missing.append(f"{example_id}/{sample_id}:{field_name}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        result.fail(f"missing or non-finite token score fields: {preview}{suffix}")
    else:
        result.ok("generation_uq token score fields are populated")


def check_level4_output_dir(
    output_dir: str | Path,
    *,
    expected_examples: int | None = None,
    expected_generations: int | None = None,
    expected_num_samples: int | None = None,
    require_cuda: bool = False,
    require_nli: bool = False,
    require_token_scores: bool = True,
) -> HealthCheckResult:
    """Validate the expected Level 4 output artifact set."""

    output_path = Path(output_dir)
    result = HealthCheckResult(output_dir=output_path)
    paths = {name: output_path / filename for name, filename in LEVEL4_REQUIRED_FILES.items()}

    missing = [path.name for path in paths.values() if not path.exists()]
    if missing:
        result.fail(f"missing required files: {', '.join(sorted(missing))}")
        return result
    result.ok("all required Level 4 files are present")

    try:
        examples = read_jsonl_records(paths["examples"])
        prompts = read_jsonl_records(paths["prompts"])
        generations = read_jsonl_records(paths["generations"])
        cleaned_generations = read_jsonl_records(paths["cleaned_generations"])
        clusters = read_jsonl_records(paths["clusters"])
        score_rows = read_csv_rows(paths["scores"])
        generation_uq_rows = read_csv_rows(paths["generation_uq"])
        example_uq_rows = read_csv_rows(paths["example_uq"])
        summary = parse_summary(paths["summary"])
    except (OSError, ValueError) as exc:
        result.fail(str(exc))
        return result

    result.summary = summary
    _record_count(result, "examples.jsonl", len(examples))
    _record_count(result, "prompts.jsonl", len(prompts))
    _record_count(result, "generations.jsonl", len(generations))
    _record_count(result, "cleaned_generations.jsonl", len(cleaned_generations))
    _record_count(result, "clusters.jsonl", len(clusters))
    _record_count(result, "se_scores.csv", len(score_rows))
    _record_count(result, "generation_uq.csv", len(generation_uq_rows))
    _record_count(result, "example_uq.csv", len(example_uq_rows))

    summary_examples = _as_int(summary.get("examples"))
    summary_generations = _as_int(summary.get("generations"))
    summary_num_samples = _as_int(summary.get("num_samples"))

    expected_examples = expected_examples or summary_examples or len(examples)
    if expected_num_samples is None:
        expected_num_samples = summary_num_samples
    if expected_generations is None and expected_num_samples is not None:
        expected_generations = expected_examples * expected_num_samples
    expected_generations = expected_generations or summary_generations or len(generations)

    _check_equal(result, "examples.jsonl", len(examples), expected_examples)
    _check_equal(result, "prompts.jsonl", len(prompts), expected_examples)
    _check_equal(result, "clusters.jsonl", len(clusters), expected_examples)
    _check_equal(result, "se_scores.csv", len(score_rows), expected_examples)
    _check_equal(result, "example_uq.csv", len(example_uq_rows), expected_examples)
    _check_equal(result, "generations.jsonl", len(generations), expected_generations)
    _check_equal(result, "cleaned_generations.jsonl", len(cleaned_generations), expected_generations)
    _check_equal(result, "generation_uq.csv", len(generation_uq_rows), expected_generations)

    if summary_examples is not None:
        _check_equal(result, "summary examples", summary_examples, len(examples))
    if summary_generations is not None:
        _check_equal(result, "summary generations", summary_generations, len(generations))

    if require_cuda and summary.get("device") != "cuda":
        result.fail(f"summary device is {summary.get('device')!r}, expected 'cuda'")
    elif require_cuda:
        result.ok("summary reports CUDA")

    if require_nli:
        if summary.get("requested_clustering_method") != "nli":
            result.fail("summary requested_clustering_method is not nli")
        elif summary.get("clustering_method") != "nli_bidirectional_entailment":
            result.fail("summary clustering_method is not nli_bidirectional_entailment")
        else:
            result.ok("summary reports NLI bidirectional-entailment clustering")

    if require_token_scores:
        if summary.get("token_scores") not in {"True", "true", "1"}:
            result.fail(f"summary token_scores is {summary.get('token_scores')!r}")
        _check_token_scores(result, generation_uq_rows)

    _check_no_nan_text(result, paths)
    _check_clusters(result, clusters)
    _check_score_rows(result, score_rows)
    return result
