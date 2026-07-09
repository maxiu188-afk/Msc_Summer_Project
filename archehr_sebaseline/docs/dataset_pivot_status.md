# Dataset Pivot Status

Last updated: 2026-07-09

## Decision

ArchEHR-QA should no longer be treated as the final evaluation or SEP training
dataset for this project.

The ArchEHR-QA SE baseline is still valuable as an engineering milestone:

- XML/JSON/JSONL loading works.
- Structured grounded prompting works.
- JSON answer parsing works reliably for Gemma 3 12B.
- Multi-sample generation artifacts are complete.
- NLI clustering and answer Semantic Entropy run end to end.
- Citation-set uncertainty and token-level UQ are saved.
- Lightweight evaluation artifacts and plots are generated.

However, the released ArchEHR-QA test key does not provide the gold supervision
needed for final uncertainty evaluation.

## Label Availability

### Dev Split

The dev key contains clinician answers and sentence relevance labels:

```text
case_id
clinician_answer
answers: [{sentence_id, relevance}]
```

This supports lightweight evaluation of:

- citation F1 against essential/supplementary evidence,
- answer-reference overlap,
- AUROC/ECE/rejection curves against a derived low-quality target.

### Test Split

The test key contains only:

```text
case_id
clinician_answer
```

It does not contain:

- sentence relevance labels,
- evidence labels,
- citation labels,
- answer-quality labels,
- citation markers inside clinician answers.

Therefore test evaluation can only use reference-only lexical/coverage metrics.
This is useful for diagnostics, but it is not a strong factuality or
uncertainty benchmark.

## Latest ArchEHR-QA Runs

### Dev

Gemma 3 12B dev run:

```text
examples: 20
generations: 200
num_samples: 10
model: google/gemma-3-12b-it
```

Main finding:

```text
Answer quality can be evaluated on dev because sentence relevance labels exist.
SE AUROC remains weak, partly because stable low-quality answers often have low SE.
```

### Test

Gemma 3 12B test run:

```text
examples: 100
generations: 1000
num_samples: 10
parse_status: json for 1000/1000 generations
```

Reference-only evaluation:

```text
mean_example_quality_score: 0.216
low_quality_rate: 0.48
answer SE = 0 for 64/100 examples
```

Important finding:

```text
low-quality examples with answer SE = 0: 40/48
non-low-quality examples with answer SE = 0: 24/52
```

This shows a key limitation of the current SE baseline: stable but incomplete or
wrong answers can have low Semantic Entropy.

## Consequence For SEP

SEP needs a meaningful target. For this project, the target should ideally be:

- raw SE plus a meaningful correctness/quality label, or
- high/low uncertainty derived from a dataset where answer quality can be
  evaluated reliably.

ArchEHR-QA test does not provide enough supervision for that. Dev is too small
and should not be the main target source.

## What To Reuse

Keep and reuse:

- `generation.py`
- `prompting.py`
- `answer_parsing.py`
- `nli_clustering.py`
- `semantic_scores.py`
- `baseline_uq.py`
- `uncertainty_metrics.py`
- the CLI/Slurm structure
- the artifact layout and tests

Adapt or replace:

- dataset adapter,
- prompt template if the new dataset format differs,
- answer-quality evaluator,
- gold-label loading logic.

## Next Dataset Requirements

The replacement dataset should preferably have:

- open/public access or approved local access,
- answer-quality labels or clear correctness labels,
- enough examples for stable AUROC/ECE estimates,
- long-form or explanation-style answers if possible,
- compatibility with multi-sample generation,
- no requirement to use private patient text outside approved storage.

Minimum acceptable target:

```text
question/context/reference answer/quality label
```

Better target:

```text
question/context/reference answer/evidence labels/factuality or correctness label
```

## Current Next Step

Choose a replacement dataset, then add a new adapter under:

```text
src/archehr_sebaseline/dataset_adapters.py
```

and a dataset-specific evaluator under:

```text
src/archehr_sebaseline/evaluation/
```

The existing ArchEHR-QA pipeline should remain available as a smoke/diagnostic
baseline, but the plan should no longer depend on ArchEHR-QA gold labels.
