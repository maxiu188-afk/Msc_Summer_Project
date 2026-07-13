# Dataset Pivot Status

Last updated: 2026-07-13

## Decision

BioASQ Task B is the current main replacement-dataset direction. Immediate work
should test its supported summary, factoid, list, and yes/no paths. Continued
candidate-dataset exploration remains a supervisor recommendation and should be
treated as parallel due diligence rather than a pause in BioASQ testing.

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

## Update After Meeting With Simpson (2026-07-10)

The dataset pivot now includes an evaluation-method pivot as well:

- Test BioASQ as the main direction while continuing to compare several
  candidate datasets before fixing the final target permanently.
- Rework answer-quality evaluation for the chosen dataset.
- Do not carry over the current fixed thresholds, heuristic parsing rules, and
  manually weighted quality formula without validation.
- Trial an LLM-based answer evaluator as a more flexible option for judging
  correctness, relevance, factual support, and evidence use.
- Move to SEP experiments after the dataset and evaluation method are stable.

An LLM evaluator may be particularly useful for long-form answers where exact
match and lexical-overlap metrics are too brittle. The first implementation
should retain the current deterministic metrics as baselines and save the full
judge prompt, rubric, model/configuration, score, rationale, and raw response.
The LLM judge should be validated against gold labels where available, or a
small manually reviewed sample, before its scores are used as SEP supervision.

## BioASQ Grounded Batch Update (2026-07-13)

The first required BioASQ batch completed with Gemma 3 12B, ten sampled answers
per question, provided gold PubMed snippets in the prompt, token-level scores,
and bidirectional-entailment NLI clustering. It contains Golden summary (80
questions), Golden factoid (50), Golden list (50), and a training-summary
repeat (50). All runs passed the Level 4 health check.

The local BioASQ evaluator is useful for an initial SE signal but remains a
provisional quality target. Discrete SE AUROC against its low-quality flag is
0.471 for Golden summary, 0.676 for factoid, 0.606 for list, and 0.624 for the
training-summary repeat. Different question types, sources, and small positive
counts mean these values must not be pooled or treated as a model leaderboard.
The mixed quality rates (12.5%--36%) are sufficient for the intended filtering
study, so no generation-prompt change is warranted merely to improve lexical
quality.

Detailed figures, provenance, and limitations are in
`bioasq_runpod_results_20260713.md`. Future GPU runs return to Isambard.

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
- gold-label loading logic,
- fixed thresholds and manually weighted quality parameters,
- optional LLM-as-a-judge rubric and parsing logic.

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

Validate the current BioASQ quality target before SEP. In parallel, explore
replacement datasets and record, for each candidate:

- access and licensing constraints,
- task and answer format,
- available correctness/factuality/evidence supervision,
- dataset size and split quality,
- suitability for multi-sample Semantic Entropy,
- suitability for later SEP training and evaluation.

The BioASQ adapter and lightweight evaluator are already implemented. Before
using their labels for SEP, compare the deterministic evaluator with a fixed
LLM judge or a manually reviewed subset, then add Isambard seed repeats of the
most informative subsets. After the target is defensible, proceed with the
existing adapter under:

```text
src/archehr_sebaseline/dataset_adapters.py
```

and a dataset-specific evaluator under:

```text
src/archehr_sebaseline/evaluation/
```

The evaluation stage should compare, where feasible:

1. direct gold-label or exact task metrics,
2. lightweight deterministic/rule-based baselines,
3. an LLM-based evaluator with a fixed, versioned rubric.

The existing ArchEHR-QA pipeline should remain available as a smoke/diagnostic
baseline, but the plan should no longer depend on ArchEHR-QA gold labels.

Once this evaluation setup is stable, the project should proceed to the SEP
stage: save single-generation hidden states, use multi-sample SE as the target,
train lightweight probes, and compare fixed-token with uncertainty-aware token
selection/pooling.
