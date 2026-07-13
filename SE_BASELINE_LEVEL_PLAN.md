# ArchEHR-QA Semantic Entropy Baseline Plan

Last updated: 2026-07-13

## Current Decision

ArchEHR-QA is no longer the planned final evaluation or SEP training dataset.
The ArchEHR-QA pipeline is implemented and useful as an engineering baseline,
but the available test key has no gold evidence labels or answer-quality labels.

The current main replacement-dataset direction is BioASQ Task B. Its first
grounded Gemma 3 12B / NLI batch is complete: Golden summary (80 questions),
Golden factoid (50), Golden list (50), and a training-summary repeat (50), all
with ten samples per question. The existing ArchEHR-QA code should be retained
as a grounded long-form SE smoke/diagnostic pipeline.

The temporary Runpod recovery path is complete. Future long-running work should
return to Isambard; Runpod documents and scripts remain only as reproducibility
records for this completed batch.

## Supervisor Meeting Update (2026-07-10)

Following the discussion with Simpson, the next stages are:

1. Keep BioASQ as the current main direction while continuing to document and
   compare credible candidate datasets as a parallel research check.
2. Redesign the answer-quality evaluation for the selected dataset. The current
   evaluators rely on relatively rigid, dataset-specific rules, thresholds, and
   hand-set weights, so they should not be transferred unchanged.
3. Prototype an LLM-based evaluator for generated answers. This can provide a
   more flexible assessment of correctness, relevance, factual support, and
   evidence use when exact labels or lexical overlap are insufficient.
4. Once the dataset and evaluation target are stable, proceed to the SEP stage.

The LLM evaluator should initially be treated as an experimental comparison,
not an unquestioned gold standard. Its prompt/rubric, model version, decoding
settings, and raw judgments should be saved for reproducibility. Where possible,
its scores should be checked against available gold labels or a small manually
reviewed subset.

This document records the active plan after the Level 4 pilot cleanup. Earlier local smoke stages are complete and have been removed from the maintained code path. The active project now starts from the common-schema Level 3 pipeline and the Level 4 token-score/NLI baseline.

## Data Boundary

The real ArchEHR-QA dataset must not be read by Codex. Code may be written to load it later, but local verification by Codex should use fake or public/sanitized fixtures only. Restricted clinical datasets should be placed manually by the user and referenced through config or command-line paths.

## Active Baseline Status

### Level 3: Common-Schema Stability Run

Status: complete.

Level 3 supports common-schema examples and exact cleaned-answer clustering. PubMedQA ran successfully on the server with 10 examples, 3 samples per example, CUDA generation, and complete artifacts.

### Level 4: Development Subset Pilot

Status: complete as a pilot baseline.

The successful Level 4 run used:

```text
dataset: pubmedqa
examples: 50
generations: 250
model_name: Qwen/Qwen2.5-7B-Instruct
num_samples: 5
device: cuda
torch_dtype: float16
requested_clustering_method: nli
clustering_method: nli_bidirectional_entailment
nli_model_name: microsoft/deberta-v2-xlarge-mnli
token_scores: True
```

The structured health check passed. Outputs include:

- `examples.jsonl`
- `prompts.jsonl`
- `generations.jsonl`
- `cleaned_generations.jsonl`
- `clusters.jsonl`
- `se_scores.csv`
- `generation_uq.csv`
- `example_uq.csv`
- `summary.txt`
- `health_check.txt`

Local downloaded results and analysis are under:

```text
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5
```

Generated local reports:

```text
analysis_report.md
label_heuristic_eval.md
```

## Initial Level 4 Findings

- The artifact set is structurally sound: 50 examples, 250 generations, matching row counts, complete token-score fields, and no `NaN`/`Infinity` markers.
- NLI clustering found answer-level semantic variation in most examples: 12/50 examples had one semantic cluster; 38/50 split into 2-4 clusters.
- Discrete SE and likelihood-weighted SE are almost identical in this run, so likelihood weighting did not materially reorder examples.
- Predictive entropy and token-level UQ correlate weakly with semantic entropy, suggesting SE is capturing answer-level variation not reducible to token uncertainty.
- A simple PubMedQA yes/no/maybe heuristic gives 33/50 majority-vote accuracy. This is only a proxy, but it is enough to motivate a maintained evaluation script.

## Level 5 Direction

Goal: turn the Level 4 baseline into a reproducible evaluation setup with answer-quality targets.

Status: initial maintained implementation complete.

Implemented additions:

- PubMedQA label extraction/evaluation script.
- AUROC for detecting incorrect majority answers.
- Rejection/selective prediction curves.
- Unit tests for label extraction, majority voting, AUROC, and artifact writing.

Maintained entry points:

```text
archehr_sebaseline/src/archehr_sebaseline/evaluation/pubmedqa_labels.py
archehr_sebaseline/scripts/evaluate_pubmedqa_labels.py
archehr_sebaseline/tests/test_pubmedqa_label_eval.py
```

The local 50x5 Qwen2.5-7B pilot now has these Level 5 artifacts:

```text
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5\pubmedqa_label_predictions.csv
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5\pubmedqa_eval_summary.json
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5\rejection_curve.csv
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5\auroc_bar.svg
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5\rejection_curve.svg
```

Current maintained-evaluator results:

- Majority-vote accuracy: `28/50 = 0.560`.
- Known per-sample label accuracy: about `0.649`.
- Unknown sample predictions: `99/250`.
- AUROC for detecting incorrect majority answers:
  - normalized discrete SE: about `0.547`.
  - normalized likelihood-weighted SE: about `0.551`.
  - predictive entropy / mean normalized NLL: about `0.570`.
  - mean token entropy: about `0.557`.

Important note: the maintained evaluator is more conservative than the earlier
temporary `label_heuristic_eval.md`: tied known votes become `unknown`, so its
majority accuracy is lower but more reproducible.

Remaining Level 5 follow-ups:

- Optional `N=10` run for smoother SE estimates.
- Explore multiple public candidate datasets and document their supervision,
  answer format, scale, and compatibility with free-form generation.
- Adapt the evaluation method to the selected dataset rather than reusing the
  current hard-coded parameters unchanged.
- Add an optional LLM-as-a-judge evaluation path and compare it with available
  label-based or rule-based evaluation.
- Prepare SEP targets only after the dataset and evaluator are stable.

## SEP Direction

After the replacement dataset and its baseline evaluation are stable:

1. Save single-generation hidden states from selected layers and token positions.
2. Use Level 4 multi-sample SE as the target.
3. Train lightweight probes to predict high/low SE or raw SE.
4. Compare fixed-token features against uncertainty-selected token pooling.

## Current Next Step

The main research path now shifts away from ArchEHR-QA as a final benchmark and
toward testing BioASQ Task B as the primary replacement dataset. PubMedQA and
ArchEHR-QA remain useful for engineering smoke tests and method diagnostics.
Other candidate datasets should still be compared as advised by Simpson, but
that comparison does not block the immediate BioASQ test.

## BioASQ Batch Update (2026-07-13)

All four required runs passed the Level 4 structural check with CUDA generation,
bidirectional-entailment NLI clustering, populated token scores, and finite
entropy values. The archive is
`server_results/bioasq_se_runpod_required_results_20260713.tar.gz`; the
earlier 100-question training-summary run remains extracted under
`server_results/runpod_bioasq_summary_gemma3_12b_100x10/`.

The lightweight evaluator gives different signals by task/source: Golden
summary discrete SE AUROC is 0.471 (10 low-quality examples), while Golden
factoid and list are 0.676 and 0.606; the 50-example training-summary repeat
is 0.624. These are small, differently distributed samples and use a local
approximation of BioASQ metric families, not official BioASQ scores. They are
evidence to validate the quality target, not a stable model ranking.

Current next step:

```text
On Isambard, keep the existing gold-snippet grounded prompt unchanged; compare
the deterministic evaluator against a fixed LLM judge or reviewed subset, then
run seed repeats of the most informative BioASQ subsets. Proceed to SEP only
after that quality target is defensible.
```

See:

```text
archehr_sebaseline/docs/archehr_se_baseline_plan.md
archehr_sebaseline/docs/dataset_pivot_status.md
```

SEP and hidden-state probe work is the next major stage after the replacement
dataset and evaluation target are stable.

Status: initial ArchEHR-QA SE baseline implementation complete.

Implemented outputs:

```text
examples.jsonl
prompts.jsonl
generations.jsonl
parsed_generations.jsonl
cleaned_generations.jsonl
answer_clusters.jsonl
answer_se_scores.csv
generation_uq.csv
citation_uq.csv
analysis_report.md
summary.txt
```

Latest practical status:

- ArchEHR-QA dev/test runs completed with Gemma 3 12B and NLI clustering.
- Test generation parsed cleanly: 1000/1000 generations were valid JSON.
- Test key lacks gold evidence labels, so evaluation falls back to reference-only
  diagnostics.
- This makes ArchEHR-QA unsuitable as the final AUROC/ECE benchmark for this
  project without additional labels.
