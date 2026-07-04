# ArchEHR-QA Semantic Entropy Baseline Plan

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
- Additional public dataset adapters if useful.
- Preparation for SEP target generation from Level 4 artifacts.

## SEP Direction

After the baseline evaluation is stable:

1. Save single-generation hidden states from selected layers and token positions.
2. Use Level 4 multi-sample SE as the target.
3. Train lightweight probes to predict high/low SE or raw SE.
4. Compare fixed-token features against uncertainty-selected token pooling.

## Current Next Step

The main research path now shifts away from PubMedQA and toward an ArchEHR-QA
Semantic Entropy baseline. PubMedQA remains useful only for engineering smoke
tests and should not be used as a main SEP or SE target.

Current next step:

```text
Build a simple ArchEHR-QA SE baseline using provided evidence sentences,
structured cited answers, answer-level SE, and citation-set uncertainty.
```

See:

```text
archehr_sebaseline/docs/archehr_se_baseline_plan.md
```

SEP and hidden-state probe work is intentionally deferred until the ArchEHR-QA
SE baseline is stable.

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

Next practical step is to run this pipeline on a user-provided sanitized or
server-side ArchEHR-QA data path with `num_samples=10` and NLI clustering.
