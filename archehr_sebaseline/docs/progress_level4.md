# Level 4 Progress

Last updated: 2026-07-18

## Current Note

This file records the historical PubMedQA Level 4 pilot. It is no longer the
active research path. The later ArchEHR-QA SE baseline has also been completed
as an engineering milestone, but ArchEHR-QA is now paused as a final benchmark
because its test split lacks gold evidence/quality labels.

Current active planning is in:

```text
docs/dataset_pivot_status.md
docs/archived_low_usability/README.md
PHASE2_PROBE_PLAN.md
```

The active BioASQ medical-UQ two-seed run is complete. Its Phase-1 reference
uses no evidence, PubMedBERT-MNLI-MedNLI, and binary Claude labels. The active
Phase-2 direction is now P(True)-Probe plus Claude-label Accuracy-Probe; see
the root `PHASE2_PROBE_PLAN.md` rather than this historical Level 4 record.

## Completed Pilot

The Level 4 Qwen2.5-7B PubMedQA pilot completed successfully on Isambard:

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

Health check passed for all required artifacts and row counts.

## Local Analysis Files

Downloaded server results are stored under:

```text
server_results/archived_low_usability/level4_qwen25_7b_nli_50x5
```

Generated analysis reports:

```text
analysis_report.md
label_heuristic_eval.md
```

## Initial Findings

- The output set is structurally sound: 50 examples, 250 generations, complete UQ CSVs, complete cluster records, and no non-finite values.
- NLI clustering found semantic variation in most examples: 12/50 examples had one semantic cluster; the rest split into 2-4 clusters.
- Discrete and likelihood-weighted SE are almost identical in this run, so likelihood weighting did not materially reorder uncertainty.
- Token-level uncertainty has only weak correlation with semantic entropy, which supports keeping SE as a separate answer-level uncertainty baseline.
- A PubMedQA yes/no/maybe heuristic gives majority-vote accuracy of 33/50. This is only a proxy, but it suggests the 7B outputs are meaningful enough for pilot analysis.

## Next Work

Historical next work has been completed or superseded:

1. PubMedQA label evaluation was implemented.
2. AUROC/rejection curves were implemented.
3. ArchEHR-QA `N=10` runs were completed.
4. SEP target generation is deferred until the BioASQ pilot and its evaluation
   target are stable; candidate-dataset comparison continues in parallel.
