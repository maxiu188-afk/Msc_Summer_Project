# Level 4 Progress

Last updated: 2026-07-03

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
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5
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

1. Add a maintained evaluation script for PubMedQA yes/no/maybe extraction.
2. Compute AUROC/rejection curves for detecting incorrect majority answers.
3. Decide whether to increase sampling to `N=10` for smoother SE values.
4. Add the next dataset adapter or begin SEP target generation from the Level 4 artifacts.
