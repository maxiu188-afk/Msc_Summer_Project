# ArchEHR-QA UQ Code Handoff Guide

Last updated: 2026-07-03

The active project is `archehr_sebaseline`. The maintained path now starts at Level 3/4; earlier local smoke levels have been removed from the active codebase.

## Current Stage

Level 4 is complete as a pilot baseline:

- PubMedQA common-schema loading.
- HuggingFace causal-LM generation with Qwen2.5-7B-Instruct.
- Token log-probability, token entropy, sequence NLL, and per-example UQ summaries.
- Likelihood-weighted Semantic Entropy.
- NLI bidirectional-entailment clustering.
- Structured health-check validation.

The latest server result is:

```text
dataset: pubmedqa
examples: 50
generations: 250
device: cuda
model_name: Qwen/Qwen2.5-7B-Instruct
requested_clustering_method: nli
clustering_method: nli_bidirectional_entailment
nli_model_name: microsoft/deberta-v2-xlarge-mnli
token_scores: True
```

## Core Files

- `src/archehr_sebaseline/dataset_adapters.py`: fake fixtures and PubMedQA adapter.
- `src/archehr_sebaseline/generation.py`: HuggingFace generation and token scores.
- `src/archehr_sebaseline/nli_clustering.py`: bidirectional entailment clustering.
- `src/archehr_sebaseline/baseline_uq.py`: token-score UQ CSV summaries.
- `src/archehr_sebaseline/entropy.py`: discrete, weighted, and predictive entropy helpers.
- `src/archehr_sebaseline/semantic_scores.py`: shared SE score rows for Level 3.
- `src/archehr_sebaseline/health_check.py`: Level 4 output validation.
- `src/archehr_sebaseline/pipeline_level3.py`: common-schema exact-clustering stability path.
- `src/archehr_sebaseline/pipeline_level4.py`: main token-score/NLI pilot path.

## Entry Points

- `scripts/run_level3.py`
- `scripts/run_level3.sbatch`
- `scripts/run_level4.py`
- `scripts/run_level4.sbatch`
- `scripts/check_level4_outputs.py`

## Tests

Run from `D:\work\FinalProject\code`:

```powershell
python -m unittest discover archehr_sebaseline\tests
```

## Local Results

Downloaded 50x5 server results are under:

```text
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5
```

Analysis reports:

```text
analysis_report.md
label_heuristic_eval.md
```

## Next Work

1. Promote the PubMedQA label heuristic into a maintained evaluation script.
2. Compute AUROC and rejection curves for incorrect-answer detection.
3. Consider an `N=10` run for smoother SE estimates.
4. Begin SEP target generation using the Level 4 artifacts.
