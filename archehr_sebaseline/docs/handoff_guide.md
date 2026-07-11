# ArchEHR-QA UQ Code Handoff Guide

Last updated: 2026-07-03

Last updated: 2026-07-09

The active project is `archehr_sebaseline`. The package now contains:

- historical PubMedQA Level 3/4/5 smoke and pilot paths,
- implemented ArchEHR-QA grounded long-form SE pipeline,
- lightweight uncertainty/evaluation utilities,
- documentation for the current dataset pivot.

Current research direction: test BioASQ Task B as the main replacement dataset
while continuing to compare credible candidates as Simpson advised. ArchEHR-QA
remains useful as an engineering diagnostic, but it is not the final evaluation
or SEP training dataset because its test key does not include gold evidence
labels or answer-quality labels.

See:

```text
docs/dataset_pivot_status.md
docs/archehr_se_baseline_plan.md
docs/archehr_evaluation_architecture.md
```

## Current Stage

ArchEHR-QA SE engineering baseline is complete, but dataset pivot is required.

Latest ArchEHR-QA test run:

```text
dataset: archehr_qa
split: test
examples: 100
generations: 1000
model_name: google/gemma-3-12b-it
num_samples: 10
clustering_method: nli_bidirectional_entailment
parse_status: json for 1000/1000 generations
```

Important limitation:

```text
test key fields: case_id, clinician_answer
missing: sentence relevance labels, citation labels, answer-quality labels
```

Therefore ArchEHR-QA test supports reference-only diagnostics but not final
factuality/citation-quality evaluation.

Historical Level 4 is complete as a pilot baseline:

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
- `src/archehr_sebaseline/pipeline_archehr_se.py`: ArchEHR-QA grounded long-form SE pipeline.
- `src/archehr_sebaseline/prompting.py`: ArchEHR-QA structured citation prompt construction.
- `src/archehr_sebaseline/answer_parsing.py`: JSON/fallback parsing for cited answers.
- `src/archehr_sebaseline/citation_uq.py`: citation-set uncertainty metrics.
- `src/archehr_sebaseline/evaluation/archehr_answer_quality.py`: lightweight ArchEHR-QA dev/reference-only evaluator.
- `src/archehr_sebaseline/evaluation/uncertainty_metrics.py`: AUROC, ECE, rejection curves, and SVG plots.

## Entry Points

- `scripts/run_level3.py`
- `scripts/run_level3.sbatch`
- `scripts/run_level4.py`
- `scripts/run_level4.sbatch`
- `scripts/check_level4_outputs.py`
- `scripts/run_archehr_se.py`
- `scripts/run_archehr_se.sbatch`
- `scripts/evaluate_archehr_se.py`

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

1. Test the implemented BioASQ adapter on a small public subset.
2. Validate or add BioASQ-specific quality evaluation while preserving the
   common artifact layout.
3. Compare deterministic metrics with an LLM-based judge on a reviewed subset.
4. Re-run SE with `num_samples=10` after the BioASQ pilot is stable.
5. Use BioASQ labels for AUROC/ECE/rejection curves where appropriate.
6. Defer SEP hidden-state probes until the new target is stable.
