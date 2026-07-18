# ArchEHR-QA UQ Code Handoff Guide

Last updated: 2026-07-17

The active project is `archehr_sebaseline`. The package now contains:

- historical PubMedQA Level 3/4/5 smoke and pilot paths,
- implemented ArchEHR-QA grounded long-form SE pipeline,
- lightweight uncertainty/evaluation utilities,
- documentation for the current dataset pivot.

Current research direction: validate BioASQ Task B answer-quality supervision
for SE, while continuing to compare credible candidates as Simpson advised.
ArchEHR-QA remains useful as an engineering diagnostic, but it is not the final
evaluation or SEP training dataset because its test key does not include gold
evidence labels or answer-quality labels.

See:

```text
docs/dataset_pivot_status.md
docs/bioasq_isambard_results_20260715.md
docs/bioasq_runpod_results_20260713.md
docs/experiment_runtime_log.md
docs/archehr_se_baseline_plan.md
docs/archehr_evaluation_architecture.md
```

## Current Stage

The primary Isambard baseline and its matched seed repeat are complete:

```text
job 5654721: BioASQ training summary 100x10, seed 31, 03:43:51
job 5660346: BioASQ training summary 100x10, seed 47, 03:18:31
model: google/gemma-3-12b-it
clustering: NLI bidirectional entailment
health checks: PASS / PASS
```

Answer quality is highly stable across seeds, but discrete-SE AUROC drops from
0.687 to 0.609 and the repeat interval includes chance. Job 5660345 completed a
fixed 30-question Qwen judge, but the judge ceiling effect produced zero
low-quality labels. The baseline is complete; do not present SE as a robustly
superior UQ method or the Qwen judge as external validation.

The temperature-1.0 evidence pair and matched no-evidence direct pair are
complete. Their high-temperature samples use `top_p=0.9` and `top_k=50`. The
Semantic Entropy protocol is now corrected: those ten samples calculate SE and
token UQ, while a separate `T=0.1` answer is the quality outcome. Jobs
5692776/5692777 (evidence) and 5692779/5692780 (direct) completed the four
100-answer backfills; jobs 5696576--5696579 completed the corresponding
protocol-correct self-report UQ.

Claude Sonnet 5 runs locally, not on Isambard, as a three-class
(`good`/`partial`/`poor`) text comparison of only the low-temperature answer
against BioASQ ideal answers. Direct answers use `poor` as low quality;
evidence-conditioned answers use `partial + poor`. The primary two-seed result
is P(True) AUROC 0.921 for direct answers and predictive entropy/normalized
NLL AUROC 0.642 for evidence-conditioned answers. The historical 4,000
high-temperature Claude labels are exploratory only and must not replace this
target.

The active reference evaluation now separates answer-reference coverage from
document-level cited-evidence overlap and uses their geometric combination when
BioASQ standard documents are available. Historical ROUGE-only AUROCs must not
be compared numerically with the new thresholded target before manual
calibration.

The first grounded BioASQ batch is complete and archived at:

```text
server_results/bioasq_se_runpod_required_results_20260713.tar.gz
```

It contains Golden summary 80x10, Golden factoid 50x10, Golden list 50x10, and
a training-summary 50x10 repeat, all with Gemma 3 12B and NLI clustering. All
four passed structural health checks. Interpret the local quality/SE AUROCs by
question type only; their small samples and provisional evaluator do not support
a pooled score. Runpod was temporary; future jobs run on Isambard.

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
- `src/archehr_sebaseline/evaluation/bioasq_quality.py`: BioASQ-aligned local quality and UQ evaluation.
- `src/archehr_sebaseline/evaluation/bioasq_llm_judge.py`: fixed-subset judge validation utilities.
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
- `scripts/evaluate_bioasq_quality.py`
- `scripts/run_bioasq_llm_judge.py`
- `scripts/run_bioasq_isambard.sbatch`
- `scripts/generate_bioasq_best_generations.py`
- `scripts/run_bioasq_best_isambard.sbatch`
- `scripts/run_bioasq_claude_judge.py`
- `scripts/evaluate_bioasq_claude_judge.py`

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

1. Review semantic-entropy failure cases and clustering/sample sensitivity using
   both seeds and both evidence conditions.
2. Report both seed-level values, label prevalence, and rejection curves rather
   than pooling the two evidence conditions.
3. Decide on SEP hidden-state probes only after the corrected quality target is
   stable and reviewed.
