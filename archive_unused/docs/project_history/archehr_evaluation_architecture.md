# ArchEHR-QA Evaluation Architecture

This document describes the lightweight evaluation stage for the ArchEHR-QA Semantic Entropy baseline.

## Current Status

This evaluator is maintained for diagnostics and for dev-style runs where gold
evidence labels are available. It should not be treated as the final project
benchmark for ArchEHR-QA test, because the released test key contains clinician
answers only and no sentence relevance, citation, or answer-quality labels.

## Purpose

The generation pipeline estimates uncertainty from sampled answers. The evaluation pipeline asks whether those uncertainty scores identify weak answers.

This stage is intentionally lightweight for now:

- Factuality proxy: citation overlap with gold relevant evidence.
- Relevance proxy: lexical overlap with the clinician answer.
- SE evaluation: AUROC, ECE, and rejection curves for uncertainty scores.

The design goal is to keep the pipeline runnable and maintainable, while leaving clear replacement points for stronger factuality and relevance metrics later.

## Inputs

The evaluator consumes an existing `run_archehr_se.py` output directory:

```text
examples.jsonl
cleaned_generations.jsonl
answer_se_scores.csv
generation_uq.csv
citation_uq.csv
```

It also consumes the ArchEHR-QA key JSON:

```text
archehr-qa_key.json
```

The key loader accepts several schema variants. It looks for:

- example id fields: `id`, `case_id`, `example_id`, `qid`, `question_id`
- answer fields: `clinician_answer`, `gold_answer`, `reference_answer`, `answer`, `answers`
- sentence relevance fields: `answers`, `sentence_relevance`, `sentence_labels`, `evidence_relevance`, `relevance`, `labels`
- explicit sentence id fields: `essential_sentence_ids`, `supplementary_sentence_ids`, `gold_relevant_sentence_ids`

The official ArchEHR-QA dev key uses `answers` as a list of sentence relevance labels, not as answer text. The loader also falls back to citations embedded in the clinician answer, such as `[2, 5]`, when no explicit essential labels are present.

## Modules

```text
scripts/evaluate_archehr_se.py
```

CLI wrapper. It parses paths and options, then calls the evaluation package.

```text
src/archehr_sebaseline/evaluation/archehr_answer_quality.py
```

ArchEHR-specific quality evaluation:

- loads the key JSON
- computes generation-level citation/text quality
- aggregates generation quality to example quality
- joins quality labels with uncertainty scores
- writes tables, summary JSON, Markdown report, and plots

```text
src/archehr_sebaseline/evaluation/uncertainty_metrics.py
```

Dataset-agnostic uncertainty metrics:

- AUROC
- ECE and reliability bins
- rejection curves
- dependency-free SVG plots

## Current Quality Definition

For each generated answer:

```text
strict citation F1 = F1(predicted citations, essential sentence ids)
lenient citation F1 = F1(predicted citations, essential + supplementary sentence ids)
citation_score = max(strict citation F1, lenient citation F1)
token recall = fraction of clinician-answer tokens covered by the model answer
ROUGE-L recall = longest-common-subsequence recall against the clinician answer
answer_coverage = 0.5 * token recall + 0.5 * ROUGE-L recall
lexical_similarity = 0.5 * token F1 + 0.5 * ROUGE-L F1
relevance_score = 0.75 * answer_coverage + 0.25 * lexical_similarity
quality_score = 0.75 * citation_score + 0.25 * relevance_score
```

The default example-level target is `sample0`, because it corresponds to evaluating a single answer from the model. The CLI also supports `--quality_target mean`, which averages across sampled answers.

```text
is_low_quality = quality_score < quality_threshold
```

The default threshold is `0.4`. This is intentionally softer than the first lightweight version because concise answers should not be marked low quality only because they are not lexically identical to the clinician answer.

When a key file has clinician reference answers but no gold evidence labels, the evaluator switches to reference-only mode:

```text
quality_score = relevance_score
is_low_quality = quality_score < reference_only_threshold
```

The default `reference_only_threshold` is `0.2`. This keeps test-set evaluation usable when the released key does not contain sentence relevance labels.

## SE Evaluation

The evaluator joins `is_low_quality` with these uncertainty scores:

```text
normalized_semantic_entropy
semantic_entropy
num_clusters
normalized_citation_set_entropy
mean_token_entropy
mean_normalized_nll
```

Higher score is treated as higher uncertainty.

Outputs:

```text
se_auroc.csv
se_ece.csv
se_reliability_bins.csv
se_rejection_curve.csv
auroc_bar.svg
rejection_curve.svg
reliability_diagram.svg
```

## Extension Points

To make evaluation stronger later without rewriting the pipeline:

1. Replace `evaluate_generation_quality()` with a richer factuality/relevance scorer.
2. Keep `answer_quality_generations.csv` and `answer_quality_examples.csv` field-compatible where possible.
3. Keep `is_low_quality` as the binary target consumed by `uncertainty_metrics.py`.
4. Add new uncertainty score columns to `SE_EVAL_EXAMPLE_FIELDS` and pass them through `DEFAULT_SCORE_NAMES`.

Good next upgrades:

- statement-level factuality using cited sentence entailment
- reference-answer semantic similarity using an open biomedical encoder or NLI model
- separate factuality and relevance AUROC targets
- bootstrap confidence intervals for AUROC and rejection curves
