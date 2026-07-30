# BioASQ Summary Answer-Length Intervention

Last updated: 2026-07-28

## Status

The paired current-versus-short BioASQ summary experiment is complete. The
short prompt reliably reduced answer length, but it did not improve Semantic
Entropy relative to blind P(True). The primary paired length-effect confidence
interval includes zero.

The collection used all 123 valid-Claude-labelled summary questions from the
Phase-2 efficiency cohort. Correctness and UQ ranking use the 121 question IDs
with valid Claude labels in both conditions. Two short answers remained blank
after the initial Claude batch and two same-configuration retries; they are
excluded only from correctness-dependent paired analyses, not from length or
generation diagnostics.

## Frozen intervention

The current condition retained:

```text
Write one concise biomedical paragraph that directly answers the question.
```

The short condition added only:

```text
Keep the answer to one or two brief but complete sentences.
Include only information needed to answer the question, without extra background.
```

Both conditions used the same `google/gemma-3-12b-it` snapshot, seed 31,
`max_new_tokens=192`, `top_p=0.9`, `top_k=50`, one `T=0.1` main answer, ten
`T=1.0` samples, blind-P(True) prompt, and
`pritamdeka/PubMedBERT-MNLI-MedNLI` clustering. No fixed word/token limit or
new decoding truncation was introduced, and no answer reached 192 generated
tokens.

The accepted current main answers, current Claude labels, and current blind
P(True) scores were reused. The completed efficiency benchmark had retained
aggregate current-condition UQ scores but not the sampled answer texts or full
clusters required for length and cluster-size diagnostics, so its ten current
samples were deterministically regenerated. Their normalized NLL, discrete
SE, cluster count, and blind P(True) reproduce the accepted efficiency rows
exactly, with maximum absolute difference zero across all 123 questions.

## Execution and health

| Stage | Job | State | Elapsed | Result |
| --- | ---: | --- | ---: | --- |
| Three-question smoke | `5781225` | `COMPLETED 0:0` | `00:06:28` | health check passed |
| Full paired collection | `5781246` | `COMPLETED 0:0` | `03:59:14` | health check passed |

The full job used `afterok:5781225`. Each condition contains:

- 123 examples, prompts, and main answers;
- 1,230 high-temperature sampled answers;
- 123 NLI cluster records whose cluster sizes each sum to ten; and
- 123 condition-score rows.

The short-answer Claude batches were:

```text
initial: msgbatch_01GyM5xLBcp7oyMV8BSUZ8rc, 123/123 API successes
retry 1: msgbatch_019upAX3uH2X1joZvufjNZe2, 3 requests
retry 2: msgbatch_01NAapx7qM5MBw54LCZ7hBdf, 3 requests
```

All used `claude-sonnet-5`, low effort, and 32 output tokens, matching Phase 2.
The final short labels are 56 correct, 65 incorrect, and two blank/invalid.
On the paired 121 rows, both conditions have 56 correct and 65 incorrect.

## Length and correctness

Length statistics use all 123 assigned questions:

| Statistic | Current | Short | Short minus current |
| --- | ---: | ---: | ---: |
| Main-answer generated tokens, mean | 111.37 | 69.05 | -42.33 |
| Main-answer words, mean | 82.88 | 52.11 | -30.77 |
| Main-answer sentences, mean | 3.25 | 2.00 | -1.25 |
| Ten-sample words, mean | 83.40 | 51.98 | -31.42 |
| Ten-sample sentences, mean | 3.29 | 2.00 | -1.29 |
| Main-answer one/two-sentence compliance | 7/123 | 123/123 | +116 |
| Sample one/two-sentence compliance | 74/1,230 | 1,230/1,230 | +1,156 |

The paired-bootstrap 95% CI for the main-answer word difference is
`[-32.82, -28.72]`; for sentence difference it is `[-1.36, -1.15]`. The
intervention therefore changed realised length clearly and consistently.

On the 121 valid-label pairs:

| Paired correctness outcome | Count |
| --- | ---: |
| Correct in both | 48 |
| Current only correct | 8 |
| Short only correct | 8 |
| Incorrect in both | 57 |

Both conditions have accuracy `56/121 = 46.28%`. The paired accuracy
difference is `0.0000`, with bootstrap 95% CI `[-0.0661, 0.0661]`.

## UQ ranking

`incorrect=1` is the positive risk class. AUROC/AP use the same 121 paired
questions:

| Method | Current AUROC / AP | Short AUROC / AP | AUROC change |
| --- | ---: | ---: | ---: |
| Blind P(True) | **0.7948 / 0.8026** | **0.8040 / 0.8212** | +0.0092 |
| Ten-sample normalized NLL | 0.7533 / 0.7884 | 0.7511 / 0.7637 | -0.0022 |
| Cluster count | 0.5805 / 0.6061 | 0.5451 / 0.5763 | -0.0354 |
| Discrete Semantic Entropy | 0.5782 / 0.6018 | 0.5468 / 0.5778 | -0.0313 |

Error prevalence is identical in the paired conditions, but AP remains
secondary; AUROC is the primary cross-condition ranking comparison.

The pre-declared relative quantities are:

```text
D_current = 0.7948 - 0.5782 = 0.2166
D_short   = 0.8040 - 0.5468 = 0.2571
length effect = D_short - D_current = +0.0405
```

| Quantity | Estimate | Paired-bootstrap 95% CI |
| --- | ---: | ---: |
| `D_current` | +0.2166 | `[0.1119, 0.3158]` |
| `D_short` | +0.2571 | `[0.1542, 0.3562]` |
| Length effect | +0.0405 | `[-0.0713, 0.1531]` |

Blind P(True) clearly exceeds discrete SE within both prompt conditions. The
length-effect interval includes zero, so this experiment does not establish
that shortening changes their relative ranking performance. The point estimate
moves in the opposite direction from an SE-recovery hypothesis: blind
P(True)'s advantage is slightly larger under the short prompt. This is not
statistically resolved.

## Sentence-count repair

The first derived score tables used a naive period-plus-space sentence
boundary, which incorrectly split genus abbreviations such as `C. elegans` and
`C. glabrata`. No generation or UQ value was affected. The original score
tables are retained under:

```text
archive_sentence_count_pre_fix/
```

`sentence_count_repair.json` records the before/archive/after hashes. The
repair rewrote only word/sentence/compliance fields from saved answers.

## Interpretation and completed follow-up

The controlled intervention succeeded at changing answer length without
changing paired accuracy, but shorter answers did not recover whole-answer
Semantic Entropy. Answer length alone is therefore not supported as the main
explanation for the SE-versus-P(True) gap in this cohort.

The bounded follow-up reused these fixed generations to compare the accepted
PubMedBERT NLI clusters with a frozen Claude semantic-equivalence rubric. On
93/96 complete cases, Claude-clustered SE reached 0.781 AUROC versus 0.595 for
NLI-SE. The completed stratified 24-pair review agrees with Claude clustering
on 19/24 pairs versus 9/24 for original NLI, strongly implicating long-answer
NLI errors. The follow-up remains a mechanism diagnostic rather than a
deployment method. See `summary_clustering_diagnostic.md`.

## Reproducibility

Collection and repair:

```text
scripts/run_summary_length_intervention.py
scripts/run_summary_length_intervention_isambard.sbatch
scripts/repair_summary_length_sentence_counts.py
```

Paired analysis:

```text
analysis/run_summary_length_comparison.py
analysis_outputs/bioasq_summary_length_comparison_20260726/
```

The analysis used 20,000 paired question-ID bootstrap resamples with seed
`20260725`. `summary.json` records all input SHA-256 values.
