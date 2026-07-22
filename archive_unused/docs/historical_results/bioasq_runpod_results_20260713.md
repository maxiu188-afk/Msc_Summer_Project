# BioASQ Grounded SE Batch Results (2026-07-13)

This document is the historical RunPod batch record. The matched Isambard
100x10 baseline, seed repeat, and judge follow-up are documented in
`bioasq_isambard_results_20260715.md`.

## Scope and provenance

This note records the completed temporary Runpod A40 batch before future work
returns to Isambard. Every run used `google/gemma-3-12b-it`, bfloat16 CUDA
generation, ten sampled answers per question, the grounded BioASQ prompt with
the provided gold PubMed snippets, token-score UQ, and
`microsoft/deberta-v2-xlarge-mnli` bidirectional-entailment clustering.

The raw required-batch evidence is
`server_results/bioasq_se_runpod_required_results_20260713.tar.gz`. It contains
the four run directories, health checks, local quality/SE outputs, and terminal
logs. The earlier training-summary 100x10 run remains extracted at
`server_results/runpod_bioasq_summary_gemma3_12b_100x10/`.

## Health and local quality summary

The source archive retains the original v1 post-processing outputs. The table
below is a reproducible v2 re-analysis of the same completed generations; it
does not regenerate answers or alter the prompt. `Low quality` is the legacy
fixed threshold (`quality < 0.15`). The two AUROC columns use that target for
backward comparison. The relative-target AUROCs, continuous risk associations,
and bootstrap intervals are written by v2 beside every reprocessed run.

| Run | Split / type | Questions x samples | Low-quality examples | Mean quality | Discrete SE AUROC | Weighted SE AUROC | Best AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier primary | training13b / summary | 100 x 10 | 32 (32%) | 0.224 | 0.680 | 0.682 | 0.704 (predictive entropy) |
| Golden summary | golden13b / summary | 80 x 10 | 10 (12.5%) | 0.309 | 0.471 | 0.473 | 0.590 (token entropy) |
| Golden factoid | golden13b / factoid | 50 x 10 | 26 (52%) | 0.460 | 0.618 | 0.618 | 0.641 (token entropy) |
| Golden list | golden13b / list | 50 x 10 | 12 (24%) | 0.323 | 0.606 | 0.603 | 0.606 (discrete SE) |
| Summary repeat | training13b / summary | 50 x 10 | 18 (36%) | 0.211 | 0.624 | 0.622 | 0.693 (token entropy) |

All five runs passed their structural health checks: their required artifacts
are present, counts agree, CUDA generation and NLI clustering are recorded,
token-score fields are populated, and no non-finite score markers were found.

## Reference-mode SE evaluation (re-run 2026-07-14)

The complete reference-mode evaluator was re-run over all five completed
outputs with 1,000 deterministic bootstrap resamples. This pass does **not**
load a new NLI model: quality comes from BioASQ reference/exact answers,
type-specific parsing, and citation-ID diagnostics. Each SE score is examined
against three complementary targets: the historical fixed threshold, the
within-type bottom-quality target, and continuous risk (`1 - quality`).

| Run | Discrete SE fixed AUROC [95% bootstrap CI] | Discrete SE relative AUROC [CI] | Discrete SE risk Spearman [CI] | Strongest relative-target signal |
| --- | ---: | ---: | ---: | --- |
| Earlier primary summary (100) | 0.680 [0.569, 0.793] | 0.684 [0.561, 0.798] | 0.386 [0.212, 0.548] | predictive entropy / NLL: 0.740 |
| Golden summary (80) | 0.471 [0.290, 0.668] | 0.570 [0.443, 0.707] | 0.383 [0.171, 0.573] | predictive entropy / NLL: 0.657 |
| Golden factoid (50) | 0.618 [0.475, 0.750] | 0.618 [0.487, 0.744] | 0.308 [0.071, 0.534] | token entropy: 0.641 |
| Golden list (50) | 0.606 [0.415, 0.783] | 0.573 [0.396, 0.734] | 0.304 [0.035, 0.546] | discrete SE: 0.573 |
| Summary repeat (50) | 0.624 [0.452, 0.777] | 0.637 [0.459, 0.806] | 0.393 [0.121, 0.625] | token entropy: 0.715 |

This confirms an SE signal but not a stable method ranking. In particular,
Golden summary has only ten examples below the fixed threshold, so its fixed
AUROC is highly threshold-sensitive. Its positive continuous-risk association
and relative-target AUROC show that the weak fixed result is not enough to
conclude that uncertainty has no relationship with quality. Conversely, the
wide intervals on all 50-question slices mean that apparent differences among
discrete SE, weighted SE, predictive entropy, NLL, and token entropy are not
yet reliable model-selection evidence.

## Simple UQ baseline comparison (2026-07-14)

Three requested non-SE baselines were evaluated from the already recorded token
scores; no model was loaded. `Avg token log-prob` is reported as its negative
so that higher values mean more uncertainty. It is mathematically identical to
the existing length-normalized NLL, so it is kept for terminology comparison
only rather than counted as independent evidence. `G-NLL / sequence NLL` is
the unnormalized negative sequence log-probability.

| Run | Discrete SE (relative AUROC) | Avg token log-prob, negative | Avg token entropy | G-NLL / sequence NLL |
| --- | ---: | ---: | ---: | ---: |
| Earlier primary summary (100) | 0.684 | **0.740** | 0.720 | 0.676 |
| Golden summary (80) | 0.570 | 0.657 | 0.649 | **0.675** |
| Golden factoid (50) | 0.618 | 0.628 | 0.641 | **0.723** |
| Golden list (50) | **0.573** | 0.492 | 0.500 | 0.506 |
| Summary repeat (50) | 0.637 | 0.690 | **0.715** | 0.665 |

The apparent factoid advantage of sequence NLL must be interpreted cautiously:
its Spearman correlation with mean answer length is 0.592--0.803 across the
five slices. This makes raw sequence NLL a necessary simple baseline but not a
length-fair primary method. The length-normalized token log-probability and
mean token entropy are stronger fair comparisons to Semantic Entropy.

The remaining two requested baselines are implemented but not yet measured:
verbalized confidence (`1 - reported confidence`) and P(True)
(`1 - P(True | question, evidence, answer)`). They require one extra Gemma
post-processing session over the existing generated answers, not new answer
generation or NLI. Their outputs will be auto-discovered by the BioASQ
evaluator after `uq_baselines/self_report_examples.csv` is written.

## Interpretation

1. The experiment is a **gold-snippet-conditioned** grounded QA setting. The
   model receives the text of BioASQ's provided snippets; document URLs are
   metadata and full paper text is not retrieved. Therefore this measures
   answer-generation uncertainty conditional on gold evidence, not retrieval
   uncertainty.
2. The results contain enough quality variation for SE filtering: the v2
   fixed-threshold low-quality rate ranges from 12.5% to 52%. The factoid
   increase is an evaluation correction: v2 measures whether the answer starts
   with the exact answer, rather than treating a later mention as fully correct.
   Do not alter the generation
   prompt merely to make answers lexically stronger; that would work against
   the aim of measuring whether SE identifies weaker answers.
3. The SE signal is not stable across these small, heterogeneous slices. It is
   near chance on the fixed Golden-summary target, helpful on factoid and list,
   and moderately useful on the training-summary samples. V2 makes the evidence
   more robust by also reporting per-type bottom-quality AUROC, Spearman
   correlation with continuous risk (`1 - quality`), a coverage-risk curve, and
   stratified bootstrap intervals. For example, discrete-SE risk correlation is
   positive for all five slices (0.304--0.393), but intervals remain wide on the
   50-question slices. No pooled AUROC or cross-type model ranking is justified.
4. The `lightweight_bioasq_v2` evaluator approximates BioASQ metric families
   only. Summary scoring is lexical ROUGE-2/ROUGE-SU4; factoid uses strict,
   answer-first, and lenient diagnostics; list uses normalized set P/R/F1; and
   cited snippet IDs are checked against the supplied evidence IDs. It is not
   the official BioASQ evaluation service: there is no manual ideal-answer
   review, synonym resource, or official ranked-answer protocol. Neither the
   fixed `0.15` threshold nor the bottom-quality quantile is a BioASQ pass mark.

## Isambard follow-up

1. Keep the current gold-snippet prompt and artifact schema fixed.
2. Run the new grounded v2 post-processing over the completed outputs with the
   existing open NLI model on CUDA. It scores reference agreement, snippet
   entailment, contradiction, and cited-claim support separately; do not mix
   these future grounded scores with the reference-only table above.
3. Re-run reference-mode v2 post-processing for future Level 4 outputs with
   1,000 bootstrap draws; this is CPU-only and can run after generation.
4. On Isambard, compare the deterministic quality label with a fixed-prompt,
   versioned LLM judge or a manually reviewed subset; save all judgments.
5. Run independent seed repeats of the summary, factoid, and list subsets to
   quantify variance before selecting a primary SEP target.
6. Start hidden-state SEP work only after this answer-quality target is stable.
