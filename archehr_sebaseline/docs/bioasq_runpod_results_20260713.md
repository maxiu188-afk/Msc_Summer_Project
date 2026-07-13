# BioASQ Grounded SE Batch Results (2026-07-13)

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

| Run | Split / type | Questions x samples | Low-quality examples | Mean quality | Discrete SE AUROC | Weighted SE AUROC | Best token-derived AUROC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier primary | training13b / summary | 100 x 10 | 32 (32%) | 0.224 | 0.680 | 0.682 | 0.704 (NLL/PE) |
| Golden summary | golden13b / summary | 80 x 10 | 10 (12.5%) | 0.309 | 0.471 | 0.473 | 0.590 (token entropy) |
| Golden factoid | golden13b / factoid | 50 x 10 | 15 (30%) | 0.680 | 0.676 | 0.675 | 0.602 (token entropy) |
| Golden list | golden13b / list | 50 x 10 | 12 (24%) | 0.309 | 0.606 | 0.603 | 0.484 (all token-derived scores) |
| Summary repeat | training13b / summary | 50 x 10 | 18 (36%) | 0.211 | 0.624 | 0.622 | 0.693 (token entropy) |

All five runs passed their structural health checks: their required artifacts
are present, counts agree, CUDA generation and NLI clustering are recorded,
token-score fields are populated, and no non-finite score markers were found.

## Interpretation

1. The experiment is a **gold-snippet-conditioned** grounded QA setting. The
   model receives the text of BioASQ's provided snippets; document URLs are
   metadata and full paper text is not retrieved. Therefore this measures
   answer-generation uncertainty conditional on gold evidence, not retrieval
   uncertainty.
2. The results contain enough quality variation for SE filtering: the local
   low-quality rate ranges from 12.5% to 36%. Do not alter the generation
   prompt merely to make answers lexically stronger; that would work against
   the aim of measuring whether SE identifies weaker answers.
3. The SE signal is not stable across these small, heterogeneous slices. It is
   near chance on Golden summary, helpful on factoid and list, and moderately
   useful on the training-summary samples. The prior 100-question training
   summary also shows a useful but not dominant SE signal. No pooled AUROC or
   cross-type model ranking is justified.
4. The `lightweight_bioasq_v1` evaluator approximates metric families only. It
   is not the official BioASQ evaluation service: summary scoring is lexical,
   there is no manual ideal-answer review, synonym handling, or official
   ranked-answer protocol. Its `0.15` threshold is an operational SE-analysis
   cut-off, not a BioASQ pass mark.

## Isambard follow-up

1. Keep the current gold-snippet prompt and artifact schema fixed.
2. On Isambard, compare the deterministic quality label with a fixed-prompt,
   versioned LLM judge or a manually reviewed subset; save all judgments.
3. Run independent seed repeats of the summary, factoid, and list subsets to
   quantify variance before selecting a primary SEP target.
4. Start hidden-state SEP work only after this answer-quality target is stable.
