# Gemma 3 model-scale experiment

Last updated: 2026-07-27

## Status

The first added scale point is frozen as `google/gemma-3-4b-it`, seed 31.
Both Isambard jobs completed successfully:

| Stage | Job | Selection | Dependency | Last verified state |
| --- | ---: | --- | --- | --- |
| Smoke | `5785064` | 6 questions, 2 per type | none | `COMPLETED 0:0`, 00:03:37 |
| Full | `5785065` | exact Phase-1 1,000 questions | `afterok:5785064` | `COMPLETED 0:0`, 05:46:48 |

The full output passed the Level-4 health check with 1,000 examples, 10,000
high-temperature generations, 1,000 clusters, complete token/UQ fields, no
NaN/Infinity markers, and CUDA/bfloat16 execution. Exact postflight alignment
against the accepted 12B seed-31 cohort passed: 480 factoid, 320 list, 200
summary, with identical records for all IDs.

Claude batch `msgbatch_01AbxKWPN4pdgYoh3CF2V42G` completed with 1,000/1,000
API successes and 986 valid binary labels. The single protocol-matched
14-response retry `msgbatch_01NtMndSj62sKhvhMrkQbopS` raised this to 997 valid
labels: 796 incorrect and 201 correct. The remaining three invalid responses
are excluded without another retry.

## Scientific question

Test whether model capability changes the relative correctness-ranking
performance of the two original UQ methods:

```text
blind P(True) versus discrete Semantic Entropy
```

This is not a new Probe experiment. No hidden-state Probe is trained or
transferred across model sizes.

## Phase-1 alignment

The 4B run must use exactly the accepted Phase-1 1,000-question cohort:

| Field | Frozen value |
| --- | --- |
| Dataset | BioASQ training13b, no evidence |
| Type quotas | factoid 480, list 320, summary 200 |
| Selection seed | `20260718` |
| Reference model run | Gemma 3 12B, seed 31 |
| Reference `examples.jsonl` SHA-256 | `4c40280a6ab234d204ad1f60e9c7d06a588f8387ee5ee21a4d2c1028e907b7d0` |

The Slurm entrypoint validates the source-derived selection before loading the
model and validates the saved output again after collection. The full run
requires exact ID and record equality. The smoke uses the first two
deterministically shuffled questions from each Phase-1 type and requires those
six records to be an exact subset of the same reference.

## Frozen generation and UQ settings

Only the generator checkpoint changes from the accepted seed-31 Phase-1 run:

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-4b-it` |
| Seed | 31 |
| Main answer | one sample, temperature 0.1 |
| SE samples | ten per question, temperature 1.0 |
| Sampling | `top_p=0.9`, `top_k=50` |
| Maximum new tokens | 192 |
| Maximum input tokens | 4096 |
| Precision/device | bfloat16 / CUDA |
| NLI | `pritamdeka/PubMedBERT-MNLI-MedNLI` |
| Clustering | Phase-1 type-specific rules |
| Prompt | accepted Phase-1 no-evidence, type-specific prompt |
| Other UQ | saved token/NLL fields, verbal confidence, blind P(True) |

Claude binary correctness judging remains offline post-processing after the
full artifacts pass health and alignment checks.

## Validated generation context

These label-free values are descriptive context, not UQ correctness results:

| Type | 4B main words | 12B main words | 4B sampled tokens | 12B sampled tokens |
| --- | ---: | ---: | ---: | ---: |
| Overall | 21.04 | 22.34 | 35.58 | 37.41 |
| Factoid | 2.39 | 3.32 | 6.86 | 8.09 |
| List | 11.58 | 13.21 | 32.71 | 35.30 |
| Summary | 80.95 | 82.61 | 109.10 | 111.12 |

The 4B full run completed its internal pipeline in 05:46:38, versus 08:00:08
for the accepted 12B seed-31 computation on the same Isambard environment
family. This is a run-level observation, not a controlled throughput benchmark.

## Submission gate

The smoke runs six questions, two per type, with all ten SE samples and the
full 192-token generation limit. The 1,000-question job must be submitted with:

```text
afterok:<smoke_job_id>
```

It may enter the queue immediately but cannot start unless the smoke exits
successfully. No recurring monitor is started.

## Completed comparison

The paired analysis uses the 990 exactly aligned questions with valid labels
for both models. `incorrect` is the positive class. Because error prevalence
differs substantially, AUROC is the primary cross-model ranking metric; AP is
retained in the artifacts but is not compared directly across models.

| Scope | Model | Accuracy | Blind P(True) AUROC | Discrete-SE AUROC | Normalized-NLL AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| Overall | 4B | 0.203 | 0.758 | 0.768 | 0.755 |
| Overall | 12B | 0.332 | 0.811 | 0.764 | 0.729 |
| Factoid | 4B | 0.235 | 0.773 | 0.739 | 0.720 |
| Factoid | 12B | 0.397 | 0.794 | 0.721 | 0.738 |
| List | 4B | 0.107 | 0.764 | 0.832 | 0.870 |
| List | 12B | 0.155 | 0.845 | 0.845 | 0.792 |
| Summary | 4B | 0.279 | 0.648 | 0.643 | 0.794 |
| Summary | 12B | 0.462 | 0.808 | 0.565 | 0.753 |

The primary paired result is:

| Scope | P(True) minus SE, 4B | P(True) minus SE, 12B | 4B-minus-12B gap change |
| --- | ---: | ---: | ---: |
| Overall | -0.009 `[-0.051,+0.031]` | +0.047 `[+0.017,+0.076]` | -0.056 `[-0.103,-0.010]` |
| Factoid | +0.034 `[-0.027,+0.091]` | +0.073 `[+0.025,+0.121]` | -0.040 `[-0.114,+0.033]` |
| List | -0.068 `[-0.170,+0.030]` | +0.000 `[-0.070,+0.071]` | -0.068 `[-0.169,+0.028]` |
| Summary | +0.005 `[-0.085,+0.094]` | +0.243 `[+0.173,+0.313]` | -0.238 `[-0.347,-0.129]` |

Intervals are 20,000-resample paired bootstraps, seed `20260727`, stratified
by the joint 4B/12B correctness-label state.

Model capability changes the UQ operating regime. At 12B, blind P(True) has a
small but resolved overall advantage over SE. At 4B, that advantage
disappears; SE is statistically indistinguishable overall and has the higher
list point estimate. The overall change in the P(True)-minus-SE gap is itself
resolved. Summary drives the clearest difference: the 12B P(True) advantage
of +0.243 collapses to +0.005 at 4B.

This does not mean SE becomes a strong 4B summary method. Both P(True) and SE
are weak there (0.648 and 0.643 AUROC), while normalized NLL reaches 0.794.
The main scale signal is therefore loss of P(True)'s self-evaluation advantage
at lower model capability, not a general improvement in SE.

## Completed Gemma 3 1B staged extension

The next scale point is `google/gemma-3-1b-it`, seed 31. It tests whether the
P(True)-versus-SE operating regime continues to change below 4B. It is not a
Probe experiment: no hidden-state Probe is trained or transferred.

The first jobs were submitted on 2026-07-27. Smoke `5802163` passed its cohort
preflight and reached a CUDA node but failed `3:0` during model loading because
the 1B weights were not yet present in the shared Hugging Face cache. It
generated no answers. Dependent staged job `5802164` never ran and was
cancelled after its dependency became unsatisfiable.

All ten repository files for fixed revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752` were then downloaded on the login
node and passed `hf cache verify`. The replacement entrypoint checks that
snapshot before allocation work and uses `local_files_only` for both the
generator and NLI model.

Replacement smoke `5807823` still failed during construction. A metadata-only
reproduction then identified the code error: the shared generator treated
every Gemma 3 checkpoint as multimodal and called `AutoProcessor` plus
`Gemma3ForConditionalGeneration`. The 1B checkpoint is instead
`Gemma3TextConfig` / `Gemma3ForCausalLM` and deliberately has no image
processor. The generator now routes `model_type=gemma3_text` through
`AutoTokenizer` and `AutoModelForCausalLM`, while leaving the accepted
multimodal 4B/12B route unchanged. All 118 local unit tests passed, including a
dedicated route test. Dependent job `5807824` never ran and was cancelled.

| Stage | Job | Cohort | Dependency | Final status |
| --- | ---: | --- | --- | --- |
| Smoke | `5808905` | 2 factoid / 2 list / 2 summary | none | `COMPLETED 0:0`, 00:02:06 |
| Staged | `5808906` | 50 factoid / 50 list / 200 summary | `afterok:5808905` | `COMPLETED 0:0`, 02:06:35 |

Local and Isambard preflights reproduced cohort-ID SHA-256 values
`64ffbd1a...db01` for smoke and `f2b4a852...8108` for staged. Both final
outputs passed Level-4 health, exact reference-subset checks, and exact frozen
cohort checks. The staged output has 300 examples, 300 main generations, 3,000
high-temperature generations, 300 cluster/SE rows, complete self-report and
example-UQ rows, CUDA/bfloat16 provenance, and no NaN/Infinity markers.

The output was downloaded to
`outputs/bioasq_gemma3_1b_300x10_phase1_aligned_seed31_20260727`. Correctness
batch `msgbatch_01LCGoC6Mh4EBNKHAq3XEB5U` used the accepted
`claude-sonnet-5`, low-effort, 32-token binary main-answer protocol for all 300
questions. It ended with 300/300 API successes and 298 valid labels. The two
invalid rows are both summary responses stopped by `max_tokens`; one bounded
low-effort 64-token retry, `msgbatch_01AWNB4xt1h86wyfRdm6fq2F`, returned two
API successes and one additional valid label. The final result is 299/300
valid; the remaining invalid summary response again stopped at `max_tokens`
and is excluded without another retry.

The complete feasibility subsets already give 8/50 correct factoid answers
(16%, Wilson 95% CI 8.3%--28.5%) and 3/50 correct list answers (6%, Wilson 95%
CI 2.1%--16.2%). These are accuracy-gate results, not complete type-level UQ
experiments. Neither remaining full type cohort is recommended: list is at a
clear correctness floor, while factoid's limited additional scientific value
does not justify expanding beyond the completed summary scale test.

A full 1,000-question run is not authorized initially. The 4B model reached
only 20.3% overall accuracy, including 23.5% on factoid, 10.7% on list, and
27.9% on summary. A still smaller model may leave too few correct factoid/list
answers for a meaningful correctness-ranking comparison.

The staged protocol is:

1. Run a six-question smoke containing two exactly aligned questions per type,
   with the complete ten-sample SE, NLI, normalized-NLL, blind-P(True), and
   low-temperature main-answer path. Submit the staged job with a Slurm
   `afterok` dependency so it can continue only after a successful smoke exit
   and artifact health/alignment pass. Do not start continuous monitoring.
2. Before inspecting 1B outcomes, freeze all 200 aligned summary questions and
   deterministic 50-question subsets from the aligned 480 factoid and 320 list
   cohorts.
3. Preserve the Phase-1 seed-31 prompts, sampling settings, maximum lengths,
   type-specific NLI rules, correctness judge, and 4B/12B question mapping;
   only the generator checkpoint changes.
4. Use the 200-summary cohort for the formal aligned 1B/4B/12B comparison,
   reporting accuracy, answer length, blind-P(True), discrete-SE and
   normalized-NLL AUROC/AP, paired P(True)-minus-SE gaps, bootstrap intervals,
   and runtime.
5. For the 50 factoid and 50 list questions, report accuracy and the valid
   correct/incorrect label counts as feasibility evidence. Any UQ metrics from
   these subsets are preliminary and must not be presented as final type-level
   scale results.
6. Decide separately, after reviewing those accuracy results, whether either
   type retains enough correctness signal to justify collecting the remaining
   430 factoid or 270 list questions. Neither expansion is automatic.

The frozen local analysis entrypoint is
`analysis/run_gemma3_1b_staged_scale_comparison.py`. It enforces exact record
alignment, uses only summary questions with valid labels for all three models,
and runs a 20,000-resample joint-label-stratified paired bootstrap with seed
`20260727`. It writes factoid/list accuracy with valid correct/incorrect counts
and Wilson intervals separately; it does not automate the expansion decision.

## Completed 1B/4B/12B summary comparison

The formal cohort contains 196 summary questions with valid correctness labels
for all three models. `incorrect` is the positive class. AP is retained in the
artifacts but is not compared across models because error prevalence changes
sharply with scale.

| Model | Accuracy | Main words | Blind P(True) AUROC | Discrete-SE AUROC | Normalized-NLL AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1B | 0.122 | 66.14 | 0.501 | 0.597 | 0.789 |
| 4B | 0.281 | 80.94 | 0.652 | 0.644 | 0.796 |
| 12B | 0.459 | 82.52 | 0.807 | 0.564 | 0.756 |

| Comparison | P(True) minus SE or gap change | 95% CI |
| --- | ---: | ---: |
| 1B P(True) minus SE | -0.096 | [-0.244,+0.060] |
| 4B P(True) minus SE | +0.008 | [-0.081,+0.096] |
| 12B P(True) minus SE | +0.243 | [+0.171,+0.312] |
| 1B minus 4B gap change | -0.104 | [-0.267,+0.059] |
| 4B minus 12B gap change | -0.234 | [-0.341,-0.128] |
| 1B minus 12B gap change | -0.338 | [-0.508,-0.157] |

Intervals use 20,000 joint-correctness-label-stratified paired resamples with
seed `20260727`. The P(True)-minus-SE point estimate continues downward from
12B through 4B to 1B, and the total 1B-versus-12B change is resolved. However,
the incremental 1B-versus-4B interval crosses zero, so the experiment does not
establish that 1B differs from 4B beyond the already resolved 12B-to-4B shift.

At 1B, blind P(True) is essentially chance-ranked and SE has the better point
estimate, but their within-1B difference is not resolved. Normalized NLL is
clearly the strongest 1B point estimate. Together with 12.2% summary accuracy,
this supports a capability-collapse interpretation rather than claiming a
general SE improvement at small scale.

Final artifacts are under
`analysis_outputs/gemma3_1b_4b_12b_summary_scale_seed31_20260727`.

The 270M and 27B checkpoints remain outside the authorized experiment.
