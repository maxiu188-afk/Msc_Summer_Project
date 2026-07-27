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
