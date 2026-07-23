# Phase-2 UQ Efficiency Benchmark

Last updated: 2026-07-23

## Status

The benchmark is submitted on Isambard but has not started. It has no measured
latency, GPU-time, throughput, token-count, or AUROC result yet.

| Stage | Slurm job | Selection | Dependency | Status at documentation time |
| --- | ---: | ---: | --- | --- |
| Smoke | `5761273` | 6 labelled test questions, balanced across factoid/list/summary | none | pending |
| Full | `5761275` | all 384 valid-labelled Phase-2 test questions | `afterok:5761273` | pending on smoke |

The full job cannot start after a failed smoke. Both jobs include the
Semantic-Entropy branch so that the smoke measures its actual sampling and NLI
cost before the full allocation runs.

## Question

The two selected linear Probes use the same Gemma 3 12B block-24/LT vector.
This experiment tests whether that shared hidden-state replay provides a useful
efficiency advantage over the best original Phase-1 UQ methods.

The benchmark measures **incremental UQ cost after the saved `T=0.1` main
answer already exists**. It never regenerates or changes the main answer. The
common Gemma load and original answer-generation cost are excluded from
steady-state per-question measurements because the model is already resident
when UQ is requested. Method-specific NLI model loading is recorded separately.

## Frozen evaluation set

The input is the Phase-2 `test` split only:

| Type | Valid Claude labels |
| --- | ---: |
| Factoid | 158 |
| List | 103 |
| Summary | 123 |
| **Total** | **384** |

There are 252 `incorrect` and 132 `correct` main answers. The same 384
questions are used for every full-run method and `incorrect=1` is the ranking
target. No Probe fitting, feature selection, threshold selection,
recalibration, or Claude call occurs in this benchmark.

## Selected comparators

The original UQ comparison is intentionally small. Phase 1 established blind
P(True) as the strongest overall method (AUROC `0.811/0.821`), with discrete
SE at `0.763/0.779`, cluster count at `0.765/0.781`, and normalized NLL at
`0.729/0.745`. Verbalized confidence (`0.696/0.695`) and weaker variants are
not rerun.

| Reported method | Incremental computation | Extra generated tokens |
| --- | --- | ---: |
| P(True)-Probe | one saved prompt+answer hidden-state replay, then its frozen CPU linear head | 0 |
| Accuracy-Probe | the same shared replay, then its frozen CPU linear head | 0 |
| Both Probes jointly | one shared replay and both frozen CPU heads | 0 |
| Blind P(True) | score the `True` and `False` continuations with two causal-LM calls | 0 |
| 10-sample normalized NLL | ten `T=1.0` generations with token scores | all sampled tokens |
| Discrete SE | the same ten generations plus biomedical NLI clustering | the same sampled tokens |
| Cluster count | the same ten generations and NLI clusters as discrete SE | the same sampled tokens |

The individual Probe rows each include the shared replay cost so they remain
valid stand-alone deployment estimates. The joint row demonstrates that
serving both Probes costs almost the same as serving one because only the tiny
CPU heads are duplicated. Normalized NLL, discrete SE, and cluster count
likewise share the same ten sampled answers; SE and cluster count also share
the same NLI clustering result.

## Fixed implementation

- Generator: `google/gemma-3-12b-it`, `bfloat16`, CUDA.
- Sampling: 10 samples, `T=1.0`, `top_p=0.9`, `top_k=50`,
  `max_new_tokens=192`, seed 31.
- NLI: `pritamdeka/PubMedBERT-MNLI-MedNLI`.
- Clustering: set-aware one-to-one bidirectional entailment for factoid/list;
  ordinary bidirectional entailment for summary, matching Phase 1.
- Probe feature: one deterministic no-cache replay over the exact saved
  prompt and generated token IDs; block 24, final answer token (`LT`).
- Frozen Probe bundle: the accepted Phase-2 P(True)-Probe and Accuracy-Probe;
  no target-side changes are possible in the scoring path.

The implementation is:

- `scripts/benchmark_phase2_uq_efficiency.py`
- `scripts/run_phase2_uq_efficiency_isambard.sbatch`

## Reported efficiency fields

CUDA events and wall-clock timers are recorded per question and per stage. The
final tables report:

- mean, median, and p95 incremental latency;
- total and mean CUDA GPU time;
- total and per-question extra generated tokens;
- questions per second and per minute;
- generated tokens per second for sampling methods;
- causal-LM call count and NLI forward-call count;
- one-time NLI load cost separately from steady-state cost; and
- AUROC/AP against Claude `incorrect` on the same question rows.

The output directory retains:

```text
selection.csv
stage_timing_per_example.csv
method_efficiency_per_example.csv
efficiency_summary.csv
uq_scores.csv
uq_ranking_metrics.csv
benchmark_summary.json
benchmark_status.json
run_timing.txt
```

Append-only JSONL checkpoints are also written during the run so that a
partial failure remains inspectable.

## Semantic-Entropy time budget

The completed Phase-1 jobs required `7:14:24` and `8:00:08` for 1,000
questions with ten samples and NLI. A simple question-count projection gives
about `2:47` to `3:04` for 384 questions. This is only a planning estimate:
answer lengths, NLI comparison counts, model-load time, and node performance
are data-dependent. The 12-hour full-job limit is therefore conservative, and
the pending smoke is the authoritative runtime check.

No conclusion about the Probe efficiency advantage or SE feasibility should
be reported until the smoke and full artifacts pass their health checks.
