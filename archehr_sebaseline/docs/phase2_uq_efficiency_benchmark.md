# Phase-2 UQ Efficiency Benchmark

Last updated: 2026-07-25

## Status

The six-question smoke and replacement 384-question benchmark completed and
passed their artifact health checks.

| Stage | Slurm job | Selection | Dependency | Status |
| --- | ---: | ---: | --- | --- |
| Smoke | `5761273` | 6 labelled test questions, balanced across factoid/list/summary | none | completed `0:0` in `00:04:05`; health check passed |
| First full attempt | `5761275` | all 384 valid-labelled Phase-2 test questions | `afterok:5761273` | failed `1:0` after 19 s during Python startup |
| Replacement full | `5773786` | all 384 valid-labelled Phase-2 test questions | none; accepted smoke reused | completed `0:0` in `03:31:33`; health check passed |

Job `5761275` reached a different node after the smoke but inherited a
nonexistent `/local/user/...` temporary directory. Importing PyTorch attempted
to create a temporary directory there and raised `FileNotFoundError` before
the generator loaded. This is a node-environment failure, not a benchmark
result. The batch script now creates and exports a job-specific temporary
directory under `$SCRATCHDIR`. The incomplete output is preserved with suffix
`_failed_job5761275`; replacement job `5773786` reuses the passed smoke and
does not depend on it again.

All attempts include the Semantic-Entropy branch so that measured cost covers
its actual sampling and NLI work.

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

| Reported method | Incremental computation | Extra generated tokens | Fixed continuations scored |
| --- | --- | ---: | ---: |
| P(True)-Probe | one saved prompt+answer hidden-state replay, then its frozen CPU linear head | 0 | 0 |
| Accuracy-Probe | the same shared replay, then its frozen CPU linear head | 0 | 0 |
| Both Probes jointly | one shared replay and both frozen CPU heads | 0 | 0 |
| Blind P(True) | teacher-forced likelihood of fixed `True` and `False` continuations, using two causal-LM calls | 0 | 2 per question |
| 10-sample normalized NLL | ten `T=1.0` generations with token scores | all sampled tokens | 0 |
| Discrete SE | the same ten generations plus biomedical NLI clustering | the same sampled tokens | 0 |
| Cluster count | the same ten generations and NLI clusters as discrete SE | the same sampled tokens | 0 |

“Zero extra generated tokens” does not mean zero causal-LM computation. Blind
P(True) never calls free-running `generate()`: it appends each fixed
continuation to the prompt and scores its likelihood in a teacher-forced
forward pass. Its two calls per question are therefore represented by latency,
GPU time, and causal-LM call count, not by generated-output token count.

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

## Results

### Correctness ranking

All methods use the same 384 questions and `incorrect=1` target. The completed
job initially inverted both frozen-head probabilities even though their
positive classes already meant high P(True) uncertainty and answer incorrect.
This produced invalid Probe AUROCs `0.2548` and `0.1942`. The code now preserves
the heads' native uncertainty direction, and a deterministic saved-score repair
regenerated `uq_scores.*`, `uq_ranking_metrics.csv`, and
`benchmark_summary.json` without any GPU rerun. The pre-fix files remain in
`archive_probe_direction_pre_fix/`.

| Method | AUROC | AP |
| --- | ---: | ---: |
| Accuracy-Probe | **0.8058** | **0.8884** |
| Blind P(True) | 0.7900 | 0.8383 |
| Discrete SE | 0.7484 | 0.8446 |
| Cluster count | 0.7475 | 0.8368 |
| P(True)-Probe | 0.7452 | 0.8357 |
| 10-sample normalized NLL | 0.7382 | 0.8466 |

These Probe values reproduce the already accepted Phase-2 test results rather
than adding a second independent performance experiment. The new evidence here
is the matched efficiency measurement. The completed paired bootstrap gives
Accuracy-Probe minus blind P(True) AUROC `+0.01584`, 95% CI
`[-0.03890, 0.07173]`; superiority is therefore not statistically resolved.
See `phase2_probe_completion_statistics.md`.

### Incremental efficiency

| Method | Mean latency | Median | P95 | Throughput | Total GPU time | Extra generated tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy-Probe | **60.40 ms** | 59.24 ms | 67.06 ms | **16.56 q/s** | 23.17 s | 0 |
| P(True)-Probe | 60.50 ms | 59.30 ms | 67.32 ms | 16.53 q/s | 23.17 s | 0 |
| Both Probes jointly | 60.54 ms | 59.34 ms | 67.36 ms | 16.52 q/s | 23.17 s | 0 |
| Blind P(True) | 154.37 ms | 153.55 ms | 161.32 ms | 6.48 q/s | 59.27 s | 0 |
| 10-sample normalized NLL | 30.55 s | 14.90 s | 77.88 s | 0.0327 q/s | 11,731.40 s | 191,595 total |
| Discrete SE | 32.75 s | 18.20 s | 78.06 s | 0.0305 q/s | 12,577.68 s | 191,595 total |
| Cluster count | 32.75 s | 18.20 s | 78.06 s | 0.0305 q/s | 12,577.68 s | 191,595 total |

One P(True)-Probe is `2.55x` faster than blind P(True), reducing mean
incremental latency by `60.8%`; Accuracy-Probe is `2.56x` faster. Serving both
heads together adds only `0.06%` to the P(True)-Probe latency because they
share the replay. P(True)-Probe is `504.9x` faster than ten-sample normalized
NLL and `541.3x` faster than SE. Sampling methods generated `191,595` shared
tokens (`498.95` per question), whereas Probes and blind P(True) generated
none.

SE adds `2.20` seconds per question over the same ten-sample normalized-NLL
generation, or `846.27` seconds over all 384 questions. It made `128,842` NLI
forward calls; its one-time NLI model load was only `0.28` seconds. On this
configuration the ten generations dominate cost, although NLI still adds about
14.1 minutes. SE and cluster count share both generation and clustering, so
their identical costs must not be added together.

## Semantic-Entropy runtime

The earlier Phase-1 question-count projection was `02:47`--`03:04`. The
completed 384-question job took `03:31:33` under Slurm and `03:31:04` inside
the benchmark. The difference from the projection is consistent with
data-dependent generation lengths and NLI comparison counts. A future identical
run should budget at least four to five hours even though the 12-hour limit was
conservative.
