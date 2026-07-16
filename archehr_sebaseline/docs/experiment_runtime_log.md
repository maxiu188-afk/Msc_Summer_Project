# Experiment Runtime Log

This file records completed formal server runs for planning future allocations.
Queueing time is excluded: elapsed time is measured from Slurm start to end.

## Runtime policy

- Every formal Isambard run must retain its Slurm job ID, start/end timestamps,
  elapsed time, exit code, main configuration, and output health status.
- `scripts/run_bioasq_isambard.sbatch` writes these fields automatically to
  `OUTPUT_DIR/run_timing.txt`, including failed runs.
- `scripts/run_bioasq_llm_judge.sbatch` applies the same timing policy to the
  fixed-subset independent quality validation.
- After a run completes, add one entry here and retain the Slurm `.out`/`.err`
  logs with the downloaded results.
- Estimates should include a safety margin and should not silently substitute a
  different model, device, precision, sample count, or evaluation stage.

## Submitted runs awaiting result collection

### BioASQ three-axis ideal-answer NLI re-evaluation - temperature 1.0 pair

```text
submitted: 2026-07-16
status: RUNNING (both started 2026-07-16T22:18:34)
jobs: 5683932 (seed 31), 5683933 (seed 47)
purpose: add ideal/exact-answer NLI coverage to already completed outputs
model: microsoft/deberta-v2-xlarge-mnli
quality axes: lexical reference coverage, cited-document overlap, NLI coverage
generation: not rerun
```

The jobs use `scripts/evaluate_bioasq_nli_isambard.sbatch`, select the CUDA
environment from the allocated node's driver, and record their own
`reference_nli_timing.txt` in each output directory. Download the refreshed
`bioasq_eval/` artifacts after they complete.

## Completed runs

### BioASQ training summary 100x10 - temperature 1.0 paired repeat

```text
submitted: 2026-07-16
status: COMPLETED, exit 0:0; both Level 4 health checks PASS
jobs: 5679663 (seed 31), 5679664 (seed 47)
purpose: controlled sampling-temperature sensitivity check for UQ
model: google/gemma-3-12b-it
temperature / top_p: 1.0 / 0.9
seeds: 31 and 47
examples / samples: 100 / 10
max new tokens: 192 (unchanged from the historical baseline)
clustering: microsoft/deberta-v2-xlarge-mnli, bidirectional entailment
self-report UQ: enabled
elapsed: 02:36:46 (seed 31) and 02:28:21 (seed 47), excluding queue time
nodes: nid010501 (seed 31) and nid010661 (seed 47)
driver/environment: 565.57.01 / cuda127-cu126, torch 2.6.0+cu126
```

The downloaded artifacts contain 100 examples and 1,000 generations per seed.
Their local citation-aware reference re-evaluation is complete; separate queued
jobs are adding the NLI axis. The fixed quality threshold remains provisional until
manual calibration.

### Isambard job 5654721 - BioASQ training summary 100x10

```text
status: COMPLETED (exit 0:0)
Slurm start: 2026-07-14T22:12:03
Slurm end: 2026-07-15T01:55:54
elapsed: 03:43:51 (13,431 seconds)
resources: 1 GPU, 8 requested CPUs, 64 GB requested memory
batch MaxRSS: 8,758,336 K (about 8.35 GiB)
dataset: BioASQ training13b summary
examples: 100
samples per example: 10
generations: 1,000
model: google/gemma-3-12b-it
generation dtype/device: bfloat16 / CUDA
max new tokens: 192
clustering: microsoft/deberta-v2-xlarge-mnli, bidirectional entailment
self-report UQ: enabled
health check: PASS
```

Observed phase times from the Slurm log:

```text
sampled generation: 03:18:20 (11.9 seconds per generation)
NLI clustering: 00:01:51 (1.11 seconds per example)
model loading + self-report UQ + evaluation: approximately 00:23:40
```

Planning reference for the same model and pipeline:

- Observed end-to-end rate: about 134 seconds per example at 10 samples.
- A linear 50-example estimate is about 1 hour 52 minutes; 80 examples is about
  2 hours 59 minutes; 100 examples is the observed 3 hours 44 minutes.
- Add at least 15% for normal planning, and more for an empty model cache or
  different answer lengths. Keep the current 12-hour Slurm limit until repeated
  runs establish stable variance.

These estimates apply only to the same single-GPU Gemma 3 12B, 10-sample, NLI,
self-report configuration. Factoid/list workloads and hidden-state SEP runs may
scale differently.

### Isambard job 5660345 - fixed-subset independent Qwen judge

```text
status: COMPLETED (exit 0:0)
Slurm start: 2026-07-15T09:17:27
Slurm end: 2026-07-15T09:38:24
elapsed: 00:20:57 (Slurm), 00:20:56 / 1,256 seconds (script timer)
resources: 1 GPU, 72 allocated CPUs, 48 GB requested memory
batch MaxRSS: 2,740,544 K (about 2.61 GiB)
source run: Isambard job 5654721, seed 31
judge: Qwen/Qwen2.5-7B-Instruct, greedy decoding
selection: 30 questions, equal samples from lexical-quality thirds
judgments: 300 generations
valid parsed judgments: 296
selection seed: 20260715
```

Observed end-to-end rate was about 4.19 seconds per judgment, including model
loading and output analysis. Four responses reached the 160-token generation
limit after emitting all rubric scores but before closing the JSON rationale;
future judge runs should allow at least 256 new tokens or enforce a shorter
rationale.

### Isambard job 5660346 - BioASQ training summary 100x10 seed repeat

```text
status: COMPLETED (exit 0:0)
Slurm start: 2026-07-15T12:50:11
Slurm end: 2026-07-15T16:08:43
elapsed: 03:18:32 (Slurm), 03:18:31 / 11,911 seconds (script timer)
resources: 1 GPU, 72 allocated CPUs, 64 GB requested memory
batch MaxRSS: 8,530,112 K (about 8.13 GiB)
dataset: BioASQ training13b summary
examples: 100
samples per example: 10
generations: 1,000
model: google/gemma-3-12b-it
generation seed: 47
generation dtype/device: bfloat16 / CUDA
max new tokens: 192
clustering: microsoft/deberta-v2-xlarge-mnli, bidirectional entailment
self-report UQ: enabled
health check: PASS
```

Across the two matched 100x10 runs (seeds 31 and 47), elapsed time ranged from
3:18:31 to 3:43:51, with a mean of about 3:31:12. Continue planning around
4:20 for this cached configuration (approximately 15% above the slower
observation), excluding queue time; retain the 12-hour limit until more runs
show that a smaller allocation is consistently safe.
