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

## Pending Phase-2 collection

```text
submitted: 2026-07-19
smoke job: 5719092, PENDING (scheduler priority at last check)
full job: 5719110, PENDING with afterok:5719092 dependency
source: complete eligible BioASQ training13b corpus, 3,930 questions
  (train 3,144; validation 393; test 393), frozen type-stratified manifest
model / generation: google/gemma-3-12b-it, bfloat16 CUDA, one T=0.1 answer,
  top_p=0.9, top_k=50, max_new_tokens=192
included UQ: blind P(True), verbalized confidence, single-answer token NLL and
  entropy summaries
hidden states: blocks 24/32/40/48 × TBG/SLT/LT, bf16,
  [example, 4, 3, 3840] tensors split by manifest partition
excluded: high-temperature generations, Semantic Entropy, NLI clustering,
  sample disagreement, P(True)-10
planned validation: post-run health check verifies one answer/P(True) row per
  selected question and all hidden-state tensor/index counts and shapes
```

This is a submission record, not a completed-run record: queueing and model
runtime are unknown until Slurm begins and writes `run_timing.txt`.

## Phase-1 final BioASQ baseline

```text
submitted: 2026-07-18
generation/NLI jobs: 5706186 (seed 31), 5706187 (seed 47)
generation/NLI status: completed all artifacts; initial exit 1 only because the
  health check expected the old ordinary-NLI name after set-aware clustering
repair: commit 7b90d54; repaired health checks PASS / PASS
dataset: BioASQ training13b; 1,000 fixed questions (480 factoid, 320 list,
  200 summary), ten samples per question, 10,000 generations per seed
model / generation: google/gemma-3-12b-it, bfloat16 CUDA, T=1.0, top_p=0.9
target answer: one T=0.1 answer per question
NLI: pritamdeka/PubMedBERT-MNLI-MedNLI; set-aware factoid/list, ordinary summary
elapsed: 08:00:08 (seed 31), 07:14:24 (seed 47), excluding queue time
self-report jobs: 5715701 (seed 31), 5715703 (seed 47), completed exit 0:0
self-report elapsed: 00:06:58 / 00:06:34; blind P(True) only
Claude batches: 1,000 requests per seed plus 24/29 blank-response retries
final labels: 991 valid per seed; remaining nine blanks per seed excluded
```

## Protocol-correction runs

### BioASQ low-temperature main-answer backfill

```text
submitted: 2026-07-17
status: COMPLETED, exit 0:0
jobs: 5692776/5692777 (evidence, seeds 31/47); 5692779/5692780 (direct, seeds 31/47)
purpose: add one paper-protocol accuracy-target answer to each completed 100x10 run
model: google/gemma-3-12b-it
temperature / top_p / top_k: 0.1 / 0.9 / 50
examples / answers: 100 / 100 per job
max new tokens: 192
generation: reads each existing prompts.jsonl; does not rerun high-temperature samples or NLI
elapsed: 00:14:30 / 00:15:23 (evidence seeds 31/47); 00:11:30 / 00:11:30 (direct seeds 31/47)
```

Each job wrote `best_generations.jsonl` and `best_generation_timing.txt`.
Claude Sonnet 5 subsequently labelled the 400 low-temperature main answers
locally; no API call was submitted through Slurm.

### BioASQ protocol-correct self-report UQ

```text
submitted: 2026-07-18
status: COMPLETED, exit 0:0
jobs: 5696576/5696577 (evidence, seeds 31/47); 5696578/5696579 (direct, seeds 31/47)
purpose: compute verbalized confidence and P(True) for each low-temperature main answer
model: google/gemma-3-12b-it
examples / answers: 100 / 100 per job
P(True) context: the existing ten high-temperature cleaned generations
driver/environment: cuda127-cu126
elapsed: 00:01:20 / 00:01:33 (evidence seeds 31/47); 00:01:27 / 00:01:27 (direct seeds 31/47)
```

The downloaded tables were verified at 100 questions per run with
`answer_source=best_generation_low_temperature` and ten high-temperature
samples recorded as P(True) context.

## Completed runs

### BioASQ training summary 100x10 - no-evidence direct-answer pair

```text
submitted: 2026-07-16
status: COMPLETED, exit 0:0; both Level 4 health checks PASS
jobs: 5684358 (seed 31), 5684360 (seed 47)
purpose: controlled evidence-ablation with direct biomedical answering
model: google/gemma-3-12b-it
temperature / top_p: 1.0 / 0.9
examples / samples: 100 / 10
max new tokens: 192
prompt evidence: omitted (`INCLUDE_EVIDENCE=0`, `bioasq_direct_v1`)
clustering: microsoft/deberta-v2-xlarge-mnli, bidirectional entailment
self-report UQ: enabled
elapsed: 01:59:42 (seed 31) and 01:59:20 (seed 47), excluding queue time
nodes: nid010486 (seed 31) and nid010568 (seed 47)
driver/environment: 565.57.01 / cuda127-cu126, torch 2.6.0+cu126
```

The evaluator retains ideal-answer metadata but treats the citation axis as not
applicable, producing the ROUGE + ideal-answer-NLI two-axis fallback. Its saved
bottom-30%-within-type deterministic labels are historical diagnostics; the
primary UQ outcome will use the separate low-temperature Claude label after the
backfill jobs complete.

### Isambard jobs 5683932 and 5683933 - three-axis NLI re-evaluation

```text
status: COMPLETED, exit 0:0
source runs: temperature-1.0 evidence-conditioned seeds 31 and 47
elapsed: 01:38 (seed 31) and 01:34 (seed 47)
nodes: nid010208 (seed 31) and nid010291 (seed 47)
model: microsoft/deberta-v2-xlarge-mnli, local cached files only
environment: cuda127-cu126, torch 2.6.0+cu126
generation: not rerun
final quality mode: three_axis
```

The refreshed evaluation artifacts report mean three-axis quality of 0.1968
(seed 31) and 0.1864 (seed 47). These values are not directly comparable to
the earlier reference-only scores, and the direct-answer ablation will use the
two-axis fallback because citations are intentionally absent.

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
