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

## Phase-2 collection and local linear-Probe pass

```text
submitted: 2026-07-19
smoke job: 5719092, completed 0:0, elapsed 00:00:27
full job: 5719110, completed 0:0, elapsed 03:44:58
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
health validation: PASS; 3,930 answers/P(True) rows and hidden-state rows
  split 3,144/393/393 with `[example, 4, 3, 3840]` layout
local post-processing: CPU-only regularised linear Probes over saved artifacts;
  train fit, validation selection, then one frozen test evaluation
Claude labels: 3,858 valid / 3,930 after two retries; 72 blanks excluded only
  from Accuracy-Probe (valid train/validation/test rows 3,090/384/384)
frozen results: P(True)-Probe hard-even L2 logistic block 24/LT, target AUROC
  0.9026; Accuracy-Probe L2 logistic block 24/LT, Claude AUROC 0.8058/AP 0.8884
local ElasticNet rerun: 2026-07-23, continuous blind P(True) training target,
  validation-selected block 24/SLT; test Spearman 0.6057/MAE 0.2189
correctness-UQ diagnostic: ElasticNet AUROC 0.6532/AP 0.7945 on 382 valid
  Claude-labelled rows; matched hard-even Probe AUROC 0.7456/AP 0.8362
cross-dataset follow-up: no-context PubMedQA transfer completed; the separate
  official-context condition is recorded in the next section
```

The server collection is complete. Probe fitting is local and consumes the
saved artifacts only; it does not add an Isambard allocation.

## Phase-2 UQ efficiency benchmark

```text
submitted: 2026-07-23; replacement submitted 2026-07-24
status: complete; full artifact health_check PASS
smoke job: 5761273, COMPLETED 0:0 in 00:04:05; six questions; health_check PASS
first full job: 5761275, FAILED 1:0 in 00:00:19 before model loading because
  the node-provided TMPDIR did not exist; no benchmark rows
fix: batch script exports a job-specific temporary directory under SCRATCHDIR;
  incomplete output retained with suffix _failed_job5761275
replacement full job: 5773786, COMPLETED 0:0 in 03:31:33; 384/384 questions;
  no dependency because the accepted smoke was reused
cost boundary: incremental UQ work after the saved T=0.1 main answer exists;
  common Gemma load and original answer generation excluded from steady state
selection: Phase-2 test only, 158 factoid / 103 list / 123 summary;
  252 Claude-incorrect / 132 correct
Probes: frozen block-24/LT P(True)-Probe and Accuracy-Probe; one hidden-state
  replay shared by both CPU linear heads; zero extra generated tokens;
  60.50/60.40 ms per question and 60.54 ms jointly
blind P(True): 154.37 ms per question, zero freely generated tokens, but two
  teacher-forced fixed-continuation LM calls per question
original UQ comparators: blind P(True), ten-sample normalized NLL, discrete SE,
  and cluster count; verbalized confidence and weaker Phase-1 variants excluded
sampling / NLI: ten T=1.0 samples, top_p=0.9, top_k=50, max_new_tokens=192;
  pritamdeka/PubMedBERT-MNLI-MedNLI with the Phase-1 type-specific rules
reported cost fields: mean/median/p95 latency, CUDA GPU time, extra generated
  tokens, question/token throughput, causal-LM/NLI calls, and NLI load time
performance: Accuracy-Probe 0.8058/0.8884 AUROC/AP; blind P(True)
  0.7900/0.8383; discrete SE 0.7484/0.8446; P(True)-Probe 0.7452/0.8357
sampling cost: normalized NLL 30.55 s/question; SE 32.75 s/question;
  191,595 shared generated tokens, 498.95/question
score repair: job 5773786 initially inverted both already-uncertainty-positive
  Probe heads; saved scores were repaired without GPU rerun, and original
  files are retained under archive_probe_direction_pre_fix
measured benchmark runtime: 03:31:04 internal, versus historical planning
  projection 02:47--03:04
```

The complete cost boundary, sharing rules, result tables, and correction
provenance are in `phase2_uq_efficiency_benchmark.md`.

## Phase-2 statistical completion

```text
completed: 2026-07-25
compute: local CPU analysis of saved validation/test artifacts only; no GPU,
  generation, NLI, or Claude call
rows: 384 valid-labelled validation questions and 384 test questions
paired bootstrap: Accuracy-Probe minus blind P(True) test AUROC +0.015843;
  20,000/20,000 valid resamples; 95% CI [-0.038897, 0.071729]
fusion: StandardScaler + L2 logistic regression on validation P(True)-Probe
  and Accuracy-Probe scores, then one frozen test evaluation
fusion test: AUROC 0.8125, AP 0.8871, Brier 0.1688
Accuracy-Probe test: AUROC 0.8058, AP 0.8884, Brier 0.2162
interpretation: Accuracy-Probe advantage over blind P(True) is not
  statistically resolved; fusion is exploratory and not a new main model
output: analysis_outputs/bioasq_phase2_probe_completion_20260725
```

The protocol, coefficients, input hashes, complete metrics, and Phase-2
completion boundary are in `phase2_probe_completion_statistics.md`.

## PubMedQA frozen-Probe transfer

```text
submitted: 2026-07-20
smoke job: 5734703, MAX_EXAMPLES=3
smoke status: COMPLETED 0:0, Slurm elapsed 00:00:39; internal timed stage 33 s
source: official PubMedQA PQA-L 500-question test subset; smoke selects the
  first three deterministic PMIDs only
model / generation: unchanged google/gemma-3-12b-it, one T=0.1 no-evidence
  yes/no/maybe answer with a concise explanation
Probes: frozen BioASQ block-24/LT P(True)-Probe and Accuracy-Probe; no target
  fitting, recalibration, threshold tuning, or feature selection
other UQ: blind P(True), verbal confidence, sequence/normalized NLL, mean/max
  token entropy; no high-temperature sampling or Semantic Entropy
smoke health: PASS; 3/3 leading labels parsed, 3/3 answer/P(True)/hidden rows,
  hidden shape [3,4,3,3840], both frozen Probes and all UQ tables written
full 500-question job: 5739322, submitted 2026-07-21 after smoke gate PASS;
  COMPLETED 0:0 in 00:40:55, 500/500 answer/P(True)/hidden rows and
  [500,4,3,3840] hidden layout
no-context outcome: 125/500 official decisions correct; predicted labels
  maybe/yes/no = 373/123/4, motivating a separately named context rerun
context rerun: prompt version pubmedqa_context_explanation_v1; official
  CONTEXTS supplied to generation and self-report prompts, LONG_ANSWER withheld
context smoke: 5748203, MAX_EXAMPLES=3, COMPLETED 0:0 in 00:01:00
context full: 5748205, 500 questions, COMPLETED 0:0 in 00:37:43;
  500/500 output rows, hidden shape [500,4,3,3840], zero LONG_ANSWER leakage
context outcome: 295/500 official decisions correct; predicted labels
  maybe/yes/no = 163/237/100; 197 no-context errors corrected and 27 previous
  correct answers broken
context correctness UQ: verbal confidence AUROC 0.7933, blind P(True) 0.6800,
  frozen P(True)-Probe 0.6630, frozen Accuracy-Probe 0.5960;
  token-only UQ 0.5603--0.5716
question-only result disposition: archived on 2026-07-23 because the official
  PubMedQA decision depends on the omitted article evidence
Appendix-C v2 prompt: pubmedqa_context_explanation_v2; same official contexts,
  500 IDs, Gemma snapshot, generation settings, frozen block-24/LT Probes,
  BioASQ threshold, and evaluator as context v1
v2 smoke: 5750742, MAX_EXAMPLES=3, COMPLETED 0:0 in 00:00:46
v2 full: 5750745, 500 questions, COMPLETED 0:0 in 00:40:16;
  500/500 output rows, hidden shape [500,4,3,3840], zero LONG_ANSWER leakage
v2 outcome: 362/500 correct (72.4%); predicted labels yes/no/maybe =
  298/164/38; context-v1 paired outcomes 274 both correct, 21 v1-only correct,
  88 v2-only correct, 117 both wrong; exact paired p=6.11e-11
v2 correctness UQ: frozen P(True)-Probe AUROC 0.6839/AP 0.4519,
  blind P(True) 0.6490/0.4210, verbal confidence 0.6402/0.4164,
  frozen Accuracy-Probe 0.5901/0.3916; error prevalence 0.276
v2 limitation: yes/no/maybe recall 84.1%/72.8%/12.7%; retain the low maybe
  recall as type-specific error analysis, not as a rejection of the overall gain
```

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
