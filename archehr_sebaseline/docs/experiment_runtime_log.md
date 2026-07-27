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

## BioASQ summary answer-length intervention

```text
submitted/completed: 2026-07-26
cohort: all 123 valid-labelled summary questions from the Phase-2 efficiency
  test cohort; current/short correctness-UQ comparison uses 121 paired rows
smoke job: 5781225, COMPLETED 0:0 in 00:06:28; three questions; health PASS
full job: 5781246, COMPLETED 0:0 in 03:59:14; dependency afterok:5781225;
  123/123 questions and both-condition health check PASS
model / prompts: unchanged google/gemma-3-12b-it and current direct-summary
  prompt versus the frozen one-or-two-sentence/no-extra-background addition
generation: one T=0.1 main answer and ten T=1.0 samples; top_p=0.9, top_k=50,
  max_new_tokens=192, seed 31; no answer reached 192 tokens
NLI: pritamdeka/PubMedBERT-MNLI-MedNLI, question-conditioned bidirectional
  entailment, matching the accepted summary rule
artifact health: each condition has 123 main answers, 1,230 samples, 123
  complete cluster records, and 123 condition-score rows
current reuse: accepted main answer, Claude label, and blind P(True); current
  high-temperature samples were deterministically regenerated because the
  efficiency output lacked answer texts/full clusters; all four UQ score
  fields exactly reproduce the prior 123 summary rows
short Claude batches: msgbatch_01GyM5xLBcp7oyMV8BSUZ8rc plus same-config
  retries msgbatch_019upAX3uH2X1joZvufjNZe2 and
  msgbatch_01NAapx7qM5MBw54LCZ7hBdf; 121 valid / 123, two blanks excluded only
  from paired correctness-UQ analysis
length result: main words 82.88 -> 52.11; sentences 3.25 -> 2.00;
  short main/sample one-or-two-sentence compliance 123/123 and 1,230/1,230
paired accuracy: 56/121 in both conditions; 8 current-only correct and 8
  short-only correct
UQ AUROC current -> short: blind P(True) 0.7948 -> 0.8040; discrete SE
  0.5782 -> 0.5468; cluster count 0.5805 -> 0.5451; normalized NLL
  0.7533 -> 0.7511
primary length effect: [blind-P(True) AUROC - SE AUROC] short minus current
  +0.04052; 20,000-resample paired-bootstrap 95% CI [-0.07126, 0.15306]
interpretation: shortening clearly changed length but did not recover SE
  relative to blind P(True); the relative-performance change is unresolved
sentence repair: abbreviation-aware deterministic repair of derived fields
  only; original tables archived and hashes recorded; no GPU rerun
```

Complete protocol, result tables, and artifact pointers are in
`summary_length_intervention.md`.

## Bounded long-summary clustering diagnostic

```text
submitted: 2026-07-26
scope: 48 current-condition summary questions selected by deterministic
  label-stratified sampling (24 correct, 24 incorrect), with ten fixed saved
  samples per question; no answer regeneration and no short-condition run
Claude batch: msgbatch_01BcAEtYs7FmcPVtNgaMaz2D; 48 requests;
  claude-sonnet-5, low effort, max_tokens=128
first batch outcome: all 48 API requests succeeded, but only 2/48 outputs were
  complete; 20 ended normally with 42/45 decisions and 26 hit max_tokens
repair boundary: preserve the first batch; retry only the 46 invalid rows with
  the same cohort, rubric, and pair order, changing only serialization to
  schema-constrained JSON and max_tokens to 512
repair smoke: one invalid row returned 45/45 valid decisions with end_turn
repair batch: msgbatch_01DvxtFBPJoeBtGequbL4Wpc; 46 requests
first repair outcome: 46/46 API requests succeeded; 37 passed strict 45-item
  validation and 9 hit max_tokens, giving 39/48 valid rows in total
second repair: msgbatch_0147chhsAUMSZxS7zkTiBaE9; only the 9 truncated rows,
  same schema and scientific inputs, max_tokens=2048
second repair outcome: 9/9 API requests succeeded; 8 passed strict validation
  and one correct-class row again hit max_tokens
final analysis set: 47/48 complete cases, 23 correct and 24 incorrect; no
  further retry
UQ AUROC: blind P(True) 0.8252; original NLI-SE 0.5788;
  Claude-clustered SE 0.7274
Claude-SE minus NLI-SE AUROC: +0.1486; 20,000-resample paired-bootstrap
  95% CI [-0.0163, 0.3125]
P(True) minus Claude-SE AUROC: +0.0978; 95% CI [-0.0607, 0.2536]
cluster structure: mean clusters NLI 1.23 versus Claude 3.98; mean pair
  agreement 0.592; mean ARI 0.396; 821 NLI-same/Claude-different pairs versus
  41 in the opposite direction
interpretation: NLI over-merging is a credible contributor to weak summary SE,
  but the AUROC-change CI crosses zero and blinded human review is pending
local validation: 115/115 current unit tests, compileall, git diff --check,
  and the earlier 48-question synthetic-copy analysis smoke passed
monitoring: no recurring monitor started
initial quantitative result: complete; expanded result follows below
```

The frozen bounded protocol is in `summary_clustering_diagnostic.md`.

Nested extension:

```text
submitted: 2026-07-26
target: 96 long-summary questions, 48 correct and 48 incorrect
reuse: original 48-question cohort is an exact subset; only 48 new Claude
  requests are submitted
purpose: mechanism diagnosis of long-answer NLI difficulty, not a practical
  Claude-clustering improvement
batch: msgbatch_015tEX5qdiBHpb2sUUME4gDW; claude-sonnet-5, low effort,
  structured output, max_tokens=2048
batch outcome: 48/48 API requests succeeded; 46 passed strict validation and
  2 hit max_tokens
retry boundary: retain valid complete cases and do not retry invalid rows
final expanded analysis: 93/96 complete cases, 46 correct and 47 incorrect
UQ AUROC: blind P(True) 0.8148; original NLI-SE 0.5946;
  Claude-clustered SE 0.7812
Claude-SE minus NLI-SE AUROC: +0.1866; 20,000-resample paired-bootstrap
  95% CI [0.0800, 0.2907]
P(True) minus NLI-SE: +0.2202 [0.1098, 0.3254]
P(True) minus Claude-SE: +0.0335 [-0.0786, 0.1466]
cluster structure: mean clusters NLI 1.19 versus Claude 3.88; mean pair
  agreement 0.588; mean ARI 0.368; 1,673 NLI-same/Claude-different pairs
  versus 50 in the opposite direction
interpretation: fixed-answer reclustering closes about 85% of the original
  P(True)-minus-SE point-estimate gap, strongly implicating long-answer NLI
  clustering difficulty; Claude remains an expensive diagnostic, not the
  proposed practical method, and blinded pair review remains pending
```

## Gemma 3 4B Phase-1-aligned model-scale run

```text
submitted: 2026-07-26
model: google/gemma-3-4b-it, bfloat16 CUDA, seed 31
reference: accepted Gemma 3 12B Phase-1 seed-31 examples.jsonl
reference SHA-256:
  4c40280a6ab234d204ad1f60e9c7d06a588f8387ee5ee21a4d2c1028e907b7d0
alignment: source-derived preflight plus saved-output postflight; full requires
  exact 480/320/200 ID and record equality, smoke requires a 2/2/2 subset
generation/UQ: unchanged Phase-1 no-evidence prompts; one T=0.1 main answer,
  ten T=1.0 samples, top_p=0.9, top_k=50, max_new_tokens=192,
  PubMedBERT NLI, token/NLL fields, verbal confidence, blind P(True)
smoke job: 5785064, six questions, COMPLETED 0:0, 00:03:37
full job: 5785065, 1,000 questions, dependency afterok:5785064,
  COMPLETED 0:0, 05:46:48
full internal timing: 20,798 seconds / 05:46:38
health: Level-4 PASS; 1,000 examples, 10,000 generations, 1,000 clusters,
  complete token/UQ fields, CUDA bfloat16, no NaN/Infinity markers
postflight alignment: PASS exact; 480 factoid, 320 list, 200 summary;
  all records identical to accepted 12B Phase-1 seed-31 cohort
Claude correctness batch: msgbatch_01AbxKWPN4pdgYoh3CF2V42G;
  1,000/1,000 API successes; 986 valid labels (787 incorrect, 199 correct);
  11 truncated and 3 refusal rows invalid
bounded retry: msgbatch_01NtMndSj62sKhvhMrkQbopS; only 14 invalid rows,
  14/14 API successes; final 997/1,000 valid labels (796 incorrect,
  201 correct); 3 invalid rows excluded, no second retry
paired comparison: 990 questions with valid labels for both 4B and 12B
overall AUROC 4B: blind P(True) 0.7582, discrete SE 0.7675,
  normalized NLL 0.7549
overall AUROC 12B: blind P(True) 0.8107, discrete SE 0.7641,
  normalized NLL 0.7291
P(True)-minus-SE gap: 4B -0.0093 [-0.0509, 0.0311];
  12B +0.0466 [0.0169, 0.0757]
primary model-size gap change, 4B minus 12B: -0.0560,
  95% CI [-0.1032, -0.0102]
summary gap change: -0.2382, 95% CI [-0.3474, -0.1285];
  4B P(True)/SE AUROC 0.6478/0.6427 versus 12B 0.8079/0.5645
interpretation: lower model capability removes blind P(True)'s overall and
  summary advantage; this is P(True) weakening, not SE becoming a strong 4B
  summary method
monitoring: no recurring monitor started
scientific result: complete
```

The frozen protocol is in `gemma3_model_scale_experiment.md`.

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
