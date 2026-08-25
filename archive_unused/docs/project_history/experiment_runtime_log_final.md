# Experiment Runtime Log

> Archived on 2026-08-17 after experimental closeout. This complete log remains
> the job-level provenance record; it is not an active scheduling plan. Use
> `../../../PROJECT_CLOSEOUT.md` for the concise current map.

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

## PubMedQA-v2 SE temperature sensitivity — completed

```text
submission: 2026-08-06
source revision: 5a302c1
smoke job: 5931810, six-question runnability-only gate
formal job: 5931811, 200-question paired temperature study
dependency: afterok:5931810
smoke result: COMPLETED 0:0 in 00:09:51; 6 questions, 120 new generations;
  interface and health checks PASS
formal result: COMPLETED 0:0 in 04:10:11; internal timer 14,995 seconds;
  200 questions, 4,000 new generations, health and paired-analysis checks PASS
formal Slurm interval: 2026-08-07T02:37:01 to 2026-08-07T06:47:12
formal batch MaxRSS: 1,641,216 K
cohort: deterministic seed-20260806 stratification by frozen incorrect x
  official yes/no/maybe; 200 questions, 56 incorrect and 144 correct; selection
  does not inspect the accepted T=1.0 SE values
temperature arms: reuse accepted T=1.0; generate only T=0.7 and T=1.3
new generation volume: 2 x 200 x 10 = 4,000 answers in the formal task
frozen settings: google/gemma-3-12b-it, context-v2 prompts, ten samples,
  generation seed 31, top_p=0.9, top_k=50, max_new_tokens=192, unchanged
  PubMedBERT bidirectional question-conditioned NLI, unchanged error labels
analysis: paired AUROC difference is primary; AP, mean SE, mean cluster count,
  and single-cluster fraction are secondary; 20,000 paired bootstrap resamples
environment: explicit .venv_isambard_cuda127
local validation: 136 tests, compileall, CLI, shell/Slurm syntax, and
  git diff --check passed
remote preflight: exact source/code hashes and 200-row six-stratum manifest PASS
monitoring: no recurring or continuous monitor started
AUROC at T=0.7/1.0/1.3: 0.5837053571 / 0.6213417659 / 0.5944940476
AP at T=0.7/1.0/1.3: 0.3656766917 / 0.3809991008 / 0.3850356269
mean SE at T=0.7/1.0/1.3: 0.0514188060 / 0.0639836085 / 0.0549893823
single-cluster fraction at T=0.7/1.0/1.3: 0.905 / 0.855 / 0.885
paired AUROC, T=0.7 minus T=1.0: -0.0376364087,
  95% CI [-0.0913477665, 0.0123046920]
paired AUROC, T=1.3 minus T=1.0: -0.0268477183,
  95% CI [-0.0774007367, 0.0212768241]
paired single-cluster fraction, T=0.7 minus T=1.0: +0.050,
  95% CI [0.010, 0.095]
interpretation: no clear temperature effect on SE ranking performance within
  the tested range and cohort; low T increases single-cluster collapse, but
  T=1.0 is only the best AUROC point estimate, not a resolved optimum
manifest SHA-256:
  99506184c8682552d641046fac948631322f575360de2c25d369fcc422006195
comparison summary SHA-256:
  34660b3d1d187e83ce24bf388f8afdf4ad66adcbdfaaf6ea1172ed42173263a3
temperature metrics SHA-256:
  0595f94bfc30aa98cd7807342129c108abcddfd8fceb6ea673d39df9016f49dd
paired differences SHA-256:
  013ae270901469caa4f038478c37257d5fe035df2f26503a518eadef9e109221
log review: no traceback, OOM, NaN, or execution error; only non-blocking
  Transformers processor deprecation warnings
status: accepted; no further temperature rerun is planned
```

## PubMedQA context-v2 Semantic Entropy — completed

```text
submission: 2026-08-05
source revision: d778553
smoke job: 5921808, three accepted v2 questions; runnability-only gate
formal job: 5921809, exact 500-question accepted Appendix-C v2 cohort
dependency: afterok:5921808
source: accepted v2 examples/prompts and existing official-decision incorrect
  labels from the 2026-07-22 full500 run
model / generation: google/gemma-3-12b-it, ten T=1.0 samples, top_p=0.9,
  top_k=50, seed 31, max_new_tokens=192
clustering: pritamdeka/PubMedBERT-MNLI-MedNLI, bidirectional entailment,
  question-conditioned, unchanged from the accepted Phase-2 SE protocol
primary metric: discrete Semantic Entropy AUROC/AP against the frozen v2
  incorrect target
environment: explicit .venv_isambard_cuda127 override; runner import preflight
  passed there, while .venv_isambard lacks scipy and was not used
scope exclusion: no low-temperature regeneration, Probe or P(True) rerun,
  calibration, correctness judging, prompt tuning, model, or dataset expansion
monitoring: no recurring or continuous monitor started
smoke result: COMPLETED 0:0 in 00:02:28; internal timer 120 seconds;
  3 questions, 30 generations, health check PASS
formal result: COMPLETED 0:0 in 04:59:51; internal timer 17,978 seconds;
  500 questions, 5,000 generations, 500 clusters/predictions; health check PASS
formal Slurm interval: 2026-08-06T12:09:17 to 2026-08-06T17:09:08
formal batch MaxRSS: 1,684,672 K
generation health: zero empty answers and zero 192-token-limit answers
result: discrete Semantic Entropy AUROC 0.5884378253; AP 0.3674408080;
  138/500 incorrect, prevalence 0.276
cluster counts: 444/46/7/3 questions with 1/2/3/4 clusters
provenance: exact equality to the 500 accepted examples/prompts and frozen
  labels; recorded source hashes, source revision d778553, runner hash, and
  batch hash all match the accepted files
formal summary SHA-256:
  dd5da74bc4821eda9f68753b5166efa7c43c08efeeeadb2a2f107df00ea1a62b
formal metrics SHA-256:
  4c9b4568178bb19ca0208bd5c9be638a42067056a27e547c1ec3afec307b3759
formal SE-score SHA-256:
  ba9274de686f487ab165b20925fc4adcc6e16a9ddedbd238813c97175cfe2b3a
log review: no traceback or execution error; Transformers emitted one
  non-blocking processor deprecation warning
status: accepted; the bounded missing PubMedQA-v2 SE row is complete
```

The result is eligible for the owning PubMedQA result document. The high
single-cluster rate is retained as a limitation rather than converted into a
new clustering or prompt experiment.

## Phase-2 validation SE for calibration — completed

```text
submission: 2026-08-01
revision: f84836f82982b88357a8b5086f8977ef4b95d44e
smoke job: 5863759, six validation questions balanced across the three types
formal job: 5863760, 384 valid-labelled validation questions
dependency: afterok:5863759
model / generation / NLI: unchanged from accepted test job 5773786
split: validation only; the completed test generation is not rerun
output: new validation-only directories; existing test artifacts remain frozen
monitoring: no recurring or continuous monitor started
smoke result: COMPLETED 0:0 in 00:04:18; internal benchmark 00:03:10;
  six questions, factoid/list/summary = 2/2/2; health check PASS
formal result: COMPLETED 0:0 in 04:01:10; internal benchmark 04:00:03;
  384/384 rows, factoid/list/summary = 157/104/123; health check PASS
formal UQ-score SHA-256:
  cc86e63cd1197d3b896fc8e14808c2fa3be308ce96961b40a2b3d5e0962d6245
cohort validation: exact match to the 384-row frozen valid-labelled validation
  cohort; zero ID overlap with the 384-row test cohort; no missing SE values
matched protocol: Gemma 3 12B, ten T=1.0 samples, seed 31, PubMedBERT NLI,
  and frozen probe-bundle SHA-256
  05c4dee461cdf789af5fc1ca45ef223dbf8fc4eab54d4724e57c59993d129f5a
status: accepted; complete offline calibration analysis generated locally
```

The accepted output is retained locally under
`analysis_outputs/bioasq_phase2_uq_calibration_validation384_20260801/`; the
complete offline result is under
`analysis_outputs/bioasq_phase2_uq_calibration_full_20260801/`.

## Gemma 3 1B staged model-scale generation — completed

```text
initial submission: 2026-07-27
initial smoke job: 5802163, FAILED 3:0 in 00:00:30 during model loading;
  cohort preflight passed and CUDA was available, but no answers were generated
initial staged job: 5802164; never ran because afterok:5802163 could not be
  satisfied; explicitly cancelled before replacement submission
failure cause: the gated 1B repository was accessible and config/tokenizer
  loaded, but model.safetensors was absent from the shared cache; the compute
  node failed while attempting to load the uncached weights
repair: downloaded and verified all 10 files for fixed model revision
  dcc83ea841ab6100d6b47a070329e1ba4cf78752 on the login node; the replacement
  entrypoint requires that snapshot and uses local_files_only for model and NLI
first replacement smoke job: 5807823, FAILED 3:0 in 00:00:24 during model
  construction despite the complete cache
first replacement staged job: 5807824; never ran because afterok:5807823 could
  not be satisfied; explicitly cancelled before the second replacement
confirmed code cause: the shared generator routed every gemma-3 checkpoint
  through AutoProcessor + Gemma3ForConditionalGeneration, but 1B is text-only
  Gemma3TextConfig + Gemma3ForCausalLM and has no image processor
code repair: route model_type=gemma3_text through AutoTokenizer +
  AutoModelForCausalLM while preserving the multimodal 4B/12B path; 118 local
  unit tests passed, including a dedicated route test
second replacement smoke job: 5808905; COMPLETED 0:0 in 00:02:06; internal
  pipeline 00:01:55; six questions, factoid/list/summary = 2/2/2
second replacement staged job: 5808906; COMPLETED 0:0 in 02:06:35; internal
  pipeline 02:05:58; 300 questions, factoid/list/summary = 50/50/200;
  dependency afterok:5808905 was satisfied
model / generation: google/gemma-3-1b-it, seed 31; otherwise the accepted
  Phase-1 no-evidence prompts, ten T=1.0 samples, one T=0.1 main answer,
  top_p=0.9, top_k=50, max_new_tokens=192, PubMedBERT NLI, and blind P(True)
reference cohort SHA-256:
  4c40280a6ab234d204ad1f60e9c7d06a588f8387ee5ee21a4d2c1028e907b7d0
smoke cohort ID SHA-256:
  64ffbd1a52375afe632549726890d95b015b107dfefb35cf200d81e4c9ffdb01
staged cohort ID SHA-256:
  f2b4a85281f2bf0604b6effa7a7ae238f3832ca88cb489678e03b4bb46b38108
preflight: local and Isambard source-derived subset/record checks PASS
health/postflight: both final jobs PASS Level-4 health, reference-subset, and
  exact frozen-cohort checks; staged artifacts contain 300 main generations,
  3,000 high-temperature generations, and 300 complete SE/self-report/UQ rows
correctness judging: protocol-matched 300-request Claude batch
  msgbatch_01LCGoC6Mh4EBNKHAq3XEB5U ended with 300/300 API successes and
  298 valid labels; factoid 8/50 correct, list 3/50 correct; two summary rows
  stopped at the 32-token limit
bounded label repair: msgbatch_01AWNB4xt1h86wyfRdm6fq2F, exactly two summary
  requests, same model/low effort with max_tokens increased once to 64; two API
  successes, one additional valid label; final 299/300 valid, no further retry
monitoring: no recurring or continuous monitor started
formal common-valid summary analysis: n=196; 1B accuracy 0.122; blind P(True),
  discrete SE, normalized NLL AUROC = 0.501/0.597/0.789;
  P(True)-minus-SE = -0.096, 95% CI [-0.244,+0.060];
  1B-minus-4B gap change = -0.104, 95% CI [-0.267,+0.059]
decision: do not expand the remaining factoid or list cohorts
status: staged 1B experiment and planned analysis complete
```

The dependency gate worked as intended: staged job `5808906` started only after
smoke `5808905` exited `0:0`. Scientific analysis is complete.

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
  statistically resolved; fusion is exploratory, archived, and not a new main
  model
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
  but this initial AUROC-change CI crosses zero; the terminal extension and
  completed human review below own the final mechanism conclusion
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
  proposed practical method
blinded 24-pair review completed: original NLI agreement 9/24 with 5 false
  merges and 10 false splits; Claude clustering agreement 19/24 with 0 false
  merges and 5 false splits; Claude direct judgement agreement 17/24 with
  0 false merges and 7 false splits
review boundary: four comparison cells were deliberately sampled as 8/8/4/4;
  agreement rates are diagnostic and not population accuracy estimates
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
