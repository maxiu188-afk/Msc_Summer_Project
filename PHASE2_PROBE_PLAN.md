# Phase 2 Probe Research Plan

Last updated: 2026-08-06

## Final status and research boundary

All three research phases and the bounded mechanism diagnostics are complete.
After the subsequent supervisor discussion, one thesis-closing UQ evaluation
was approved: validation-fitted calibration and selective prediction for SE,
blind P(True), and the frozen Probes, plus a correctness-label audit and
zero-refit PubMedQA calibration transfer. This evaluation is now complete. The
90-answer audit has 95.56% raw agreement, 95.12% design-weighted agreement, and
Cohen's kappa 0.902; its four corrections do not reverse the primary UQ
conclusions. The fixed protocol and complete results are in
`archehr_sebaseline/docs/phase2_uq_calibration_completion.md`. One later,
separately authorized addition is also complete: the existing ten-sample
Semantic Entropy method was run on the accepted 500-question PubMedQA
Appendix-C context-v2 prompt. It obtained 0.5884 AUROC / 0.3674 AP without
changing any low-temperature answer, correctness label, Probe, P(True),
calibration mapping, model, prompt, or NLI rule. No other prompt, model-scale,
Probe-training, type-specific, or clustering expansion is reopened; the
experimental programme is closed.

## Execution Status (updated 2026-08-06)

The single-answer artifact collector passed local unit/CLI/shell validation and
a server-side interface preflight. The Isambard Gemma 3 environment reports 48
text transformer blocks, hidden size 3,840, and a forward output with a
`hidden_states` field. The frozen manifest SHA-256 is
`d11953a5e6973809b2d024a9ad456098ec252db8290e73109234a1f1ac7b6433`.

- Smoke job **5719092**: completed `0:0` in 27 seconds, validating one
  question from each split.
- Full job **5719110**: completed `0:0` in 3:44:58. The health check verified
  3,930 main answers, blind P(True) rows, and hidden-state rows split as
  3,144/393/393 with layout `[N, 4, 3, 3840]`.
- Claude batches **`msgbatch_01CkzfKxvDPQT44XWiZDp7B6`** plus retries
  **`msgbatch_01Sb5PjfhzLdKWMHXr7cuN2D`** (105 records) and
  **`msgbatch_01Kn9cebjpWTMM2d58oZEg6D`** (80 records): completed and merged.
  The result has 3,858 valid labels; 72 blank/invalid labels are retained as
  missing and, by explicit decision, excluded from Accuracy-Probe fitting and
  evaluation rather than retried again.

The generation, single-answer UQ, and hidden-state artifacts are available and
accepted for P(True)-Probe training. Accuracy-Probe uses only its frozen
valid-label subset (train 3,090; validation 384; test 384); P(True)-Probe
continues to use all 3,930 questions.

The separate incremental-cost comparison is complete.
Smoke job **5761273** completed its six labelled test questions in `00:04:05`
and passed the artifact health check. The first 384-question job **5761275**
then failed during Python startup because its node exposed a nonexistent local
temporary directory; it produced no benchmark rows. The batch script now
creates a job-specific temporary directory under `$SCRATCHDIR`; replacement
full job **5773786** completed in `03:31:33` and passed its 384-question health
check. Both shared-replay Probes take about `60.5 ms/question`, blind P(True)
`154.4 ms`, ten-sample normalized NLL `30.55 s`, and SE/cluster count
`32.75 s`. Both Probes jointly cost `60.54 ms`, confirming that their CPU heads
are negligible after one replay. Blind P(True) generates no free-running output
tokens but scores two fixed continuations per question in two causal-LM calls.
See
`archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`.

The remaining Phase-2 statistics are also complete. On the same 384 test
questions, Accuracy-Probe minus blind P(True) has AUROC difference `+0.01584`
with paired-bootstrap 95% CI `[-0.03890, 0.07173]`; the interval includes zero.
The validation-fitted two-Probe fusion produced only a small mixed AUROC/AP
change and is not part of the main result. Its provenance is retained under
`archive_unused/docs/historical_results/feature_fusion_diagnostics_20260728.md`.
The active statistical conclusion is the paired Accuracy-Probe versus blind
P(True) interval above.

Initial Gemma 3 1B smoke **5802163** passed cohort/CUDA preflights but failed
`3:0` before generation because the 1B weights were absent from the shared
cache. Dependent staged job **5802164** never ran and was cancelled. The fixed
1B revision was then downloaded and checksum-verified; replacement smoke
**5807823** and staged job **5807824** were submitted with dependency
`afterok:5807823`. That smoke exposed a second issue before generation: the
shared generator incorrectly routed the text-only 1B checkpoint through the
multimodal Gemma 3 processor/model. The generator now selects the causal-LM
route from `model_type=gemma3_text`; 118 tests pass, and the existing
multimodal 4B/12B route is unchanged. Job **5807824** never ran and was
cancelled. Second replacement smoke **5808905** completed `0:0` in `00:02:06`;
staged job **5808906**, submitted with `afterok:5808905`, then completed `0:0`
in `02:06:35`. Both passed Level-4 health and exact frozen-cohort checks. The
frozen staged cohort is 50 factoid / 50 list / 200 summary questions. Local and
Isambard preflights passed with cohort-ID SHA-256
`f2b4a85281f2bf0604b6effa7a7ae238f3832ca88cb489678e03b4bb46b38108`.
The 300-question output is downloaded locally. Protocol-matched correctness
batch **`msgbatch_01LCGoC6Mh4EBNKHAq3XEB5U`** ended with 300/300 API successes
and 298 valid labels. The single bounded 64-token retry
**`msgbatch_01AWNB4xt1h86wyfRdm6fq2F`** returned two API successes and raised
the final total to 299/300 valid labels; the remaining invalid summary response
is excluded without another retry. Factoid has 8/50 correct (16%) and list has
3/50 correct (6%), so neither full type expansion is recommended. The formal
196-question common-valid summary analysis is complete: 1B accuracy is 12.2%,
and blind-P(True)/SE/normalized-NLL AUROC is 0.501/0.597/0.789.
P(True)-minus-SE is `-0.096 [-0.244,+0.060]` at 1B; its 1B-minus-4B change is
`-0.104 [-0.267,+0.059]`, so the continued point-estimate decline below 4B is
not itself statistically resolved. No recurring monitor was used.

### First local P(True)-Probe pass (2026-07-20)

`scripts/train_phase2_linear_probes.py` completed a CPU-only first pass over
the saved artifacts. It fitted train-only scalers and linear models for each
of the twelve independent `24/32/40/48 × TBG/SLT/LT` vectors, selected each
analysis on validation, and then emitted test metrics for the selected model
only. The ignored local output directory is
`analysis_outputs/bioasq_phase2_linear_probes_seed31` and contains the frozen
configuration, every train/validation candidate metric, and selected test
predictions.

| Analysis | Train target | Validation-selected feature | Test result |
| --- | --- | --- | --- |
| Hard-threshold even split, L2 logistic | train uncertainty threshold `1.877536e-06`; 1,568 high / 1,576 low | block 24, LT | AUROC 0.9026; AP 0.8948; Brier 0.1414 |
| Hard-threshold minimum-within-variance, L2 logistic | train threshold `0.468912`; 433 high / 2,711 low | block 40, LT | AUROC 0.7677; AP 0.3629; Brier 0.1436 |
| Ridge | continuous uncertainty | block 24, TBG | MAE 0.4378; Spearman 0.2146 |
| ElasticNet sensitivity check | continuous uncertainty | block 24, SLT; 391 valid test rows | MAE 0.2189; Spearman 0.6057 |

These are P(True)-fidelity results, not answer-correctness results.

The final P(True)-Probe is fixed as the **hard-threshold even split, L2
logistic regression, block 24 LT** model. It has the highest validation AUROC
and the lowest validation Brier score among the hard-threshold-even candidates.
The complete validation AUROC selection matrix was:

| Block | TBG | SLT | LT |
| --- | ---: | ---: | ---: |
| 24 | 0.8348 | 0.8578 | **0.8754** |
| 32 | 0.8125 | 0.8423 | 0.8630 |
| 40 | 0.8311 | 0.8129 | 0.8319 |
| 48 | 0.8074 | 0.7924 | 0.8400 |

TBG/LT rows each contain all 393 validation questions. SLT has 389 valid
answer positions and excludes the four short-answer cases, without imputation.
The frozen block-24 LT model subsequently achieved test AUROC 0.9026, AP
0.8948, and Brier 0.1414 against its train-derived hard P(True) target.

For a secondary **exploratory** UQ diagnostic only, each of the twelve already
train-fitted hard-threshold-even models was scored against Claude incorrect on
the valid-label test subset. This post-selection table is not used to alter the
frozen P(True)-fidelity choice above:

| Block | TBG | SLT | LT |
| --- | ---: | ---: | ---: |
| 24 | 0.7431 | **0.7653** | 0.7452 |
| 32 | 0.7186 | 0.7417 | 0.7521 |
| 40 | 0.7250 | 0.6916 | 0.7245 |
| 48 | 0.7379 | 0.6977 | 0.7492 |

These are Claude-correctness AUROCs, not P(True)-fidelity AUROCs. TBG/LT use
384 valid-label test questions; SLT uses 382 because two further answers lack
the penultimate answer-token position. The small spread confirms that this
diagnostic does not justify changing the P(True)-Probe layer/token after test.

### First local Accuracy-Probe pass (2026-07-20)

The same linear runner was rerun with the valid-label subset. Accuracy-Probe
selected block 24, LT on validation and achieved test AUROC **0.8058**, AP
**0.8884**, and Brier **0.2162** on 384 questions (252 Claude-incorrect).
The nine missing-label questions in each of validation/test and 54 in training
were excluded only from this track. On exactly those 384 test questions, direct
blind P(True) achieved AUROC 0.7900; the selected even hard-threshold
P(True)-Probe reached 0.7452, minimum-within-variance 0.6065, Ridge 0.5635,
and ElasticNet 0.6532 (382 rows because SLT is unavailable for two additional
valid-label answers). Thus this is preliminary evidence for a distinct
Accuracy-Probe, not evidence that P(True)-fidelity models recover correctness.

On the valid test labels (252/384 incorrect; AP prevalence 0.6563), the
following is the final fixed-score comparison. Most scores use all 384 rows;
ElasticNet uses 382 because SLT is missing for two answers. It includes the
frozen two Probes, the continuous-regression sensitivity check, and every
collected single-answer UQ; no high-temperature samples, NLI clustering, or
new generations were introduced. Brier is reported only for scores natively
in `[0,1]`; it remains a diagnostic against Claude risk, not a claim that a
non-Claude training target is calibrated for Claude correctness.

| Score | AUROC | AP | Brier | Notes |
| --- | ---: | ---: | ---: | --- |
| Accuracy-Probe, block 24 LT | **0.8058** | **0.8884** | **0.2162** | validation-selected L2 logistic classifier |
| Direct blind P(True) uncertainty | 0.7900 | 0.8383 | 0.5347 | strongest non-Probe discriminator; poorly calibrated to Claude risk |
| P(True)-Probe, even hard threshold, block 24 LT | 0.7452 | 0.8357 | 0.3177 | frozen P(True)-fidelity Probe; its Claude result is secondary |
| ElasticNet continuous P(True)-Probe, block 24 SLT | 0.6532 | 0.7945 | — | continuous P(True) fit; 382 valid rows; secondary correctness-UQ diagnostic |
| Verbalized confidence uncertainty | 0.6662 | 0.7426 | 0.5557 | 12 discrete score values |
| Mean token entropy | 0.5583 | 0.7096 | — | single-answer decoder statistic, not a probability |
| Normalized NLL | 0.5546 | 0.7077 | — | single-answer decoder statistic, not a probability |
| Max token entropy | 0.5357 | 0.6737 | — | 11 discrete score values, not a probability |
| Sequence NLL | 0.5304 | 0.6687 | — | single-answer decoder statistic, not a probability |

AUROC 0.9026 for the hard-even Probe is a different result: fidelity to its
binarized P(True) training target. The correctness-UQ comparison uses Claude
`incorrect`. On the exact 382 rows available to ElasticNet, the hard-even
Probe reaches AUROC 0.7456 / AP 0.8362 versus ElasticNet 0.6532 / 0.7945.
Thus the continuous regressor does not outperform the hard-even Probe as UQ.
The reproducible cross-target metrics are in
`analysis_outputs/bioasq_phase2_linear_probes_seed31_elasticnet_uq_auc_20260723/selected_p_true_models_correctness_metrics.csv`.
| Sample consistency exact | — | — | — | constant at 1.0 with one generation; not evaluable |

### Post-hoc provenance sidecar

The completed collection is accompanied by an append-only
`phase2_provenance_sidecar.json`. It preserves the original run metadata and
adds the resolved Hugging Face model snapshot, tokenizer content hashes,
Torch/Transformers runtime, collector source-bundle hash, and SHA-256 values
for every core generation/UQ/hidden-state artifact. The sidecar deliberately
excludes the separately submitted Claude batch, which receives its own batch
metadata and manifest. Creating or updating the sidecar never regenerates an
answer or changes a tensor.

## Purpose

Phase 1 established a protocol-correct BioASQ medical-QA UQ reference. Phase 2
asks a different question: can a probe of the source model's hidden states
recover either its direct blind P(True) uncertainty or the correctness of its
low-temperature answer at lower inference cost?

The post-Simpson thesis direction is broader than proving that one UQ method is
always preferable. The main line now asks **under which tested conditions
Semantic Entropy or blind P(True) performs better, and which mechanism may
explain the difference**. Answer length is the first controlled factor; the
semantic-equivalence judge is a conditional mechanism diagnostic.

The Probe study is a connected deployment-oriented branch. It asks whether a
simple hidden-state score can provide useful uncertainty at lower incremental
cost and whether a frozen Probe retains any signal under dataset/prompt shift.
P(True)-Probe fidelity, Accuracy-Probe correctness ranking, efficiency, and
PubMedQA transfer remain separate claims. The project does not return to the
discarded adaptive-token-selection phase.

## Two Probe Tracks

| Track | Supervision target | Primary question | Primary comparator |
| --- | --- | --- | --- |
| P(True)-Probe | `p_true_blind_uncertainty` from direct blind P(True) | Can hidden states approximate direct P(True) faithfully and more cheaply? | Direct blind P(True) |
| Accuracy-Probe | Claude binary label of the same low-temperature answer (`incorrect=1`, `correct=0`) | Can hidden states identify answers judged incorrect? | Direct blind P(True), P(True)-Probe, and available UQ scores |

The two tracks are deliberately separate. The P(True)-Probe is a fidelity and
cost study, not a claim that it exceeds its own direct target. The
Accuracy-Probe is an answer-correctness prediction task; it must not use Claude
labels, gold answers, exact answers, or ideal answers as input features.

## Dataset and Frozen Split

The Phase-2 source is the complete eligible BioASQ `training13b` corpus:
factoid, list, and summary only. Yes/no remains out of scope because its answer
format and evaluation target differ from the active free-form protocol.

The fixed manifest contains 3,930 questions:

| Split | Factoid | List | Summary | Total |
| --- | ---: | ---: | ---: |---:|
| Train | 1,280 | 837 | 1,027 | 3,144 |
| Validation | 160 | 105 | 128 | 393 |
| Test | 160 | 105 | 128 | 393 |

The observed Phase-1 1,000-question cohort is train-side only and carries
`phase1_reference=true`; validation and test contain no Phase-1 question.
Exact normalized duplicate questions are grouped before assignment. The source
SHA-256, reconstruction provenance, and realised counts are stored in the
ignored local split summary. See
`archehr_sebaseline/docs/phase2_bioasq_dataset_split.md`.

## Phase 2A — Full BioASQ Artifact Collection

Run the frozen manifest across train, validation, and test with the Phase-1
no-evidence, type-specific prompting protocol. The core collection is needed
before either probe can be trained.

For every question, retain:

- split ID, BioASQ type, prompt version, model revision, decoding settings, and
  source-data/split-manifest hashes;
- one low-temperature main answer, which is the single answer evaluated by both
  direct P(True) and Claude;
- direct blind P(True) and its uncertainty transformation, the P(True)-Probe
  target;
- question-level UQ fields available from the chosen run configuration, with
  their exact definitions and runtime cost; and
- complete failure/status fields, so a missing answer, score, or label cannot
  silently enter training.

Claude judging is a separate post-processing stage. It produces only the binary
correct/incorrect target for the corresponding low-temperature answer. It does
not change generation, direct P(True), or hidden-state artifacts.

### Approved efficiency comparison: ten high-temperature samples

The completed core run did **not** generate ten high-temperature samples.
That branch is now approved only as a separate efficiency/comparator package
on the 384 valid-labelled test questions. It supplies ten-sample normalized
NLL, discrete SE, and cluster count while measuring the extra sampling and NLI
cost separately from the saved main answer. It does not change either Probe,
the main answer, the frozen split, or either supervision target. The submitted
smoke/full jobs and cost contract are recorded in
`archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`.

### Initial feature contract: hidden states

The first collection saves representations from Gemma 3 12B's text transformer
blocks **24, 32, 40, and 48**. Layer 16 is deliberately excluded: the initial
probe focuses on mid-to-late representations, where the reviewed factuality
and uncertainty-probe literature provides the most relevant evidence. “Block
48” means the output of the forty-eighth text transformer block, not a
framework-dependent tuple offset; the extractor must record its exact
`hidden_states` index in the feature metadata.

At every selected block, save one 3,840-dimensional vector for each of these
positions in the original low-temperature answer transcript:

| Position | Definition | Role in the initial comparison |
| --- | --- | --- |
| `TBG` | final prompt token immediately before the first answer token | prompt/question-side control |
| `SLT` | penultimate generated answer token, immediately before the final answer token | answer-completion signal |
| `LT` | final generated answer token before EOS | main answer-content candidate |

The artifact is therefore 12 vectors per valid question, stored in `bfloat16`
with row ID, block name, token ID/index, answer-boundary indices, prompt
version, model revision, and tokenizer revision. A question whose generated
answer cannot supply a position (for example, too few answer tokens for
`SLT`) must be explicitly marked missing rather than shifted to another token.
For 3,930 questions this raw feature grid is about 0.36 GB before metadata.

Extract the states through a deterministic, no-cache forward pass over exactly
the stored prompt-plus-low-temperature-answer token sequence; do not infer
feature positions from the hidden-state output of `generate()`, and never
regenerate an answer for this step. The answer boundary and EOS convention
must be written into the run configuration before collection.

## Probe Training and Evaluation

The first pass is deliberately restricted to regularised *linear* probes. One
selected 3,840-dimensional hidden-state vector is the only input: there is no
layer/token concatenation, top-k feature selection, tree/boosting model, or
neural probe. This is a potential check, not an attempt to maximise a
within-dataset score with a flexible model.

### P(True)-Probe: two simple targets

The continuous target is `p_true_blind_uncertainty = 1 - P(True)`, so that a
larger score always means greater uncertainty/risk. (If reporting raw P(True),
its direction is inverted only for presentation.) The training split defines
each threshold once from this continuous target, before inspecting any
hidden-state feature. The resulting threshold and the realised class counts
are recorded and then shared by every layer/token candidate:

| Analysis | Training label/model | Prediction used as score |
| --- | --- | --- |
| Hard-threshold even split | Deterministic middle-rank split of train uncertainty into low=0/high=1, with tie handling recorded; L2 logistic regression | probability of high uncertainty |
| Hard-threshold minimum-within-variance | Train-only threshold minimizing the sum of within-group squared deviations, over valid non-empty splits; L2 logistic regression | probability of high uncertainty |
| Continuous regression | Ridge regression, plus ElasticNet as a sparse linear sensitivity check, trained on the continuous uncertainty target | predicted uncertainty (with any clipping and its rate reported) |

The two hard-label definitions are both first-pass experiments. Neither is
treated as the preferred definition before validation. The
continuous models answer the complementary question of whether the raw direct
P(True) target can be recovered without discretising it.

### Accuracy-Probe

Accuracy-Probe is a separate L2 logistic-regression binary classifier: Claude
`incorrect=1`, `correct=0`; its output is the probability that the same
low-temperature answer is incorrect. It is not a regression target and it
does not receive any Claude, gold-answer, or reference-answer content as an
input feature.

For every model, scaling and regularisation fitting use training rows only.
Each of the `24/32/40/48 × TBG/SLT/LT` vectors is evaluated independently;
SLT models exclude only the rows marked invalid for that position rather than
imputing a zero vector. Regularisation choices remain small and linear, and
are selected on validation rather than expanded into a broad model search.
A more complex probe is out of scope for this first pass unless the linear
results motivate a separately documented follow-up.

The experimental discipline is already fixed:

1. Develop feature extraction and probe choices using train only.
2. Select the final configuration using validation only, retaining a decision
   log of every comparison.
3. Freeze the feature contract and model configuration before reading the test
   outcome.
4. Evaluate once on test, overall and separately for factoid, list, and
   summary; do not pool types as though they were one homogeneous task.

The P(True)-Probe reports fidelity to direct blind P(True): continuous error
and rank agreement for Ridge/ElasticNet, plus discrimination/calibration of
the two hard-threshold targets. Its secondary usefulness for Claude correctness
may be reported, but it does not replace the Accuracy-Probe.

The Accuracy-Probe reports correctness discrimination and calibration against
the binary Claude label. Compare it with direct blind P(True), the P(True)-Probe
output, and each UQ score that was actually collected. Use the same held-out
questions and report prevalence, AUROC, AUPRC where informative, calibration,
and selective/rejection behaviour. Any optional ten-sample UQ result is a
separate comparator, not a training target.

## Cross-Dataset Test

The first external dataset is fixed as the official 500-question PubMedQA
PQA-L test subset. It passed local completeness, label-agreement, duplicate,
and exact-question-overlap checks. The active transfer path keeps the Gemma 3
12B source model fixed, supplies the official PubMed abstract context, requires
`yes/no/maybe + explanation`, and applies the already selected block-24/LT
P(True)-Probe and Accuracy-Probe. This compares two Probe models across
datasets; it is not an unsupported transfer of one coefficient vector across
different language-model hidden spaces.

For a genuine transfer result, do not fit probe parameters or tune thresholds
on the target dataset. If later adaptation is useful, report it as a distinct
fine-tuned transfer experiment, never as zero-shot generalisation. Preserve
target-dataset answer generation, direct P(True), and binary correctness labels
as separate artifacts and report answer-type/distribution differences.

Evaluate both frozen Probes against both PubMedQA targets. For the
P(True)-Probe's original target, retain the BioASQ-train even threshold exactly;
for Accuracy-Probe, define incorrect from mismatch between the required leading
decision and the official PubMedQA label. Also compare the separately named
blind P(True), verbalized confidence, sequence/normalized NLL,
and mean/max token entropy on the same questions. Preserve generated
explanations and `LONG_ANSWER` for a later separate Claude alignment stage,
without assuming that `LONG_ANSWER` is a manually curated high-quality
explanation. See
`archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`.

The question-only PubMedQA run is archived rather than used as an active
result because it omitted the article evidence that defines the official
decision. Its 25.0% accuracy and full provenance remain in
`archive_unused/docs/historical_results/pubmedqa_no_context_transfer_20260721.md`.

The active comparison is between two completed abstract-context prompts on the
same 500 IDs. Context v1 reached 59.0% accuracy and predicted
`yes/no/maybe = 237/100/163`. The Appendix-C v2 prompt adds the official narrow
label definitions while keeping the data, Gemma snapshot, generation settings,
block-24/LT Probes, BioASQ threshold, and evaluator unchanged. It reached 72.4%
accuracy and `298/164/38`, with 88 v1 errors corrected and 21 v1 correct
answers broken (`p=6.11e-11`). Macro-F1 improved from 0.5211 to 0.5659.

Against v2 decision error, frozen P(True)-Probe is strongest at 0.6839 AUROC /
0.4519 AP, followed by blind P(True) at 0.6490 / 0.4210,
verbalized confidence at 0.6402 / 0.4164, and frozen Accuracy-Probe at 0.5901 /
0.3916. P(True)-Probe has only 0.5899 AUROC against the v2 blind P(True)
frozen-threshold teacher target, far below its 0.9026 held-out BioASQ fidelity.
The transfer result is therefore cross-target correctness ranking, not strong
preservation of the source mapping.

The main v2 limitation is `maybe` recall: 12.7%, down from 43.6% in v1, while
`yes` and `no` recall improve to 84.1% and 72.8%. This is a documented
minority-class limitation of an otherwise clear overall improvement.
No further PubMedQA-specific error analysis or prompt tuning is planned. The
completed v2 result and its `maybe` limitation remain documented as final
frozen transfer evidence.

## Post-Simpson Follow-up Plan

### Stage 1 — complete the current Phase-2 Probe study

Finish the current Probe branch before launching a larger new experiment:

1. **Complete:** the 384-question incremental-efficiency benchmark retains
   latency, CUDA GPU time, extra generated tokens, throughput, and same-row
   correctness ranking for both Probes and the selected Phase-1 UQ
   comparators.
2. **Complete:** paired-bootstrap uncertainty for Accuracy-Probe versus direct
   blind P(True) uses all 384 test rows; the AUROC-difference interval includes
   zero.
3. **Archived exploration:** the validation-fitted two-Probe fusion produced
   only a small mixed AUROC/AP change and did not become a main model. Its
   result is retained under `archive_unused/`.

The already reported factoid/list/summary tables remain visible, but no new
type-specific thresholds, type-specific Probes, type-feature optimization, or
additional question-type decomposition is planned now.

### Stage 2 — controlled summary answer-length experiment — complete

This completed experiment tested whether changing summary-answer length alters
the relative correctness-ranking performance of blind P(True) and Semantic
Entropy. It is a paired prompt intervention on the same BioASQ summary
questions, not a new dataset comparison.

The current condition retains the accepted instruction:

```text
Write one concise biomedical paragraph that directly answers the question.
```

The shorter condition changes only the prompt by adding:

```text
Keep the answer to one or two brief but complete sentences.
Include only information needed to answer the question, without extra background.
```

There is no word/token cap and no hard truncation. `max_new_tokens` remains
192. Model, revision, question IDs, seeds, low- and high-temperature decoding
settings, sample count, blind-P(True) prompt, NLI model/rule, and correctness
judge remain unchanged. The prompt versions and exact IDs must be frozen
before inspecting the new outcomes.

The preferred paired cohort is the 123 valid-labelled summary questions in the
current 384-question efficiency benchmark. If its full artifacts pass health
checks, the accepted current-condition main answers, ten samples, blind
P(True), and NLI clusters are reused rather than regenerated. Only the shorter
condition requires new generation and judging.

This intervention is feasible but not assumed to force every answer into one
sentence. Existing Phase-2 summary main answers average about 80--83 words and
3.2--3.3 sentences. On validation, the shortest independent official ideal
answer has a median length of 32 words and 72.7% use at most two sentences.
These figures justify a one-or-two-sentence instruction while avoiding an
arbitrary numerical cutoff.

For both conditions report:

- realised main-answer and sampled-answer token/word/sentence distributions;
- compliance with the one-or-two-sentence instruction, without excluding
  non-compliant rows;
- answer correctness and error prevalence;
- blind P(True), discrete SE, cluster count, and normalized-NLL AUROC/AP;
- cluster count/size distribution and largest-cluster fraction; and
- paired-bootstrap confidence intervals.

Define the primary relative-performance quantities as:

```text
D_current = AUROC(blind P(True), current) - AUROC(SE, current)
D_short   = AUROC(blind P(True), short)   - AUROC(SE, short)
length effect = D_short - D_current
```

Bootstrap the paired question IDs across both conditions. Because correctness
can change after the prompt intervention, preserve the joint correctness
categories where stratification is needed. The primary analysis remains by
assigned prompt condition; do not condition it on observed length compliance
or on answers being correct in both conditions.

Smoke job `5781225` and dependency-gated full job `5781246` completed `0:0`
and passed their artifact health checks. The short prompt reduced mean
main-answer length from 82.88 to 52.11 words and from 3.25 to 2.00 sentences;
all 123 short main answers and all 1,230 short samples satisfied the
one-or-two-sentence instruction after an abbreviation-aware derived-field
repair. No generation or UQ value was changed by that repair.

The final correctness/UQ comparison uses 121 IDs with valid Claude labels in
both conditions; two short labels remained blank after the initial batch and
two same-configuration retries. Both conditions were correct on 56/121, with
eight current-only and eight short-only correct answers. Blind P(True) AUROC
changed from 0.7948 to 0.8040; discrete SE from 0.5782 to 0.5468; cluster count
from 0.5805 to 0.5451; and ten-sample normalized NLL from 0.7533 to 0.7511.

The pre-declared quantities are:

```text
D_current = 0.2166, 95% CI [0.1119, 0.3158]
D_short   = 0.2571, 95% CI [0.1542, 0.3562]
length effect = +0.0405, 95% CI [-0.0713, 0.1531]
```

The intervention clearly changed length, but shortening did not recover SE
relative to blind P(True). The relative-performance change is not statistically
resolved, and its point estimate slightly widens the P(True)-over-SE gap. See
`archehr_sebaseline/docs/summary_length_intervention.md`.

No additional answer-length intervention is planned. The two-seed Phase-1
factoid/list/summary results already provide a natural observational length
contrast using the samples from which SE is calculated. Mean sampled-answer
lengths are approximately `8/35/111` generated tokens. Corresponding
blind-P(True)-minus-discrete-SE AUROC gaps are approximately `+0.075`,
`0.000/-0.029`, and `+0.243/+0.231`: SE is competitive on medium-length
structured lists but fails on long summaries. This is descriptive rather than
causal because answer structure, prompt, and NLI rule also vary by type. The
replicated cross-type pattern plus the completed paired shortening result is
sufficient for the answer-length/UQ summary; see
`archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md`.

### Stage 3 — conditional semantic-judge bottleneck diagnostic

This stage does not propose Claude-based SE as a deployment method. Its only
question is whether the current biomedical NLI semantic-equivalence judge is a
major contributor to weak whole-answer SE behaviour for longer summaries.

Reuse fixed generations from Stage 2. The first bounded diagnostic uses only
the original longer-answer condition: 48 label-balanced questions (24 correct
and 24 incorrect under the frozen Claude correctness labels), selected by
SHA-256 ranking with seed `20260726`. This is 480 saved answers and 48 batch
requests, not all 1,230 answers in either condition. It is intended to detect a
large mechanism signal and must not expand automatically when inconclusive.

Recluster those fixed answer sets with:

1. the accepted PubMedBERT NLI pipeline; and
2. a frozen Claude semantic-equivalence rubric.

No answer is regenerated and the correctness labels are not changed. Retain a
small manually reviewed answer-pair subset to check false splits and false
merges; otherwise an apparent Claude-SE AUROC gain could reflect shared-model
bias with the Claude correctness judge rather than better equivalence labels.

Interpret the diagnostic conservatively:

- better manually supported Claude equivalence plus recovery of long-answer
  SE supports current NLI as an important bottleneck;
- better equivalence labels without SE recovery means NLI has errors but does
  not explain the main UQ gap;
- similar clusters with persistently weak SE point instead toward whole-answer
  uncertainty, the answer distribution, or systematic model errors.

Diagnosing an NLI bottleneck is a sufficient project outcome. This stage does
not require inventing or implementing a new NLI method.

The frozen bounded protocol is recorded in
`archehr_sebaseline/docs/summary_clustering_diagnostic.md`.

The quantitative diagnostic is complete on 47/48 valid Claude-clustered
questions (23 correct, 24 incorrect); the one repeatedly truncated row is
excluded without further retry. Claude-clustered SE reached AUROC 0.727 versus
0.579 under the accepted NLI clusters, a `+0.149` change with paired-bootstrap
95% CI `[-0.016,+0.313]`. Blind P(True) remained higher at 0.825. Claude
produced many more clusters (mean 3.98 versus 1.23), with 821
NLI-same/Claude-different answer pairs versus 41 in the opposite direction.
This initial bounded result motivated the already authorized 96-question
extension; it is not the final mechanism estimate.

Because 47 complete cases remain small for that mechanism inference, one
nested extension is authorized to a 96-question label-balanced target. It
reuses the original 48 questions and calls Claude only for 48 new questions;
batch `msgbatch_015tEX5qdiBHpb2sUUME4gDW` completed with 48/48 API successes.
Two new rows hit `max_tokens`; with the invalid base row excluded, the final
analysis contains 93 complete cases (46 correct, 47 incorrect). The same
answers, rubric, pair order, and clustering conversion are frozen. This is the
terminal scale expansion: do not extend to the 1,000-question Phase-1 corpus.

On 93 complete cases, Claude-clustered SE reaches AUROC 0.781 versus 0.595 for
NLI-SE, a `+0.187` change with paired-bootstrap 95% CI
`[+0.080,+0.291]`. Blind P(True) is 0.815. Its gap over SE falls from
`+0.220 [+0.110,+0.325]` under NLI clustering to
`+0.034 [-0.079,+0.147]` under Claude clustering. The cluster structure
replicates strong NLI merging: mean 1.19 versus 3.88 clusters, with 1,673
NLI-same/Claude-different pairs and 50 in the opposite direction.

The scientific target remains diagnosis: determine whether an alternative
semantic-equivalence judge materially changes long-summary SE and therefore
implicates NLI difficulty. Claude clustering is too expensive for the intended
practical pipeline and must not be presented as the proposed improvement.
The completed stratified 24-pair human review agrees with original NLI on
9/24 pairs, Claude clustering on 19/24, and Claude direct judgement on 17/24.
Original NLI made five false merges and ten false splits; Claude clustering
made no false merge and five false splits. This supports NLI as an important
long-answer SE error source while retaining Claude's conservative
over-splitting and non-deployability as limitations.

### Stage 4 — SE/P(True) operating-regime extension — complete

Model scale is not part of the Probe branch. If time and compute remain after
Stages 1--3, a small within-family comparison may test whether model capability
changes the relative behaviour of direct blind P(True) and SE. It must compare
the two original UQ methods under a frozen protocol and report answer accuracy,
answer length, AUROC/AP, their paired difference, and inference cost.

Do not train model-scale Probes, construct a model-family grid, or make this
optional extension a prerequisite for the thesis contribution.

The current generator is `google/gemma-3-12b-it`. Official Gemma 3 core
instruction-tuned checkpoints exist at:

| Size | Hugging Face checkpoint | Release/context note |
| ---: | --- | --- |
| 270M | `google/gemma-3-270m-it` | later 2025-08 release; text-only, 32K context |
| 1B | `google/gemma-3-1b-it` | original scale release; text-only, 32K context |
| 4B | `google/gemma-3-4b-it` | original scale release; text/image, 128K context |
| 12B | `google/gemma-3-12b-it` | current experimental anchor; text/image, 128K context |
| 27B | `google/gemma-3-27b-it` | original scale release; text/image, 128K context |

The completed comparison uses the 1B, 4B, and 12B original-release checkpoints.
The 1B run used a staged design because the 4B accuracy was already low: 20.3%
overall, 23.5% on factoid, 10.7% on list, and 27.9% on summary. The 270M and
27B checkpoints are not authorized.

The first added scale point is frozen as Gemma 3 4B, seed 31, aligned exactly
to the accepted Phase-1 1,000-question seed-31 cohort. All Phase-1 prompts,
type quotas, generation/sampling settings, NLI rules, and original UQ methods
remain unchanged; no Probe is trained. Smoke job `5785064` and dependency-gated
full job `5785065` completed successfully. The full run passed Level-4 health
and exact 480/320/200 pre/post cohort alignment. Initial Claude correctness
batch `msgbatch_01AbxKWPN4pdgYoh3CF2V42G` and its single bounded retry produced
997/1,000 valid labels. The final paired comparison uses 990 questions valid
for both 4B and 12B. Overall P(True)-minus-SE AUROC changes from
`+0.047 [+0.017,+0.076]` at 12B to `-0.009 [-0.051,+0.031]` at 4B; the
4B-minus-12B change is `-0.056 [-0.103,-0.010]`. On summary it changes from
`+0.243 [+0.173,+0.313]` to `+0.005 [-0.085,+0.094]`, a
`-0.238 [-0.347,-0.129]` scale effect. Lower capability therefore removes
blind P(True)'s advantage; SE itself is not strong on 4B summary, where
normalized NLL is best. See
`archehr_sebaseline/docs/gemma3_model_scale_experiment.md`.

The staged Gemma 3 1B extension keeps seed 31 and changes only the generator
checkpoint to `google/gemma-3-1b-it`. It preserves the accepted Phase-1
no-evidence prompts, temperature/top-p/top-k settings, ten SE samples, blind
P(True), type-specific NLI rules, correctness judge, and existing 4B/12B
cohort mapping. No Probe is trained.

Execution is deliberately staged:

1. Run a six-question smoke with two exactly aligned questions per type and
   the complete generation/UQ path. The staged collection may proceed only
   through a Slurm `afterok` dependency after the smoke exits successfully and
   passes its artifact health and alignment checks. Do not start a continuous
   monitor.
2. Freeze a 300-question 1B cohort before inspecting any 1B outcome: all 200
   summary questions from the aligned 4B/12B cohort, plus deterministic
   50-question subsets from the aligned 480 factoid and 320 list questions.
3. Treat the aligned 200-summary result as the planned formal 1B/4B/12B
   comparison of answer accuracy, blind P(True), discrete SE, normalized NLL,
   and the P(True)-minus-SE AUROC gap.
4. Treat the 50 factoid and 50 list questions only as feasibility gates. Report
   their accuracy and valid correct/incorrect label counts first; do not treat
   their small-sample UQ metrics as final type-level scale estimates.
5. After those accuracy results are reviewed, decide separately whether either
   type has enough non-floor correctness signal to justify completing its full
   aligned cohort. The 50-question runs do not automatically authorize the
   remaining 430 factoid or 270 list questions.

The completed common-valid summary result shows P(True)-minus-SE AUROC
declining from `+0.243` at 12B to `+0.008` at 4B and `-0.096` at 1B. The
1B-minus-4B change is `-0.104 [-0.267,+0.059]`, so the below-4B continuation
is a point-estimate trend rather than a resolved incremental effect. The
factoid/list gates returned only 8/50 and 3/50 correct; neither full expansion
is recommended.

Official sources:

- https://ai.google.dev/gemma/docs/get_started
- https://ai.google.dev/gemma/docs/core/model_card_3
- https://ai.google.dev/gemma/docs/releases

## Milestones

1. **Complete — data split:** freeze the 3,930-question manifest and its
   provenance.
2. **Complete — collection:** the `24/32/40/48 × TBG/SLT/LT` extractor and
   one-answer run contract produced the full accepted BioASQ artifacts.
3. **Complete — full collection:** generated the one-answer core
   artifacts, direct blind P(True), selected UQ fields, and hidden states
   according to that contract.
4. **Complete — label collection:** obtained and validated binary Claude labels for the
   low-temperature answers, respecting the frozen splits.
5. **Complete — Probe development:** trained/selected P(True)-Probe and Accuracy-Probe on
   train/validation.
6. **Complete — final in-domain evaluation:** one frozen BioASQ test evaluation
   with type-stratified performance comparisons.
7. **Complete — initial transfer evaluation:** ran both frozen Probes on the
   PubMedQA question-only diagnostic and official-context v1 condition; the
   question-only result is now archived.
8. **Complete — PubMedQA prompt improvement:** added the official Appendix-C
   label definitions once and completed context v2 without changing either
   Probe or tuning on PubMedQA metrics.
9. **Complete — in-domain efficiency comparison:** measured the two
   shared-replay Probes against selected Phase-1 UQ methods on the 384 labelled
   test questions; replacement job 5773786 completed and passed its health
   check after the node-environment failure from job 5761275 was archived.
10. **Complete — Phase-2 statistical completion:** the paired Accuracy-Probe
    versus blind-P(True) bootstrap is recorded. The exploratory two-Probe
    fusion is archived and no type-aware Probe optimization was started.
11. **Complete — summary length intervention:** the short prompt changed
    realised length but did not recover SE relative to blind P(True); the
    paired-bootstrap length-effect interval includes zero.
12. **Complete — clustering mechanism and human review:** 93/96 complete
    cases show a bootstrap-supported Claude-SE recovery and close about 85% of
    the original P(True)-minus-SE point-estimate gap. The stratified 24-pair
    review agrees with Claude clustering on 19/24 pairs versus 9/24 for NLI.
    Claude remains a diagnostic rather than a deployable improvement.
13. **Complete — SE/P(True) model-capability check:** the aligned 4B/12B
    comparison shows a significant reduction in P(True)'s relative advantage
    at 4B, driven most clearly by summary questions. Do not train
    scale-specific Probes or interpret this as SE improving at smaller scale.
14. **Complete — staged Gemma 3 1B extension:** smoke and staged jobs
    completed, 299/300 correctness labels are valid, and the formal
    196-question common-valid summary analysis is complete. Factoid/list
    accuracy is 8/50 and 3/50, so do not expand either full type cohort. No
    1B PubMedQA follow-up is planned.
15. **Complete — PubMedQA-v2 Semantic Entropy:** jobs 5921808/5921809
    completed `0:0`; the formal run retained all 500 accepted v2 prompts and
    official-decision error labels, produced 5,000 `T=1.0` answers, and passed
    provenance/health checks. Discrete SE obtained 0.5884 AUROC / 0.3674 AP;
    444/500 questions formed one semantic cluster. No v2 main answer, Probe,
    P(True), calibration, judging, or prompt selection was rerun.

## Non-negotiable Boundaries

- Do not use P(True)-10; direct blind P(True) is the only P(True) target.
- Do not train an unsupported multi-target SE/NLL/P(True) probe.
- Do not let answers, hidden states, P(True), UQ rows, or Claude labels cross
  their manifest split.
- Keep the completed PubMedQA-v2 high-temperature SE addition separate from
  the original single-answer transfer; do not retroactively describe it as
  part of the frozen-Probe collection run.
- Do not alter the initial `24/32/40/48 × TBG/SLT/LT` feature grid or replace
  the linear baseline after test outcomes are visible; any expansion is a new
  experiment.
- Do not impose a word/token cap or hard decoding truncation in the summary
  length intervention; only the frozen natural-language prompt addition may
  differ.
- Do not describe Claude reclustering as a stronger production UQ method. It is
  a fixed-generation diagnostic of whether NLI is a major bottleneck.
- Do not train Probes for the optional model-scale comparison.
- Label analyses motivated by already inspected test outcomes as exploratory;
  do not use them to replace the frozen primary Probe results.

## Historical Context

Phase-1 results and the prior SE engineering path remain documented in
`archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md`,
`archehr_sebaseline/docs/bioasq_medical_uq_protocol.md`, and the archived
low-usability reports. They provide comparators and provenance, not the active
Phase-2 work plan.
