# Phase 2 Probe Research Plan

Last updated: 2026-07-23

## Execution Status (2026-07-20)

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

This is no longer an SE-baseline stage. Semantic Entropy remains an optional
comparison score, particularly for list questions, but it is not the organising
target of this phase.

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

### Deferred UQ Decision: ten high-temperature samples

The core run does **not** assume that ten high-temperature samples will be
generated. That costly branch is required for multi-sample scores such as SE,
cluster count, and sampled-answer disagreement, but is not required to define
or train the two probes. Decide later whether to run it as an optional UQ
comparison package. If approved, it must use the same frozen split and report
its additional compute cost separately; it must not change the main answer or
either supervision target.

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
minority-class limitation of an otherwise clear overall improvement. Further
work is fixed error analysis, not repeated prompt tuning.

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
6. **Complete — final in-domain evaluation:** one frozen BioASQ test evaluation with
   type-stratified comparisons and cost accounting.
7. **Complete — initial transfer evaluation:** ran both frozen Probes on the
   PubMedQA question-only diagnostic and official-context v1 condition; the
   question-only result is now archived.
8. **Complete — PubMedQA prompt improvement:** added the official Appendix-C
   label definitions once and completed context v2 without changing either
   Probe or tuning on PubMedQA metrics.
9. **Next — fixed error analysis:** inspect the completed v2 error set,
   especially official `maybe` regressions, without changing the frozen result.

## Non-negotiable Boundaries

- Do not use P(True)-10; direct blind P(True) is the only P(True) target.
- Do not train an unsupported multi-target SE/NLL/P(True) probe.
- Do not let answers, hidden states, P(True), UQ rows, or Claude labels cross
  their manifest split.
- Do not present optional high-temperature UQ as part of the core run unless it
  was actually approved and generated.
- Do not alter the initial `24/32/40/48 × TBG/SLT/LT` feature grid or replace
  the linear baseline after test outcomes are visible; any expansion is a new
  experiment.

## Historical Context

Phase-1 results and the prior SE engineering path remain documented in
`archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md`,
`archehr_sebaseline/docs/bioasq_medical_uq_protocol.md`, and the archived
low-usability reports. They provide comparators and provenance, not the active
Phase-2 work plan.
