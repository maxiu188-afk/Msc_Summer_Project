# Phase 2 Probe Research Plan

Last updated: 2026-07-19

## Execution Status (2026-07-19)

The single-answer artifact collector has passed local unit/CLI/shell validation
and a server-side interface preflight. The Isambard Gemma 3 environment reports
48 text transformer blocks, hidden size 3,840, and a forward output with a
`hidden_states` field. The frozen manifest was copied to the server with
SHA-256 `d11953a5e6973809b2d024a9ad456098ec252db8290e73109234a1f1ac7b6433`.

- Smoke job **5719092**: one manifest question from each of train, validation,
  and test; submitted and pending scheduler priority. Its health check must
  verify P(True) rows and `[N, 4, 3, 3840]` hidden-state tensors.
- Full job **5719110**: all 3,930 questions, submitted with
  `afterok:5719092`; it will enter the queue only if the smoke succeeds.

No generation, UQ, hidden-state, Claude-label, or probe performance result is
available yet. These job IDs and status are operational provenance, not
evidence of a completed experiment.

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

The initial probe is intentionally simple: one selected hidden-state vector is
the input to a regularised linear logistic model, producing a scalar risk
probability. It is a binary classifier in form, not a deep hidden-state model.
Use the same model family for both tracks, but fit separate models and select
the block/position independently on validation:

| Track | Model output | Training target |
| --- | --- | --- |
| P(True)-Probe | predicted uncertainty in `[0, 1]` | direct `p_true_blind_uncertainty` as a soft binary target |
| Accuracy-Probe | predicted probability that the answer is incorrect | Claude `incorrect=1`, `correct=0` hard binary target |

Binary cross-entropy supports both: for P(True)-Probe, the direct P(True)
uncertainty is a soft probability label; for Accuracy-Probe, the Claude label
is 0 or 1. Feature normalisation and any regularisation strength are fitted on
training data only. A more complex probe is out of scope for the first pass
unless the linear baseline demonstrably fails and a separately documented
follow-up is approved.

The experimental discipline is already fixed:

1. Develop feature extraction and probe choices using train only.
2. Select the final configuration using validation only, retaining a decision
   log of every comparison.
3. Freeze the feature contract and model configuration before reading the test
   outcome.
4. Evaluate once on test, overall and separately for factoid, list, and
   summary; do not pool types as though they were one homogeneous task.

The P(True)-Probe reports fidelity to direct blind P(True) (error, rank
agreement, calibration where applicable) and its inference/storage cost. Its
secondary usefulness for Claude correctness may be reported, but it does not
replace the Accuracy-Probe.

The Accuracy-Probe reports correctness discrimination and calibration against
the binary Claude label. Compare it with direct blind P(True), the P(True)-Probe
output, and each UQ score that was actually collected. Use the same held-out
questions and report prevalence, AUROC, AUPRC where informative, calibration,
and selective/rejection behaviour. Any optional ten-sample UQ result is a
separate comparator, not a training target.

## Cross-Dataset Test

After the in-domain BioASQ test is frozen and reported, evaluate the frozen
probe and feature contract on one or more separately selected medical-QA
datasets. The candidate dataset, label mapping, prompt compatibility, and
license/access checks are still TBD.

For a genuine transfer result, do not fit probe parameters or tune thresholds
on the target dataset. If later adaptation is useful, report it as a distinct
fine-tuned transfer experiment, never as zero-shot generalisation. Preserve
target-dataset answer generation, direct P(True), and binary correctness labels
as separate artifacts and report answer-type/distribution differences.

## Milestones

1. **Complete — data split:** freeze the 3,930-question manifest and its
   provenance.
2. **In progress — collection:** the `24/32/40/48 × TBG/SLT/LT` extractor and
   one-answer run contract are versioned and server-submitted behind a
   three-question smoke; separately decide whether to add the ten-sample UQ
   package.
3. **Full collection:** after smoke success, generate the one-answer core
   artifacts, direct blind P(True), selected UQ fields, and hidden states
   according to that contract.
4. **Label collection:** obtain and validate binary Claude labels for the
   low-temperature answers, respecting the frozen splits.
5. **Probe development:** train/select P(True)-Probe and Accuracy-Probe on
   train/validation.
6. **Final in-domain evaluation:** one frozen BioASQ test evaluation with
   type-stratified comparisons and cost accounting.
7. **Transfer evaluation:** run the unchanged probe on a separately documented
   dataset, then consider any adaptation as a new experiment.

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
