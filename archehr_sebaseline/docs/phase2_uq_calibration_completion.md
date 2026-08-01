# Phase-2 UQ Calibration and Selective-Prediction Completion

Last updated: 2026-08-01

## Decision and status

After the supervisor discussion, the project is closing the remaining UQ
evaluation gap before freezing experiments for thesis writing. The fixed scope
is:

1. collect the missing ten-sample Semantic Entropy scores on the labelled
   Phase-2 validation split;
2. fit scalar calibration mappings on validation and evaluate them once on the
   existing Phase-2 test split;
3. report calibration, selective-prediction, type-stratified, and paired
   uncertainty results;
4. audit a fixed blinded sample of Claude correctness labels; and
5. apply the BioASQ calibration mappings unchanged to PubMedQA v2 as a
   zero-refit transfer diagnostic.

Implementation and local tests are in progress. No calibration result has been
inspected yet, and no formal validation-SE job has been accepted.

## Common target and methods

The common deployment target is Claude `incorrect=1` for the saved
low-temperature main answer.

| Role | Method | Score before calibration |
| --- | --- | --- |
| Primary | Discrete Semantic Entropy | ten-sample PubMedBERT-NLI cluster entropy |
| Primary | Blind P(True) | `1 - P(True)` from the fixed continuation prompt |
| Primary | Accuracy-Probe | frozen block-24/LT probability of Claude incorrect |
| Auxiliary | P(True)-Probe | frozen block-24/LT probability of high P(True)-derived uncertainty |

P(True)-Probe remains an auxiliary cross-target method for correctness. Its
separate fidelity to the train-derived P(True) target is not redefined by this
analysis.

## Frozen splits

| Split | Valid correctness labels | Factoid | List | Summary | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 384 | 157 | 104 | 123 | 265 |
| Test | 384 | 158 | 103 | 123 | 252 |

The test split and completed job `5773786` remain evaluation-only. The missing
validation SE scores will be collected with the same Gemma 3 12B revision,
prompt, ten `T=1.0` samples, seed 31, token settings, and PubMedBERT-NLI
clustering contract used by `5773786`. Existing validation main answers,
correctness labels, P(True), hidden states, and frozen Probe scores are reused.
No answer judging is repeated.

## Calibration protocol

For each fixed score, fit exactly one scalar logistic mapping on validation:

```text
raw uncertainty -> validation z-score -> logistic P(incorrect)
```

The logistic regularization constant is fixed at `C=1e6`; a positive slope is
required. A negative slope is a protocol failure and must not trigger a
post-hoc direction flip. The mapping is then applied unchanged to test.

Primary test outputs are:

- Brier score and Brier skill relative to the test-prevalence constant;
- log loss;
- 10-bin equal-frequency ECE and reliability bins;
- AUROC and average precision from the unchanged raw ranking score;
- retained error risk over coverage 0.50--1.00 and AURAC over that range;
- risk at 0.80, 0.90, and 0.95 coverage; and
- 20,000 paired test-row bootstrap intervals for the three primary methods.

Question-type results use the same global validation-fitted calibrator. No
type-specific calibration model, Probe, layer, token, or threshold is fitted.
Raw SE is not a probability and therefore receives no uncalibrated Brier/ECE
claim. Native `[0,1]` P(True)/Probe scores are retained as diagnostics.

## Correctness-label audit

The frozen audit sample contains 90 test answers: 30 drawn uniformly without
replacement within each of factoid, list, and summary using seed `20260801`.
The reviewer sees the question, appropriate BioASQ reference, and model answer,
but not the example ID, Claude label, or UQ scores. Allowed labels are
`correct`, `incorrect`, and `unsure`.

Because sampling is balanced by question type rather than proportional to the
test composition, the analysis reports both within-type agreement and a
design-weighted overall agreement. This is a correctness-target audit, not a
new UQ model-selection set.

Local ignored artifacts:

```text
analysis_outputs/bioasq_phase2_correctness_audit_20260801/
  correctness_audit_blinded.csv
  correctness_audit_key.csv
  correctness_audit_manifest.json
```

## PubMedQA calibration transfer

After BioASQ validation fitting, the blind-P(True), Accuracy-Probe, and
P(True)-Probe mappings are applied unchanged to the existing 500-question
PubMedQA Appendix-C context-v2 scores. PubMedQA labels, scores, prevalence, or
metrics do not fit, select, or modify a mapping. Report native and transferred
Brier, Brier skill, log loss, ECE, AUROC, and AP.

Semantic Entropy is absent from this transfer because PubMedQA used a
single-answer protocol. This remains a protocol boundary rather than a negative
SE result.

## Stop boundary

This completion does not authorize:

- fitting or selecting anything on the BioASQ test split;
- PubMedQA target recalibration;
- new Probe architectures, fusion, type-specific heads, or hidden-state search;
- new model sizes, prompt tuning, or expanded Claude clustering; or
- regeneration of the completed Phase-2 test benchmark.

After the fixed analyses and audit are complete, update the thesis-oriented
result synthesis and freeze experiments unless a separate supervisor-approved
idea is pursued in parallel.
