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

The initial implementation is frozen at commit `f84836f`; all 128 local tests
passed before submission.
Validation-SE smoke job `5863759` and full job `5863760` were submitted on
2026-08-01 with dependency `afterok:5863759`. No SE calibration result has been
inspected, and no formal validation-SE job has been accepted.

While Isambard is unavailable, the predeclared non-SE subset has been completed
locally from existing artifacts. The accepted interim analysis includes blind
P(True), Accuracy-Probe, P(True)-Probe, selective prediction, type
stratification, paired bootstrap, and zero-refit PubMedQA calibration transfer.
It is explicitly marked `partial_without_semantic_entropy`; no SE result is
imputed or inferred.

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

## Interim non-SE results

### BioASQ test calibration

The three scalar mappings were fitted on all 384 valid-labelled validation
questions and applied unchanged to the 384 valid-labelled test questions.

| Method | AUROC | Calibrated Brier | Brier skill | Log loss | 10-bin ECE | AURAC 0.5--1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy-Probe | **0.8058** | **0.1776** | **0.2125** | **0.5386** | **0.0891** | **0.2780** |
| Blind P(True) | 0.7900 | 0.2156 | 0.0444 | 0.6174 | 0.1922 | 0.2909 |
| P(True)-Probe | 0.7452 | 0.2001 | 0.1131 | 0.5860 | 0.0984 | 0.2905 |

The test-prevalence constant has Brier `0.2256`. Before validation calibration,
native Brier was `0.2162` for Accuracy-Probe, `0.5347` for blind P(True), and
`0.3177` for P(True)-Probe. Their corresponding native ECE values were
`0.1921`, `0.5392`, and `0.3004`. Blind P(True) therefore has strong ranking but
is not a usable answer-error probability without a separately fitted mapping.

Accuracy-Probe minus blind P(True), with 20,000 paired test-row resamples:

| Metric | Difference | 95% CI | Better direction |
| --- | ---: | ---: | --- |
| AUROC | +0.01584 | [-0.03850, +0.07163] | higher |
| Brier | -0.03794 | [-0.06050, -0.01541] | lower |
| Log loss | -0.07884 | [-0.13526, -0.02177] | lower |
| AURAC 0.5--1.0 | -0.01294 | [-0.02322, -0.00316] | lower |

Thus the earlier AUROC comparison remains unresolved, but Accuracy-Probe has a
bootstrap-supported advantage in calibrated probability quality and selective
prediction under this fixed protocol.

At 80% coverage, retained error risk is `0.5877` for Accuracy-Probe, `0.6006`
for blind P(True), and `0.5942` for P(True)-Probe, versus `0.6563` at full
coverage.

### Answer-form calibration

One global validation-fitted mapping was used for every type.

| Method | Factoid Brier skill | List Brier skill | Summary Brier skill |
| --- | ---: | ---: | ---: |
| Accuracy-Probe | +0.1553 | -0.1344 | +0.1613 |
| Blind P(True) | -0.0036 | -0.3834 | -0.0428 |
| P(True)-Probe | +0.0571 | -0.2365 | +0.0326 |

List error prevalence is `0.9029`, compared with `0.5886` for factoid and
`0.5366` for summary. The pooled calibrator does not beat a type-prevalence
constant on list for any method. This extends the answer-form conclusion from
ranking to calibration: global probability mappings can hide large type-level
base-rate shifts. No type-specific calibrator is fitted post hoc.

### PubMedQA zero-refit transfer

| Method | Native Brier | BioASQ-calibrated Brier | Transferred Brier skill | Transferred ECE |
| --- | ---: | ---: | ---: | ---: |
| Blind P(True) | 0.2659 | 0.3437 | -0.7198 | 0.3855 |
| Accuracy-Probe | 0.3815 | 0.3593 | -0.7979 | 0.3526 |
| P(True)-Probe | 0.3127 | 0.3511 | -0.7572 | 0.4067 |

All three transferred mappings are worse than the PubMedQA prevalence-only
Brier baseline `0.1998`. Accuracy-Probe improves slightly relative to its own
native Brier but remains poorly calibrated; blind P(True) and P(True)-Probe
worsen. Ranking can retain some cross-dataset signal while source-fitted error
probabilities fail under the dataset, prompt, answer-space, and error-prevalence
shift. No PubMedQA label was used to repair this failure.

Local ignored result snapshot:

```text
analysis_outputs/bioasq_phase2_uq_calibration_nonse_20260801/
  metrics_overall.csv
  metrics_by_type.csv
  reliability_bins.csv
  reliability_diagram.svg
  risk_coverage.csv
  risk_coverage.svg
  paired_bootstrap.csv
  pubmedqa_transfer_metrics.csv
  summary.json
```

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
