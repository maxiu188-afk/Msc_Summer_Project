# Phase-2 UQ Calibration and Selective-Prediction Completion

Last updated: 2026-08-27

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

The initial implementation was frozen at commit `f84836f`; all 128 local tests
passed before submission. Validation-SE smoke job `5863759` completed `0:0` in
`00:04:18`, and dependent full job `5863760` completed `0:0` in `04:01:10` on
2026-08-01. Both passed the batch health check. The formal output contains all
384 expected rows and no missing SE values. The complete offline analysis and
the fixed correctness-label audit have now both finished. The thesis-closing UQ
evaluation is complete; no additional experiment is implied by this document.
A separately authorized saved-artifact supplement later tested whether
validation-selected selective-prediction thresholds preserve their operating
points on test. That offline supplement is complete and does not reopen model
generation, Probe training, calibration, or any other experimental scope.
On 2026-08-27, a further authorized saved-artifact supplement added the
already-collected `10-sample normalized NLL` to the BioASQ probability-quality
comparison. It uses the same validation-only calibration and frozen test
evaluation, requires no answer generation or judging, and does not add NLL to
the separate validation-fixed operating-point experiment.

## Common target and methods

The common deployment target is Claude `incorrect=1` for the saved
low-temperature main answer.

| Role | Method | Score before calibration |
| --- | --- | --- |
| Primary | Discrete Semantic Entropy | ten-sample PubMedBERT-NLI cluster entropy |
| Primary | Blind P(True) | `1 - P(True)` from the fixed continuation prompt |
| Primary | Accuracy-Probe | frozen block-24/LT probability of Claude incorrect |
| Auxiliary | P(True)-Probe | frozen block-24/LT probability of high P(True)-derived uncertainty |
| Auxiliary | 10-sample normalized NLL | mean per-token NLL over the ten `T=1.0` samples |

P(True)-Probe remains an auxiliary cross-target method for correctness. Its
separate fidelity to the train-derived P(True) target is not redefined by this
analysis. Ten-sample normalized NLL is an auxiliary comparator so the
probability-quality table covers the same sampled token baseline already
reported in the Phase-2 ranking, efficiency, and model-scale sections. It also
restores NLL-family coverage consistent with the cross-dataset comparison while
preserving the distinction from PubMedQA's single-answer NLL.

## Frozen splits

| Split | Valid correctness labels | Factoid | List | Summary | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 384 | 157 | 104 | 123 | 265 |
| Test | 384 | 158 | 103 | 123 | 252 |

The test split and completed job `5773786` remain evaluation-only. The
validation SE scores were collected with the same Gemma 3 12B revision,
prompt, ten `T=1.0` samples, seed 31, token settings, and PubMedBERT-NLI
clustering contract used by `5773786`. Existing validation main answers,
correctness labels, P(True), hidden states, and frozen Probe scores are reused.
No answer judging is repeated.

The validation and test UQ files have SHA-256
`cc86e63cd1197d3b896fc8e14808c2fa3be308ce96961b40a2b3d5e0962d6245` and
`181e5a77e3a7ba13a61577b6edb1d19c25539cdaf9a788b7ebe5b59785fad760`,
respectively. Their IDs exactly match the frozen valid-labelled split cohorts
and have zero overlap. Both runs use probe-bundle SHA-256
`05c4dee461cdf789af5fc1ca45ef223dbf8fc4eab54d4724e57c59993d129f5a`.

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
Raw SE and NLL are not probabilities and therefore receive no uncalibrated
Brier/ECE claim. Native `[0,1]` P(True)/Probe scores are retained as
diagnostics.

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

All 90 reviews contain a binary human decision and none is `unsure`.

| Type | Decided | Agreement | Design-weighted agreement | Cohen's kappa | Claude incorrect / human correct | Claude correct / human incorrect |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Overall | 90 | 95.56% | 95.12% | 0.902 | 4 | 0 |
| Factoid | 30 | 93.33% | 93.33% | 0.857 | 2 | 0 |
| List | 30 | 100.00% | 100.00% | 1.000 | 0 | 0 |
| Summary | 30 | 93.33% | 93.33% | 0.867 | 2 | 0 |

The four disagreements are one-directional: Claude marks an answer incorrect
while the human reviewer accepts it. They cover a reference mismatch for the
standard RUNX1T1 expansion, two sufficiently correct but less reference-matched
summary answers, and a Velcade answer that adds a legitimate secondary
indication. The audit therefore supports the Claude correctness labels as a
high-agreement target while identifying a small conservative tendency to
overstate error. It does not establish the exact population bias from four
discordant cases.

As a frozen diagnostic only, the existing validation-calibrated probabilities
were evaluated on the same audit sample after inverse-probability weighting by
question type. Nothing was fitted or selected on the human labels.

| Method | Claude-label AUROC | Human-label AUROC | Human-label Brier | Human-label log loss |
| --- | ---: | ---: | ---: | ---: |
| Accuracy-Probe | 0.8782 | **0.8939** | **0.1657** | **0.5096** |
| Blind P(True) | 0.8564 | 0.8701 | 0.2240 | 0.6284 |
| Semantic Entropy | 0.7536 | 0.7919 | 0.1754 | 0.5121 |
| P(True)-Probe | 0.8247 | 0.8345 | 0.1943 | 0.5699 |

The fixed sample is not a replacement test set and receives no new
significance claim. Its primary-method AUROC order remains Accuracy-Probe,
blind P(True), then SE; Accuracy-Probe retains the best Brier score, while SE
remains close on Brier/log loss and better than blind P(True). The four label
corrections therefore do not reverse the complete-test conclusions.

Local ignored artifacts:

```text
analysis_outputs/bioasq_phase2_correctness_audit_20260801/
  correctness_audit_blinded.csv
  correctness_audit_completed.csv
  correctness_audit_key.csv
  correctness_audit_manifest.json
  correctness_audit_agreement.csv
  correctness_audit_merged.csv
  correctness_audit_uq_sensitivity.csv
  correctness_audit_analysis_summary.json
```

The completed-review SHA-256 is
`ab1efc4a60d97424ee161cdeb42e942c39caedb960741db116a616ed89e37b94`.
The analysis summary records the key and calibrated-prediction input hashes and
marks the audit `complete`.

## PubMedQA calibration transfer

After BioASQ validation fitting, the blind-P(True), Accuracy-Probe, and
P(True)-Probe mappings are applied unchanged to the existing 500-question
PubMedQA Appendix-C context-v2 scores. PubMedQA labels, scores, prevalence, or
metrics do not fit, select, or modify a mapping. Report native and transferred
Brier, Brier skill, log loss, ECE, AUROC, and AP.

Semantic Entropy is absent from this transfer because PubMedQA used a
single-answer protocol. This remains a protocol boundary rather than a negative
SE result.

## Complete UQ results

### BioASQ test calibration

The five scalar mappings were fitted on all 384 valid-labelled validation
questions and applied unchanged to the 384 valid-labelled test questions.

| Method | AUROC | Calibrated Brier | Brier skill | Log loss | 10-bin ECE | AURAC 0.5--1.0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy-Probe | **0.8058** | **0.1776** | **0.2125** | 0.5386 | 0.0891 | **0.2780** |
| Blind P(True) | 0.7900 | 0.2156 | 0.0444 | 0.6174 | 0.1922 | 0.2909 |
| Semantic Entropy | 0.7484 | 0.1839 | 0.1847 | **0.5341** | **0.0298** | 0.2799 |
| 10-sample normalized NLL | 0.7382 | 0.1936 | 0.1419 | 0.5645 | 0.0493 | 0.2870 |
| P(True)-Probe | 0.7452 | 0.2001 | 0.1131 | 0.5860 | 0.0984 | 0.2905 |

The test-prevalence constant has Brier `0.2256`. Before validation calibration,
native Brier was `0.2162` for Accuracy-Probe, `0.5347` for blind P(True), and
`0.3177` for P(True)-Probe. Their corresponding native ECE values were
`0.1921`, `0.5392`, and `0.3004`. Blind P(True) therefore has strong ranking but
is not a usable answer-error probability without a separately fitted mapping.

Primary-method differences use 20,000 paired test-row resamples. Differences
are candidate minus comparator:

| Candidate vs comparator | Metric | Difference | 95% CI |
| --- | --- | ---: | ---: |
| Accuracy-Probe vs blind P(True) | AUROC | +0.01584 | [-0.03850, +0.07163] |
|  | Brier | -0.03794 | [-0.06050, -0.01541] |
|  | Log loss | -0.07884 | [-0.13526, -0.02177] |
|  | AURAC | -0.01294 | [-0.02322, -0.00316] |
| Accuracy-Probe vs SE | AUROC | +0.05745 | [+0.00528, +0.10949] |
|  | Brier | -0.00629 | [-0.02877, +0.01653] |
|  | Log loss | +0.00452 | [-0.05230, +0.06215] |
|  | AURAC | -0.00191 | [-0.01015, +0.00598] |
| Blind P(True) vs SE | AUROC | +0.04161 | [-0.01386, +0.09529] |
|  | Brier | +0.03165 | [+0.01620, +0.04707] |
|  | Log loss | +0.08336 | [+0.03998, +0.12633] |
|  | AURAC | +0.01103 | [+0.00087, +0.02158] |

Accuracy-Probe ranks errors significantly better than SE, but their Brier, log
loss, and AURAC differences remain unresolved. SE has the lowest ECE point
estimate. Relative to blind P(True), SE is significantly better on Brier, log
loss, and AURAC even though their AUROC difference remains unresolved.
Accuracy-Probe retains its bootstrap-supported Brier, log-loss, and AURAC
advantage over blind P(True), while their AUROC difference remains unresolved.
The auxiliary NLL comparator ranks below all four existing methods by AUROC,
but after validation calibration it has better Brier, log loss, ECE, and AURAC
point estimates than blind P(True) and P(True)-Probe. No new pairwise bootstrap
claim was added because the frozen primary-method comparison remains unchanged.

At 80% coverage, retained error risk is `0.5779` for SE, `0.5877` for
Accuracy-Probe, `0.5909` for 10-sample normalized NLL, `0.6006` for blind
P(True), and `0.5942` for P(True)-Probe, versus `0.6563` at full coverage.

### Validation-fixed selective-prediction operating points

The original risk--coverage analysis ranks the test scores and retains an
exact requested fraction within that cohort. The supplementary deployment
analysis instead selects raw-score thresholds using only the 384-question
validation split, then applies each threshold unchanged to the 384-question
test split. It uses the already declared target coverages `0.80`, `0.90`, and
`0.95`; no test label, test score distribution, calibration refit, or
type-specific threshold affects selection.

The threshold is the validation `ceil(N * target coverage)` order statistic.
Every score equal to the threshold is retained, and the resulting achieved
coverage is reported rather than splitting boundary ties on test. The full-test
error risk is `0.65625`. Confidence intervals use 20,000 paired test-row
bootstrap resamples with seed `20260820`, conditional on the one frozen
full-validation threshold; they do not include uncertainty from replacing the
validation cohort.

| Target | Method | Test coverage | Retained risk (95% CI) | Absolute risk reduction (95% CI) | Rejected risk |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.80 | SE | 0.8073 | 0.5806 [0.5253, 0.6349] | 0.0756 [0.0566, 0.0963] | 0.9730 |
| 0.80 | Blind P(True) | 0.7891 | 0.6007 [0.5452, 0.6553] | 0.0556 [0.0336, 0.0784] | 0.8642 |
| 0.80 | Accuracy-Probe | 0.7630 | 0.5666 [0.5087, 0.6228] | 0.0897 [0.0665, 0.1146] | 0.9451 |
| 0.80 | P(True)-Probe | 0.7943 | 0.5902 [0.5350, 0.6447] | 0.0661 [0.0452, 0.0884] | 0.9114 |
| 0.90 | SE | 0.9375 | 0.6333 [0.5833, 0.6825] | 0.0229 [0.0141, 0.0329] | 1.0000 |
| 0.90 | Blind P(True) | 0.9062 | 0.6351 [0.5836, 0.6851] | 0.0212 [0.0079, 0.0346] | 0.8611 |
| 0.90 | Accuracy-Probe | 0.9089 | 0.6246 [0.5735, 0.6744] | 0.0316 [0.0202, 0.0442] | 0.9714 |
| 0.90 | P(True)-Probe | 0.8880 | 0.6246 [0.5727, 0.6755] | 0.0316 [0.0175, 0.0464] | 0.9070 |
| 0.95 | SE | 1.0000 | 0.6562 [0.6094, 0.7031] | 0.0000 [0.0000, 0.0000] | not estimable |
| 0.95 | Blind P(True) | 0.9453 | 0.6446 [0.5945, 0.6929] | 0.0116 [0.0018, 0.0215] | 0.8571 |
| 0.95 | Accuracy-Probe | 0.9453 | 0.6364 [0.5865, 0.6851] | 0.0199 [0.0119, 0.0291] | 1.0000 |
| 0.95 | P(True)-Probe | 0.9609 | 0.6450 [0.5957, 0.6932] | 0.0113 [0.0038, 0.0192] | 0.9333 |

Every non-degenerate operating point reduces retained risk relative to using
all test answers in the conditional bootstrap. This is not a pairwise method
superiority test: for example, Accuracy-Probe has the lowest risk at the 0.80
target but also undershoots to `0.7630` coverage, so its risk is not directly
comparable with SE at `0.8073` coverage.

SE exposes the clearest threshold-granularity limitation. Its validation 0.90
boundary contains 11 tied scores and produces `0.9375` test coverage. Its 0.95
boundary contains 28 validation ties and 24 test ties at the maximum score, so
the inclusive rule retains all 384 test examples and provides no selective
benefit. The continuous P(True)/Probe scores remain much closer to their target
coverages, apart from Accuracy-Probe reaching only `0.7630` at the 0.80 target.

Applying the same global threshold within each answer form also reveals a
large coverage imbalance at the 0.80 target. Across methods, factoid coverage
ranges from `0.7342` to `0.9114`, list coverage from `0.4175` to `0.6505`, and
summary coverage from `0.9512` to `1.0000`. The global policy therefore rejects
list answers much more often, consistent with their much higher error
prevalence, but it does not provide uniform service coverage across answer
forms. These are descriptive subgroup audits; no type-specific threshold was
fitted.

The accepted ignored result snapshot is:

```text
analysis_outputs/bioasq_phase2_selective_operating_points_20260820/
  operating_points_overall.csv
  operating_points_by_type.csv
  bootstrap_intervals.csv
  operating_point_predictions.csv
  operating_points.svg
  operating_points.png
  summary.json
```

The analysis reuses validation/test UQ files with SHA-256
`cc86e63cd1197d3b896fc8e14808c2fa3be308ce96961b40a2b3d5e0962d6245`
and `181e5a77e3a7ba13a61577b6edb1d19c25539cdaf9a788b7ebe5b59785fad760`,
respectively. The implementation entry point is
`analysis/run_phase2_selective_operating_points.py`.

### Answer-form calibration

One global validation-fitted mapping was used for every type.

| Method | Factoid Brier skill | List Brier skill | Summary Brier skill |
| --- | ---: | ---: | ---: |
| Semantic Entropy | +0.1221 | **+0.1666** | +0.0350 |
| Accuracy-Probe | +0.1553 | -0.1344 | +0.1613 |
| Blind P(True) | -0.0036 | -0.3834 | -0.0428 |
| 10-sample normalized NLL | **+0.2047** | -0.5169 | +0.0123 |
| P(True)-Probe | +0.0571 | -0.2365 | +0.0326 |

List error prevalence is `0.9029`, compared with `0.5886` for factoid and
`0.5366` for summary. SE is the only method whose global calibrator beats the
type-prevalence constant on list. NLL has the strongest factoid Brier-skill
point estimate, while Accuracy-Probe is strongest on summary. This extends the
answer-form conclusion from ranking to calibration: method suitability remains
answer-form dependent. No type-specific calibrator is fitted post hoc.

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
analysis_outputs/bioasq_phase2_uq_calibration_full_20260801/
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

The NLL supplement is preserved separately so the accepted 2026-08-01 snapshot
is not overwritten:

```text
analysis_outputs/bioasq_phase2_uq_calibration_nll_supplement_20260827/
  metrics_overall.csv
  metrics_by_type.csv
  reliability_bins.csv
  reliability_diagram.svg
  risk_coverage.csv
  risk_coverage.svg
  calibrated_predictions.csv
  paired_bootstrap.csv
  summary.json
```

## Stop boundary

This completion does not authorize:

- fitting or selecting anything on the BioASQ test split;
- PubMedQA target recalibration;
- new Probe architectures, fusion, type-specific heads, or hidden-state search;
- new model sizes, prompt tuning, or expanded Claude clustering; or
- regeneration of the completed Phase-2 test benchmark.

The fixed analyses and audit are complete. Experiments are frozen for thesis
writing unless a separate supervisor-approved idea is explicitly opened.
