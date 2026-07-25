# Phase-2 Probe Completion Statistics

Last updated: 2026-07-25

## Status

Phase-2 is complete. This final offline analysis closes the two statistical
items that remained after the 384-question efficiency benchmark:

1. paired-bootstrap uncertainty for Accuracy-Probe versus direct blind
   P(True); and
2. one validation-fitted two-Probe fusion diagnostic evaluated once on test.

The analysis uses saved validation/test hidden states, frozen Probe parameters,
saved blind-P(True) scores, and the existing Claude binary labels only. It does
not regenerate answers, rerun NLI, call Claude, refit either Probe, search
layers or token positions, or tune on test.

## Frozen data and models

| Item | Value |
| --- | --- |
| Dataset | BioASQ Phase-2 validation and test splits |
| Valid labelled rows | 384 validation; 384 test |
| Positive class | Claude `incorrect` |
| P(True)-Probe | frozen hard-even block 24 / LT logistic head |
| Accuracy-Probe | frozen block 24 / LT logistic head |
| Direct comparator | saved blind-P(True) uncertainty |
| Bootstrap | 20,000 paired test-row resamples, seed `20260725` |
| Fusion | standardized two-score logistic regression fitted on validation only |

## Paired bootstrap

The primary comparison is the test AUROC difference:

```text
Accuracy-Probe AUROC - blind P(True) AUROC
```

| Accuracy-Probe | Blind P(True) | Difference | Paired-bootstrap 95% CI | Resamples |
| ---: | ---: | ---: | ---: | ---: |
| 0.8058 | 0.7900 | +0.01584 | [-0.03890, 0.07173] | 20,000 / 20,000 valid |

The interval includes zero. Accuracy-Probe has the higher point estimate, but
this experiment does not establish a statistically reliable AUROC improvement
over blind P(True). The fraction of bootstrap differences at or below zero was
`0.2907`; this is retained as a diagnostic, not reported as a substitute for
the two-sided interval.

## Validation-fitted two-Probe fusion

The exploratory fusion has exactly two inputs: the frozen P(True)-Probe score
and frozen Accuracy-Probe score. A standard scaler and L2 logistic regression
were fitted once on the 384 validation labels, frozen, and applied once to the
384 test rows. Both standardized coefficients are positive:

| Input | Standardized coefficient |
| --- | ---: |
| P(True)-Probe | 0.5189 |
| Accuracy-Probe | 1.1241 |

The positive P(True)-Probe coefficient is a weak complementarity signal, but
the final ranking gain is small:

| Test score | AUROC | AP | Brier |
| --- | ---: | ---: | ---: |
| Two-Probe validation-fitted fusion | **0.8125** | 0.8871 | **0.1688** |
| Accuracy-Probe | 0.8058 | **0.8884** | 0.2162 |
| Blind P(True) | 0.7900 | 0.8383 | 0.5347 |
| P(True)-Probe | 0.7452 | 0.8357 | 0.3177 |

Fusion improves AUROC by only `0.0067` over Accuracy-Probe and slightly lowers
AP by `0.0013`. Its lower Brier score is consistent with validation fitting to
the correctness target, but does not turn this post-hoc diagnostic into a new
primary method. No fusion hyperparameter or feature search was performed, and
no claim of a significant gain is made.

## Interpretation and Phase-2 boundary

The completed evidence supports three restrained conclusions:

- Accuracy-Probe is the strongest Phase-2 point estimate, but its AUROC
  advantage over blind P(True) is not statistically resolved by the paired
  bootstrap.
- P(True)-Probe contributes a positive validation-fitted fusion coefficient,
  but the test ranking gain over Accuracy-Probe alone is too small and mixed
  across AUROC/AP to justify a new combined main model.
- The efficiency result remains the clear Probe advantage: either Probe costs
  about 60.5 ms per question, both jointly cost 60.54 ms, and blind P(True)
  costs 154.4 ms.

Phase-2 therefore closes with the two frozen individual Probes, the efficiency
evidence, the paired uncertainty interval, and one explicitly exploratory
fusion result. Additional type-specific optimization is not part of this
phase. The next main experiment is the paired BioASQ-summary answer-length
intervention defined in `../../PHASE2_PROBE_PLAN.md`.

## Reproducibility

Implementation:

- `analysis/run_phase2_probe_completion.py`
- `tests/test_phase2_probe_completion.py`

Command:

```bash
cd archehr_sebaseline
../.venv/bin/python analysis/run_phase2_probe_completion.py
```

Ignored local output:

```text
analysis_outputs/bioasq_phase2_probe_completion_20260725/
  metrics.csv
  paired_bootstrap.csv
  predictions.csv
  summary.json
```

`summary.json` records the protocol, fusion parameters, input SHA-256 values,
and test-use boundary.
