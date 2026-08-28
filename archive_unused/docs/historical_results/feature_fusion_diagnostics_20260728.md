# Archived post-hoc feature-fusion diagnostics

Archived: 2026-07-28

These analyses are retained for provenance but are no longer part of the
three-phase main research narrative. Neither analysis justified a new model,
and neither should be presented as a primary project result.

## Phase-1 P(True)/SE/NLL feature fusion

Source experiment: final 1,000-question Phase-1 BioASQ runs, seeds 31 and 47.

The CPU-only feature-fusion analysis used repeated five-fold out-of-fold
logistic regression and 2,000 paired bootstrap resamples. It did not change any
generated answer or UQ artifact. Its completed exploratory script is retained
only in the local code archive and is no longer Git-tracked.

| Comparison with blind P(True) | Seed 31 AUROC change (95% CI) | Seed 47 AUROC change (95% CI) |
| --- | ---: | ---: |
| P(True) + discrete SE | +0.013 `[-0.005,+0.035]` | +0.014 `[-0.005,+0.034]` |
| P(True) + normalized NLL | -0.040 `[-0.074,-0.006]` | -0.036 `[-0.068,-0.004]` |
| P(True) + SE + NLL | +0.001 `[-0.025,+0.031]` | +0.005 `[-0.022,+0.033]` |

No fusion cleared the paired-bootstrap criterion. The result was used only to
rule out an unsupported multi-target SE/NLL/P(True) Probe.

## Phase-2 validation-fitted two-Probe fusion

Source experiment: frozen BioASQ Phase-2 validation and test splits, 384 valid
labels in each split.

A StandardScaler and L2 logistic regression were fitted once on validation
using the frozen P(True)-Probe and Accuracy-Probe scores, then evaluated once
on test.

| Test score | AUROC | AP | Brier |
| --- | ---: | ---: | ---: |
| Two-Probe fusion | 0.8125 | 0.8871 | 0.1688 |
| Accuracy-Probe | 0.8058 | 0.8884 | 0.2162 |
| Blind P(True) | 0.7900 | 0.8383 | 0.5347 |
| P(True)-Probe | 0.7452 | 0.8357 | 0.3177 |

The fusion raised AUROC by only 0.0067 over Accuracy-Probe and lowered AP by
0.0013. It remains an explicitly exploratory complementarity check, not a
third Probe or a replacement for either frozen individual Probe.

## Active-result pointer

Use `../../../RESEARCH_RESULTS_SYNTHESIS_ZH.md` for the three-phase research
story and `../../../archehr_sebaseline/docs/README.md` for active protocol and
result documents.
