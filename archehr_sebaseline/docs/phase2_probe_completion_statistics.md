# Phase-2 Probe Completion Statistics

Last updated: 2026-07-28

## Status

Phase-2 is complete. The active statistical result is the paired-bootstrap
uncertainty for Accuracy-Probe versus direct blind P(True) after the
384-question efficiency benchmark.

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

## Archived exploratory fusion

The validation-fitted two-Probe fusion changed AUROC/AP only marginally and
did not become a third main model. Its coefficients, metrics, and provenance
are retained under
`../../archive_unused/docs/historical_results/feature_fusion_diagnostics_20260728.md`.

## Interpretation and Phase-2 boundary

The completed main evidence supports two restrained conclusions:

- Accuracy-Probe is the strongest Phase-2 point estimate, but its AUROC
  advantage over blind P(True) is not statistically resolved by the paired
  bootstrap.
- The efficiency result remains the clear Probe advantage: either Probe costs
  about 60.5 ms per question, both jointly cost 60.54 ms, and blind P(True)
  costs 154.4 ms.

Phase-2 therefore closes with the two frozen individual Probes, the efficiency
evidence, and the paired uncertainty interval. Additional type-specific
optimization is not part of this phase. The complete three-phase result map is
`../../RESEARCH_RESULTS_SYNTHESIS_ZH.md`.

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

`summary.json` records the protocol, input SHA-256 values, and test-use
boundary. The exploratory fusion fields remain in the immutable generated
artifact for provenance but are not part of the active result narrative.
