# Archived PubMedQA Question-Only Transfer Result

Archive date: 2026-07-23

## Why this result is archived

This completed run asked Gemma 3 12B to answer the official PubMedQA PQA-L
questions without the associated PubMed abstract. PubMedQA's `yes`/`no`/`maybe`
decision is defined from the experiments and results in that paper, so the
question-only condition does not provide the evidence needed for the active
task. It produced a severe `maybe` distribution shift and is no longer an
active result or comparator.

The run is archived, not deleted. It remains useful as evidence that the
question-only prompt is mismatched to the official task. The current
evidence-backed result uses the supplied abstract context and is documented in
`../../../archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`.

## Provenance and health

```text
prompt version: pubmedqa_direct_explanation_v1
smoke job: 5734703, COMPLETED 0:0, 00:00:39
full job: 5739322, COMPLETED 0:0, 00:40:55
examples: 500/500
hidden states: [500,4,3,3840]
frozen Probe bundle: unchanged BioASQ block-24/LT bundle
PubMedQA fitting/calibration/threshold tuning: none
```

The retained ignored local artifacts were moved out of the active output paths
to:

```text
archive_unused/results/local_outputs/pubmedqa_question_only_transfer_20260721/run
archive_unused/results/local_outputs/pubmedqa_question_only_transfer_20260721/analysis
```

The original Isambard output and analysis directories remain unchanged as a
remote provenance backup.

## Answer behaviour

| Metric | Result |
| --- | ---: |
| Correct | 125/500 |
| Strict accuracy | 25.0% |
| Predicted `yes/no/maybe` | 123 / 4 / 373 |
| Official `yes/no/maybe` | 276 / 169 / 55 |
| `yes` recall | 30.1% |
| `no` recall | 0.6% |
| `maybe` recall | 74.5% |
| Error prevalence | 75.0% |

## UQ against decision error

| Score | AUROC | AP |
| --- | ---: | ---: |
| Verbalized-confidence uncertainty | **0.7757** | **0.8766** |
| Direct blind P(True) uncertainty | 0.7030 | 0.8431 |
| Frozen Accuracy-Probe | 0.5396 | 0.7718 |
| Sequence NLL | 0.5287 | 0.7715 |
| Normalized NLL | 0.5097 | 0.7634 |
| Max token entropy | 0.5053 | 0.7534 |
| Mean token entropy | 0.5012 | 0.7561 |
| Frozen P(True)-Probe | 0.4188 | 0.7001 |

P(True)-Probe is inverse-direction for correctness in this mismatched prompt
condition. The table must not be pooled with or substituted for the active
abstract-context v2 result.

## Frozen P(True)-threshold target

The unchanged BioASQ threshold marks 145/500 direct blind P(True)
uncertainties as high.

| Score | AUROC | AP |
| --- | ---: | ---: |
| Frozen P(True)-Probe | **0.5796** | **0.3382** |
| Frozen Accuracy-Probe | 0.5520 | 0.3313 |
| Verbalized-confidence uncertainty | 0.6829 | 0.3977 |
| Sequence NLL | 0.5463 | 0.3342 |
| Normalized NLL | 0.5271 | 0.3246 |
| Mean token entropy | 0.5164 | 0.3138 |
| Max token entropy | 0.4910 | 0.2961 |

This archived run is evidence of prompt/task mismatch, not evidence against
the context-conditioned PubMedQA transfer result.
