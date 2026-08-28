# Clinical QA Uncertainty Research Workspace

This repository studies:

> **When different uncertainty methods are appropriate in clinical QA, and
> how answer form, model scale, and semantic-clustering quality affect
> Semantic Entropy, P(True), and hidden-state probes.**

The three research phases, bounded diagnostics, supervisor-approved UQ closing
evaluation, and the separately approved PubMedQA completions are complete. The
final bounded temperature-sensitivity diagnostic reused the accepted `T=1.0`
arm and added only `T=0.7/1.3` on one fixed 200-question subset. It found no
resolved temperature effect on SE error-ranking performance over that range.
The preceding PubMedQA completion added ten-sample Semantic Entropy to the
already accepted 500-question PubMedQA Appendix-C context-v2 condition. Both
additions reused the frozen prompt, low-temperature decision errors, model, and
NLI configuration; neither reopened prompt tuning, model-scale extension,
Probe training, calibration, or correctness judging.
A final saved-artifact selective-prediction supplement then froze operating
thresholds on BioASQ validation and applied them unchanged to test; it required
no model generation or refitting.
A final offline probability-quality supplement also added the already-saved
10-sample normalized NLL score under the same validation-only calibration
protocol. It did not regenerate answers or extend the separate
validation-fixed operating-point experiment.

The thesis-oriented Chinese synthesis is
[`RESEARCH_RESULTS_SYNTHESIS_ZH.md`](RESEARCH_RESULTS_SYNTHESIS_ZH.md).
Detailed protocols and provenance remain in
[`archehr_sebaseline/docs/`](archehr_sebaseline/docs/).

## Four connected conclusions

### 1. Relative UQ performance depends on answer form

The final Phase-1 BioASQ baseline used 1,000 fixed questions at two seeds.
Factoid, list, and summary exhibit different operating regimes:

| Type | Mean sampled tokens | Discrete-SE AUROC | Blind-P(True) AUROC | P(True) minus SE |
| --- | ---: | ---: | ---: | ---: |
| Factoid | 8.08 / 8.01 | 0.720 / 0.725 | 0.795 / 0.800 | +0.075 / +0.075 |
| List | 35.42 / 35.41 | 0.845 / 0.881 | 0.845 / 0.852 | +0.000 / -0.029 |
| Summary | 111.13 / 110.92 | 0.565 / 0.595 | 0.808 / 0.826 | +0.243 / +0.231 |

SE is competitive for structured lists, while blind P(True) is stronger for
factoid and 12B summary answers. A paired summary-shortening intervention did
not recover SE, so answer length alone is not a sufficient explanation.

### 2. P(True) depends more strongly on model capability

On 196 common-valid summary questions:

| Model | Accuracy | Blind P(True) AUROC | SE AUROC | 10-sample normalized-NLL AUROC | P(True) minus SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 1B | 0.122 | 0.501 | 0.597 | 0.789 | -0.096 |
| Gemma 3 4B | 0.281 | 0.652 | 0.644 | 0.796 | +0.008 |
| Gemma 3 12B | 0.459 | 0.807 | 0.564 | 0.756 | +0.243 |

The 12B P(True) advantage disappears at 4B and declines further in point
estimate at 1B. The incremental 1B-minus-4B gap-change interval crosses zero,
so the below-4B continuation is a trend rather than a newly resolved effect.

### 3. SE is relatively stable across scale but sensitive to clustering

Replacing only the semantic-equivalence layer on 93 fixed-generation summary
questions raises SE AUROC from 0.595 with PubMedBERT NLI clustering to 0.781
with Claude clustering. The change is `+0.187`, 95% CI
`[+0.080,+0.291]`, and closes about 85% of the original P(True)-minus-SE point
estimate gap.

The completed stratified 24-pair human review supports the mechanism:

| Method | Human agreement | False merges | False splits |
| --- | ---: | ---: | ---: |
| Original NLI clustering | 9/24 | 5 | 10 |
| Claude clustering | 19/24 | 0 | 5 |
| Claude direct judgement | 17/24 | 0 | 7 |

These agreement rates describe the diagnostic sample, not population accuracy.
Claude remains too costly and non-transitive to be proposed as the production
clustering method.

### 4. Single-generation probes offer a competitive low-cost alternative

The frozen Accuracy-Probe reaches 0.8058 AUROC / 0.8884 AP for answer-error
ranking, compared with 0.7900 / 0.8383 for blind P(True) and 0.7484 / 0.8446
for SE. Its AUROC advantage over blind P(True) is not statistically resolved:
`+0.01584`, 95% CI `[-0.03890,0.07173]`.

The P(True)-Probe answers a different question. It reproduces the model's
direct P(True)-derived target with held-out AUROC 0.9026; the Accuracy-Probe
directly predicts correctness. This distinction exposes the trade-off between
recovering internal self-evaluation and supervising answer error.

Either Probe costs about 60.5 ms/question, versus 154.4 ms for blind P(True)
and 32.75 s for ten-sample SE.

The frozen heads were also evaluated without retraining or target
recalibration on the official 500-question PubMedQA PQA-L test set. Under the
final Appendix-C context-v2 prompt, answer accuracy is 72.4% and the
single-answer error-ranking comparison is:

| Method | BioASQ test AUROC | PubMedQA v2 AUROC | PubMedQA v2 AP |
| --- | ---: | ---: | ---: |
| P(True)-Probe | 0.7452 | **0.6839** | **0.4519** |
| Blind P(True) | 0.7900 | 0.6490 | 0.4210 |
| Verbalized confidence | — | 0.6402 | 0.4164 |
| Accuracy-Probe | **0.8058** | 0.5901 | 0.3916 |
| Discrete Semantic Entropy | 0.7484 | 0.5884 | 0.3674 |
| Single-answer normalized NLL | — | 0.5424 | 0.3075 |
| Mean token entropy | — | 0.5423 | 0.3080 |
| Single-answer sequence NLL | — | 0.5400 | 0.3025 |
| Max token entropy | — | 0.5372 | 0.2914 |

The source-domain correctness-supervised Accuracy-Probe degrades more than the
P(True)-Probe, which becomes the strongest PubMedQA-v2 ranking score. However,
the P(True)-Probe's fidelity to its original frozen teacher target also falls
from 0.9026 in BioASQ to 0.5899 in PubMedQA. The transfer result is therefore
limited cross-target error-ranking usefulness, not preservation of the source
mapping or dataset-independent generalization. AP is not compared across
datasets because error prevalence differs.

The original frozen-Probe transfer used one answer only. The later bounded SE
completion reused its prompts and error labels, added exactly ten sampled
answers per question, and obtained 0.5884 AUROC / 0.3674 AP. PubMedBERT placed
444/500 questions in one semantic cluster, so the accepted SE ranking is weak;
this collapse alone cannot distinguish genuinely consistent answers from NLI
over-merging. The BioASQ ten-sample normalized-NLL result remains omitted from
the cross-dataset column because the PubMedQA NLL rows use one answer.

A final paired diagnostic tested `T=0.7/1.0/1.3` on a deterministic stratified
200-question subset. AUROC was `0.5837/0.6213/0.5945`; both changes relative to
`T=1.0` had 95% bootstrap intervals crossing zero (`[-0.0913,+0.0123]` and
`[-0.0774,+0.0213]`). AP and mean-SE changes were likewise unresolved. Lowering
temperature to `0.7` did increase the single-cluster fraction by 5 percentage
points, 95% CI `[+1.0,+9.5]`, but this did not produce a resolved ranking change.
The supported conclusion is therefore no clear SE-performance effect within
the tested range and cohort, not temperature invariance in general.

## Completed thesis-closing evaluation

The final closing scope added validation-fitted calibration, selective
prediction, answer-form calibration, a correctness-label audit, and a
zero-refit PubMedQA calibration-transfer stress test without reopening model or
prompt selection.

| Method | BioASQ AUROC | Calibrated Brier | ECE | AURAC 0.5--1.0 |
| --- | ---: | ---: | ---: | ---: |
| Accuracy-Probe | **0.8058** | **0.1776** | 0.0891 | **0.2780** |
| Blind P(True) | 0.7900 | 0.2156 | 0.1922 | 0.2909 |
| Semantic Entropy | 0.7484 | 0.1839 | **0.0298** | 0.2799 |
| 10-sample normalized NLL | 0.7382 | 0.1936 | 0.0493 | 0.2870 |
| P(True)-Probe | 0.7452 | 0.2001 | 0.0984 | 0.2905 |

Accuracy-Probe ranks error significantly better than SE, while their Brier,
log-loss, and AURAC differences remain unresolved. SE has the lowest ECE and
is the only method whose global calibrator beats the type-prevalence baseline
on list questions. The auxiliary NLL row completes the probability-quality
comparison without changing the primary-method bootstrap conclusions. All
three single-answer BioASQ mappings have negative Brier skill when transferred
unchanged to PubMedQA, so ranking transfer is not calibration transfer.

The validation-fixed operating-point supplement shows the deployment boundary
behind those curves. At the nominal 0.80 coverage target, achieved test
coverage is `0.807/0.789/0.763/0.794` for SE, blind P(True), Accuracy-Probe,
and P(True)-Probe, with retained risks `0.581/0.601/0.567/0.590` versus
`0.656` at full coverage. Accuracy-Probe has the lowest risk but also the
lowest achieved coverage, so this is not a pairwise superiority claim. SE's
0.95 validation threshold retains all test examples because of boundary ties,
and the same global 0.80 thresholds produce much lower coverage on list than
summary questions. NLL was not added to this separate operating-point
supplement. Full protocol and conditional bootstrap intervals are in
[`phase2_uq_calibration_completion.md`](archehr_sebaseline/docs/phase2_uq_calibration_completion.md).

The blinded correctness audit reviewed 90 answers, balanced 30 per type. Human
and Claude labels agree on 86/90 answers: 95.56% raw agreement, 95.12%
design-weighted agreement, and Cohen's kappa 0.902. All four disagreements are
Claude-incorrect/human-correct. A frozen label-sensitivity diagnostic preserves
the primary Accuracy-Probe > blind P(True) > SE ranking and does not reverse the
calibration conclusions.

## Three completed phases

### Phase 1 — answer-form operating regimes

- established the two-seed, 1,000-question no-evidence BioASQ benchmark;
- compared SE, P(True), NLL/token scores, cluster count, and simple
  disagreement;
- established the replicated factoid/list/summary contrast.

Primary documents:

- [`bioasq_medical_uq_protocol.md`](archehr_sebaseline/docs/bioasq_medical_uq_protocol.md)
- [`bioasq_medical_uq_results_20260718.md`](archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md)

### Phase 2 — hidden-state probes

- collected leakage-safe train/validation/test hidden states for 3,930
  questions;
- selected and froze block-24/LT P(True)-Probe and Accuracy-Probe;
- completed held-out correctness, fidelity, calibration, efficiency, and
  bounded PubMedQA transfer evaluation.

Primary documents:

- [`PROJECT_CLOSEOUT.md`](PROJECT_CLOSEOUT.md)
- [`phase2_bioasq_dataset_split.md`](archehr_sebaseline/docs/phase2_bioasq_dataset_split.md)
- [`phase2_uq_efficiency_benchmark.md`](archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md)
- [`phase2_probe_completion_statistics.md`](archehr_sebaseline/docs/phase2_probe_completion_statistics.md)
- [`phase2_uq_calibration_completion.md`](archehr_sebaseline/docs/phase2_uq_calibration_completion.md)
- [`pubmedqa_frozen_probe_transfer.md`](archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md)

### Phase 3 — conditions and mechanisms

- showed that controlled summary shortening does not restore SE;
- showed that P(True)'s relative advantage declines with model scale;
- showed that long-answer semantic-clustering quality is a major SE
  bottleneck, supported by the completed human review.

Primary documents:

- [`summary_length_intervention.md`](archehr_sebaseline/docs/summary_length_intervention.md)
- [`summary_clustering_diagnostic.md`](archehr_sebaseline/docs/summary_clustering_diagnostic.md)
- [`gemma3_model_scale_experiment.md`](archehr_sebaseline/docs/gemma3_model_scale_experiment.md)

## Scope boundary

The supervisor-approved calibration/selective-prediction completion,
correctness-label audit, zero-refit PubMedQA calibration transfer, and bounded
PubMedQA-v2 Semantic Entropy addition are complete. The final 200-question
paired temperature diagnostic described above is also complete, so no
experimental scope remains open. The separately approved validation-fixed
selective-prediction operating-point analysis is likewise complete. The
following remain closed:

- no additional PubMedQA prompt or error-case tuning;
- no full 1B factoid/list expansion after the 8/50 and 3/50 feasibility
  results;
- no 270M or 27B scale point;
- no new Probe, multi-target Probe, or fusion model;
- no larger Claude clustering experiment;
- no claim that Claude clustering is deployable.

The project now returns to thesis and supervisor-discussion preparation.

## Repository layout

```text
PROJECT_CLOSEOUT.md         Frozen scope and concise reproducibility map
archehr_sebaseline/        Active code, protocols, and primary result documents
server_results/            Active local server-result snapshots
archive_unused/            Tracked historical documents and compact evidence
.local_archive/            Local-only archived code; intentionally ignored
literature/                Local source material; intentionally not Git-tracked
```

The active evidence and script map is
[`PROJECT_CLOSEOUT.md`](PROJECT_CLOSEOUT.md). Completed operational commands,
the full Phase-2 plan, and chronological job provenance are preserved under
[`archive_unused/docs/project_history/`](archive_unused/docs/project_history/).

Archived documents must not be presented as current evidence. See
[`archive_unused/README.md`](archive_unused/README.md). Superseded and
exploratory code is kept only in `.local_archive/` and is not part of GitHub.
