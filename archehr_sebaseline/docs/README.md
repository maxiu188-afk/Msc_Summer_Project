# Active Documentation Index

The project is organized around one completed three-phase research narrative:

> Clinical-QA uncertainty methods have different operating conditions, shaped
> by answer form, model capability, semantic-clustering quality, and the Probe
> supervision target.

Start with:

- `../../RESEARCH_RESULTS_SYNTHESIS_ZH.md`: thesis-oriented three-phase result
  synthesis and final scope boundary;
- `../../README.md`: concise repository-level result map;
- `../../PROJECT_CLOSEOUT.md`: frozen scope, active evidence ownership, and
  reproducibility entry points.

## Phase 1 — answer-form UQ regimes

- `bioasq_medical_uq_protocol.md`: final no-evidence BioASQ generation,
  clustering, UQ, and correctness-label protocol.
- `bioasq_medical_uq_results_20260718.md`: canonical two-seed, 1,000-question
  factoid/list/summary result.

## Phase 2 — hidden-state probes

- `phase2_bioasq_dataset_split.md`: leakage-safe train/validation/test split.
- `phase2_uq_efficiency_benchmark.md`: matched correctness-ranking and
  incremental-cost comparison.
- `phase2_probe_completion_statistics.md`: primary paired
  Accuracy-Probe-versus-blind-P(True) uncertainty result.
- `phase2_uq_calibration_completion.md`: completed supervisor-approved closing
  evaluation for validation-fitted calibration, selective prediction,
  validation-fixed deployment operating points, correctness-label audit, label
  sensitivity, zero-refit PubMedQA calibration transfer, and the saved-artifact
  10-sample normalized-NLL probability-quality supplement.
- `pubmedqa_frozen_probe_transfer.md`: bounded frozen-Probe transfer under the
  official PubMedQA evidence context, including comparison with blind P(True),
  verbalized confidence, NLL, and token-entropy baselines, plus the completed
  separately authorized context-v2 Semantic Entropy result and its completed
  paired temperature-sensitivity diagnostic.

The P(True)-Probe and Accuracy-Probe remain distinct: one targets direct
P(True)-derived uncertainty, while the other targets answer error. Their
different PubMedQA transfer behavior is part of the main Phase-2 comparison.
The ten-sample SE completion passed formal Isambard health and provenance
checks: 500 questions, 5,000 generations, 0.5884 AUROC, and 0.3674 AP.
The final 200-question paired diagnostic found no resolved change in SE AUROC,
AP, or mean SE across `T=0.7/1.0/1.3`; lower `T=0.7` produced more single-cluster
questions, but no clear ranking-performance effect.

## Phase 3 — conditions and mechanisms

- `summary_length_intervention.md`: controlled shortening result; length alone
  does not recover SE.
- `gemma3_model_scale_experiment.md`: aligned 12B/4B/1B result; P(True)'s
  relative advantage declines with model capability.
- `summary_clustering_diagnostic.md`: 93-question fixed-generation
  NLI-versus-Claude clustering result plus completed stratified 24-pair human
  review.

## Provenance and operations

- `../../PROJECT_CLOSEOUT.md`: active result-to-script map and frozen scope.
- `../../archive_unused/docs/project_history/experiment_runtime_log_final.md`:
  completed chronological jobs, failures, timing, and artifact provenance.
- `../../archive_unused/docs/project_history/ISAMBARD_COMMANDS_FINAL.md`:
  completed local/Isambard command notebook.

These records preserve failed attempts and implementation detail but do not
define additional research claims.

## Archived exploration

Superseded meeting briefs, early pilots, question-only PubMedQA transfer, and
the report for the post-hoc feature-fusion diagnostic are under
`../../archive_unused/`. The complete Phase-2 plan and the older mixed-status
package README are also archived there now that experiments are frozen.
Superseded and exploratory source code is kept only in the ignored local
`../../.local_archive/` and is not part of GitHub.
In particular:

- `../../archive_unused/docs/historical_results/feature_fusion_diagnostics_20260728.md`
  retains Phase-1 feature fusion and Phase-2 two-Probe fusion;
- `../../archive_unused/docs/project_history/SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`
  retains the superseded supervisor brief.

Archived material is provenance, not current evidence. Do not move it back
into the main narrative without a new explicit research decision.

## Closing boundary

The supervisor-approved calibration, selective-prediction, correctness-audit,
zero-refit transfer analysis, bounded PubMedQA context-v2 Semantic Entropy row,
fixed 200-question paired temperature diagnostic, and validation-fixed
selective-prediction operating-point supplement are complete. The saved
10-sample normalized-NLL probability-quality row is also complete and does not
extend the operating-point experiment. Do not
retune prompts, expand the model grid, train new Probes, extend 1B factoid/list,
or open any other clustering work. The temperature diagnostic changed no other
protocol field and found no clear SE-ranking effect over `T=0.7–1.3`; future
work returns to thesis writing unless another research decision is explicit.
