# Active Documentation Index

The project is organized around one completed three-phase research narrative:

> Clinical-QA uncertainty methods have different operating conditions, shaped
> by answer form, model capability, semantic-clustering quality, and the Probe
> supervision target.

Start with:

- `../../RESEARCH_RESULTS_SYNTHESIS_ZH.md`: thesis-oriented three-phase result
  synthesis and final scope boundary;
- `../../README.md`: concise repository-level result map;
- `../../PHASE2_PROBE_PLAN.md`: detailed frozen Phase-2 protocol and decision
  record.

## Phase 1 — answer-form UQ regimes

- `bioasq_medical_uq_protocol.md`: final no-evidence BioASQ generation,
  clustering, UQ, and correctness-label protocol.
- `bioasq_medical_uq_results_20260718.md`: canonical two-seed, 1,000-question
  factoid/list/summary result.
- `semantic_entropy_generation_protocol.md`: shared generation and SE sampling
  contract.

## Phase 2 — hidden-state probes

- `phase2_bioasq_dataset_split.md`: leakage-safe train/validation/test split.
- `phase2_uq_efficiency_benchmark.md`: matched correctness-ranking and
  incremental-cost comparison.
- `phase2_probe_completion_statistics.md`: primary paired
  Accuracy-Probe-versus-blind-P(True) uncertainty result.
- `phase2_uq_calibration_completion.md`: completed supervisor-approved closing
  evaluation for validation-fitted calibration, selective prediction,
  correctness-label audit, label sensitivity, and zero-refit PubMedQA
  calibration transfer.
- `pubmedqa_frozen_probe_transfer.md`: bounded frozen-Probe transfer under the
  official PubMedQA evidence context, including comparison with blind P(True),
  verbalized confidence, NLL, and token-entropy baselines, plus the completed
  separately authorized context-v2 Semantic Entropy result.

The P(True)-Probe and Accuracy-Probe remain distinct: one targets direct
P(True)-derived uncertainty, while the other targets answer error. Their
different PubMedQA transfer behavior is part of the main Phase-2 comparison.
The ten-sample SE completion passed formal Isambard health and provenance
checks: 500 questions, 5,000 generations, 0.5884 AUROC, and 0.3674 AP.

## Phase 3 — conditions and mechanisms

- `summary_length_intervention.md`: controlled shortening result; length alone
  does not recover SE.
- `gemma3_model_scale_experiment.md`: aligned 12B/4B/1B result; P(True)'s
  relative advantage declines with model capability.
- `summary_clustering_diagnostic.md`: 93-question fixed-generation
  NLI-versus-Claude clustering result plus completed stratified 24-pair human
  review.

## Provenance and operations

- `experiment_runtime_log.md`: chronological jobs, failures, repairs, timing,
  health checks, and artifact provenance.
- `../../ISAMBARD_COMMANDS.md`: operational local/Isambard commands.

These records preserve failed attempts and implementation detail but do not
define additional research claims.

## Archived exploration

Superseded meeting briefs, early pilots, question-only PubMedQA transfer, and
post-hoc feature-fusion diagnostics are under `../../archive_unused/`.
In particular:

- `../../archive_unused/docs/historical_results/feature_fusion_diagnostics_20260728.md`
  retains Phase-1 feature fusion and Phase-2 two-Probe fusion;
- `../../archive_unused/docs/project_history/SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`
  retains the superseded supervisor brief.

Archived material is provenance, not current evidence. Do not move it back
into the main narrative without a new explicit research decision.

## Closing boundary

The supervisor-approved calibration, selective-prediction, correctness-audit,
zero-refit transfer analysis, and bounded PubMedQA context-v2 Semantic Entropy
row are complete. No experiment remains open. Do not retune prompts, expand
the model grid, train new Probes, extend 1B factoid/list, or open any other
clustering work. Future work returns to thesis writing unless another research
decision is explicit.
