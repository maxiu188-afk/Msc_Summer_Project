# Active Documentation Index

Only current protocols, results, data contracts, and runtime records belong in
this directory. Historical material is kept under the repository-level
`archive_unused/` directory.

## Current research documents

- `../../PHASE2_PROBE_PLAN.md`: canonical Phase-2 Probe plan, frozen choices,
  cross-dataset boundaries, and the approved post-Simpson SE/P(True)
  operating-regime follow-up.
- `../../SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`: frozen brief from
  the completed meeting; later research decisions are recorded in the
  canonical Phase-2 plan rather than retroactively added to the meeting record.
- `phase2_bioasq_dataset_split.md`: leakage-safe BioASQ split contract.
- `phase2_uq_efficiency_benchmark.md`: completed 384-question incremental-cost
  comparison of the two Probes, blind P(True), sampled normalized NLL, SE, and
  cluster count, including AUROC/AP, latency, GPU time, generated tokens,
  throughput, and correction provenance.
- `phase2_probe_completion_statistics.md`: Phase-2 statistical closure with
  the paired Accuracy-Probe-versus-blind-P(True) AUROC bootstrap and the
  validation-fitted two-Probe fusion diagnostic.
- `summary_length_intervention.md`: completed paired BioASQ-summary
  current-versus-one-or-two-sentence intervention, including realised length,
  correctness, SE/P(True) ranking, paired bootstrap, and correction provenance.
- `summary_clustering_diagnostic.md`: completed 93/96-question
  fixed-generation Claude-versus-PubMedBERT clustering mechanism diagnostic
  for the original longer summary condition.
- `gemma3_model_scale_experiment.md`: completed Gemma 3 4B versus aligned
  Phase-1 12B result plus the completed staged 1B generation, feasibility
  gates, and formal three-model summary comparison.
- `pubmedqa_frozen_probe_transfer.md`: frozen-Probe PubMedQA transfer protocol,
  complete context-v1/context-v2 results, comparison tables, and evidence
  pointers. The question-only result is archived under
  `../../archive_unused/docs/historical_results/`.
- `bioasq_medical_uq_protocol.md`: Phase-1 BioASQ UQ protocol.
- `bioasq_medical_uq_results_20260718.md`: canonical Phase-1 results.
- `semantic_entropy_generation_protocol.md`: generation/SE sampling contract.
- `experiment_runtime_log.md`: chronological job and runtime provenance.

## Documentation rule

Update the owning document above instead of creating another progress summary.
Use `experiment_runtime_log.md` for job state, the relevant protocol document
for experimental design, and the root plan for research decisions. When a
document becomes obsolete, move it to `../../archive_unused/docs/` and update
this index and all active links.
