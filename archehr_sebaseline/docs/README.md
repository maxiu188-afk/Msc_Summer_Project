# Active Documentation Index

Only current protocols, results, data contracts, and runtime records belong in
this directory. Historical material is kept under the repository-level
`archive_unused/` directory.

## Current research documents

- `../../PHASE2_PROBE_PLAN.md`: canonical Phase-2 Probe plan, frozen choices,
  and cross-dataset experiment boundaries.
- `../../SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`: current meeting
  brief with the completed results and proposed remaining research direction.
- `phase2_bioasq_dataset_split.md`: leakage-safe BioASQ split contract.
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
