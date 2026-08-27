# Project Closeout and Reproducibility Map

Last updated: 2026-08-27

## Status

The experimental programme is complete and frozen for dissertation writing.
No prompt tuning, model extension, Probe training, calibration refit, clustering
expansion, or additional PubMedQA temperature run is currently planned.
The separately approved validation-fixed selective-prediction operating-point
analysis is complete and required no model generation, training, or refitting.
A separately authorized offline supplement has also added the already-saved
10-sample normalized NLL score to the BioASQ probability-quality comparison
under the same validation-only calibration protocol. It does not extend the
validation-fixed operating-point experiment.

This file is the concise operational replacement for the completed Phase-2
plan, the Isambard command notebook, and the chronological runtime log. Their
full contents remain available under `archive_unused/docs/project_history/` for
provenance, but they are no longer active instructions.

## Active evidence map

| Area | Owning document |
| --- | --- |
| Thesis-level conclusions | `RESEARCH_RESULTS_SYNTHESIS_ZH.md` |
| Repository overview and scope | `README.md` |
| Active documentation index | `archehr_sebaseline/docs/README.md` |
| Phase-1 protocol and final results | `archehr_sebaseline/docs/bioasq_medical_uq_protocol.md`; `archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md` |
| Phase-2 split and frozen Probes | `archehr_sebaseline/docs/phase2_bioasq_dataset_split.md`; `archehr_sebaseline/docs/phase2_probe_completion_statistics.md` |
| Efficiency, calibration, selective prediction, and audit | `archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`; `archehr_sebaseline/docs/phase2_uq_calibration_completion.md` |
| PubMedQA transfer, SE, and temperature result | `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md` |
| Length, clustering, and model-scale diagnostics | `archehr_sebaseline/docs/summary_length_intervention.md`; `archehr_sebaseline/docs/summary_clustering_diagnostic.md`; `archehr_sebaseline/docs/gemma3_model_scale_experiment.md` |

Only the owning document should be edited when a result or protocol statement
needs correction. The synthesis and repository README should remain concise
maps rather than duplicate runtime history.

## Reproducibility entry points

The maintained implementation remains under `archehr_sebaseline/`:

- `src/archehr_sebaseline/`: shared data, generation, clustering, UQ, Probe,
  and evaluation code;
- `scripts/`: final experiment and Slurm entry points;
- `analysis/`: saved-artifact analyses and paired comparisons;
- `tests/`: regression tests for the retained implementation.

Final-workflow entry points are grouped below. They are retained for
reproducibility, not as an instruction to rerun experiments.

| Result family | Main entry points |
| --- | --- |
| Phase-1 BioASQ | `scripts/run_bioasq_isambard.sbatch`, `scripts/run_bioasq_claude_judge.py`, `scripts/evaluate_bioasq_claude_judge.py` |
| Phase-2 artifacts and Probes | `scripts/make_bioasq_phase2_splits.py`, `scripts/run_phase2_bioasq_artifacts.py`, `scripts/train_phase2_linear_probes.py`, `scripts/materialize_frozen_phase2_probes.py` |
| Efficiency and statistical completion | `scripts/benchmark_phase2_uq_efficiency.py`, `analysis/run_phase2_probe_completion.py` |
| Calibration, selective operating points, and audit | `analysis/run_phase2_uq_calibration.py`, `analysis/run_phase2_selective_operating_points.py`, `analysis/prepare_phase2_correctness_audit.py` |
| Summary diagnostics | `scripts/run_summary_length_intervention.py`, `scripts/run_summary_claude_clustering.py`, `analysis/run_summary_clustering_diagnostic.py` |
| Model scale | `analysis/run_gemma3_model_scale_comparison.py`, `analysis/run_gemma3_1b_staged_scale_comparison.py` |
| PubMedQA | `scripts/run_pubmedqa_frozen_probe_transfer.py`, `scripts/run_pubmedqa_v2_semantic_entropy.py`, `scripts/pubmedqa_v2_se_temperature_sensitivity.py` |

The generic `scripts/run_level4.py` and `scripts/check_level4_outputs.py`
remain active because the accepted BioASQ batch wrapper calls them. Early
ArchEHR/Level-3 command-line wrappers and completed RunPod transport scripts
are now in the local-only `.local_archive/` together with other superseded or
exploratory code. That directory is ignored by Git and is not part of the
GitHub repository.

The maintained tree intentionally excludes the third-party
`semantic_uncertainty` reference checkout, early ArchEHR/Level-3 pipelines,
superseded BioASQ judge routes, completed one-off repair utilities, the
post-hoc feature-fusion analysis, and the one-time summary-clustering extension.
Their source paths are preserved inside the dated local archive so the cleanup
is reversible on this machine.

## Data and result boundary

Raw datasets, model outputs, hidden states, local analysis outputs, virtual
environments, caches, and local literature are intentionally outside Git. They
were not moved or deleted during the closeout cleanup. Accepted result
documents retain job IDs, artifact locations, and hashes needed to bind claims
to those local or Isambard artifacts.
Archived code is also outside Git under `.local_archive/`; it must not be
reintroduced into a commit without an explicit decision to restore a route.
The NLL probability-quality supplement is stored at
`archehr_sebaseline/analysis_outputs/bioasq_phase2_uq_calibration_nll_supplement_20260827/`;
its summary binds the unchanged validation/test UQ hashes and records that test
was not used for fitting or selection.

## Historical provenance

Use the archive only when tracing an old decision or reproducing a superseded
workflow:

- `archive_unused/docs/project_history/PHASE2_PROBE_PLAN_FINAL.md`: complete
  frozen plan and milestone history;
- `archive_unused/docs/project_history/experiment_runtime_log_final.md`:
  chronological Slurm/job provenance;
- `archive_unused/docs/project_history/ISAMBARD_COMMANDS_FINAL.md`: completed
  server command notebook;
- `archive_unused/docs/historical_results/`: pilots and non-current results;
- `.local_archive/2026-08-27-code-cleanup/`: local-only source archive, excluded
  from GitHub by `.gitignore`.

Archived commands may contain historical paths and must not be treated as
current run instructions.
