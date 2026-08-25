# Archived Scripts

These scripts are retained for provenance but are no longer maintained or
active run instructions. They were moved here on 2026-08-17 during project
closeout; no source content or result artifact was deleted.

## `legacy_archehr/`

- ArchEHR-QA command-line and Slurm wrappers from before the BioASQ pivot;
- Level-3 wrappers and the generic historical Level-4 Slurm wrapper.

The shared package modules and regression tests remain active under
`archehr_sebaseline/src/` and `archehr_sebaseline/tests/`. The active BioASQ
wrapper still depends on `archehr_sebaseline/scripts/run_level4.py` and
`check_level4_outputs.py`, so those two files were deliberately not archived.

## `legacy_runpod/`

Completed RunPod setup, smoke, pilot, orchestration, and result-packaging
scripts from the early provider workflow, together with their provider-specific
requirements file. Final formal evidence is recorded against Isambard or saved
offline analyses instead.

Archived scripts may contain historical paths, environment assumptions, and
output names. Revalidate them before any use; do not cite them as current
protocols.
