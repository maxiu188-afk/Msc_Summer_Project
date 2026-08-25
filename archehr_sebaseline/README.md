# Clinical-QA Uncertainty Implementation

This directory contains the maintained implementation used by the completed
BioASQ and PubMedQA uncertainty experiments. The research programme is frozen;
the code remains for auditability and reproducibility.

## Start here

- `../RESEARCH_RESULTS_SYNTHESIS_ZH.md`: thesis-facing results and conclusions;
- `../PROJECT_CLOSEOUT.md`: active evidence and reproducibility map;
- `docs/README.md`: index of the owning protocol and result documents.

The former long implementation/status README is preserved at
`../archive_unused/docs/project_history/archehr_sebaseline_README_legacy.md`.
It contains early ArchEHR, Level-3/4, and migration instructions and is not a
current progress source.

## Maintained layout

```text
archehr_sebaseline/
  src/archehr_sebaseline/   shared package code
  scripts/                  retained final experiment and Slurm entry points
  analysis/                 saved-artifact analyses
  tests/                    regression tests
  docs/                     active protocols and result records
```

The package covers:

- BioASQ and PubMedQA data adapters and prompt contracts;
- low-temperature main answers and ten-sample uncertainty generation;
- PubMedBERT bidirectional semantic clustering and Semantic Entropy;
- blind P(True), token/NLL baselines, verbalized confidence, and cluster count;
- frozen hidden-state P(True)-Probe and Accuracy-Probe evaluation;
- calibration, selective prediction, efficiency, correctness-audit, and
  paired-bootstrap analyses, including validation-fixed deployment operating
  points.

ArchEHR-QA library modules remain in `src/` and continue to have regression
coverage, but their old command-line wrappers are archived because ArchEHR was
not the final research benchmark.

## Environment

Create an independent environment on each machine. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Optional dependency sets are separated by purpose:

- `requirements-isambard-cuda127.txt`: accepted Isambard environment additions;
- `requirements-uq-analysis.txt`: offline UQ analysis;
- `requirements-claude-judge.txt`: Claude judging support.

Do not copy a virtual environment between operating systems or CPU
architectures. CUDA production evidence remains tied to the recorded Isambard
jobs; local macOS tests do not replace it.

## Retained reproducibility entry points

The concise result-to-script map is in `../PROJECT_CLOSEOUT.md`. Key families
are:

- Phase 1: `scripts/run_bioasq_isambard.sbatch` and the Claude judge scripts;
- Phase 2: split, artifact, Probe, efficiency, calibration, validation-fixed
  selective operating-point, and audit scripts;
- Phase 3: summary-length, clustering, and model-scale scripts;
- PubMedQA: frozen transfer, context-v2 SE, and temperature-sensitivity scripts.

`scripts/run_level4.py` and `scripts/check_level4_outputs.py` are retained
because the accepted BioASQ batch wrapper invokes them. Files under
`../archive_unused/scripts/` are provenance only and are not maintained entry
points. The old RunPod-specific requirements file moved with those scripts.

## Tests

From this directory on macOS/Linux:

```bash
PYTHONPATH=src ../.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

On Windows PowerShell, activate the machine-local environment and run:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p "test_*.py" -v
```

Environment-dependent CUDA/model execution is not part of the local unit-test
claim.

## Generated artifacts

The following are intentionally ignored and were not cleaned during document
consolidation:

```text
outputs/
analysis_outputs/
artifacts/
.venv*/
```

Raw datasets, model weights, hidden states, secrets, caches, and local
literature must remain uncommitted. Accepted documents record the provenance
needed to identify formal server outputs.

## Scope boundary

No new experiment is implied by keeping these entry points. The active state is
thesis writing and result interpretation. Any new prompt, model, Probe,
calibration, clustering, temperature, or dataset experiment requires a new
explicit research decision.
