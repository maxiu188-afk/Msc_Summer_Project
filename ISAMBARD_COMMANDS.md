# Isambard Command Reference

This is the single active operational guide for the current BioASQ/PubMedQA
work. Completed Phase-1, Runpod, ArchEHR, and superseded setup commands are in
`archive_unused/docs/project_history/` and must not be submitted unchanged.

## Connect from the local machine

```bash
clifton auth
clifton ssh-config write
ssh b6u.aip2.isambard
```

Upload source or public inputs from the local machine, not from an Isambard
login shell:

```bash
scp <local-file> b6u.aip2.isambard:~
scp -r <local-directory> b6u.aip2.isambard:~/bioasq_upload/
```

## Active project

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
source .venv_isambard/bin/activate
```

The PubMedQA batch script chooses `.venv_isambard` on driver 580+ nodes and
`.venv_isambard_cuda127` otherwise. Do not replace either PyTorch build during
an experiment. Before a first model download, verify the Hugging Face account:

```bash
hf auth whoami
```

## Current PubMedQA context result

The active result is the completed Appendix-C context v2 run. The question-only
result is archived under `archive_unused/docs/historical_results/` because it
omitted the article evidence that defines the official PubMedQA decision.
Context v1 remains the direct historical baseline. There is no PubMedQA Probe
fitting, calibration, threshold tuning, or feature selection.

```text
5748203  context v1 smoke (3 questions), COMPLETED 0:0, 00:01:00
5748205  context v1 full (500 questions), COMPLETED 0:0, 00:37:43
5750742  Appendix-C v2 smoke (3 questions), COMPLETED 0:0, 00:00:46
5750745  Appendix-C v2 full (500 questions), COMPLETED 0:0, 00:40:16
```

The v2 full job used `afterok:5750742`, so it started only after the smoke
succeeded. The submission pattern for a future separately authorized run is:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"

INCLUDE_CONTEXT=1 MAX_EXAMPLES=3 \
PUBMEDQA_DATA_PATH="$SCRATCHDIR/final_project/data/pubmedqa/data/ori_pqal.json" \
PUBMEDQA_GROUND_TRUTH_PATH="$SCRATCHDIR/final_project/data/pubmedqa/data/test_ground_truth.json" \
FROZEN_PROBE_BUNDLE="$PWD/analysis_outputs/bioasq_phase2_frozen_probe_bundle_seed31/frozen_probe_bundle.json" \
sbatch scripts/run_pubmedqa_frozen_probe_transfer_isambard.sbatch
```

Do not remove `MAX_EXAMPLES=3` until the smoke has valid explained leading
labels, self-report rows, hidden-state layout, and transfer tables. The full
protocol and acceptance checks are in
`archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`.

## Monitor jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
sacct -j 5750742,5750745 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80

tail -f pubmedqa-probe-xfer-5750742.out
cat pubmedqa-probe-xfer-5750742.err
```

A failed job exits without continuing later stages because the batch script
uses `set -euo pipefail`.

## Inspect accepted outputs

```bash
RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"
ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"

cat "$RUN/run_timing.txt"
wc -l "$RUN/examples.jsonl" "$RUN/best_generations.jsonl"
head -n 2 "$RUN/uq_baselines/self_report_examples.csv"
cat "$ANALYSIS/transfer_summary.json"
```

Record completed job state and runtime in
`archehr_sebaseline/docs/experiment_runtime_log.md`. Research decisions and
metrics belong in the owning protocol/result document, not in this command
reference.

## Active document entry points

- `PHASE2_PROBE_PLAN.md`: frozen Probe choices and research boundaries.
- `archehr_sebaseline/docs/README.md`: active document index.
- `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`: current transfer
  protocol and results.
- `archehr_sebaseline/docs/experiment_runtime_log.md`: job provenance.
