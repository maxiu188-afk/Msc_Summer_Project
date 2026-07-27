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

## Completed BioASQ summary-length intervention

The paired current-versus-short summary experiment is complete:

```text
5781225  three-question smoke, COMPLETED 0:0, 00:06:28
5781246  full 123-question collection, COMPLETED 0:0, 03:59:14
```

The full job used `afterok:5781225`. Its accepted output is:

```bash
RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_length_full123_seed31_20260725"

cat "$RUN/run_timing.txt"
cat "$RUN/status.json"
cat "$RUN/collection_summary.json"
cat "$RUN/sentence_count_repair.json"
wc -l "$RUN"/{current,short}/{best_generations,generations,clusters}.jsonl
```

The first sentence-count fields split genus abbreviations such as
`C. elegans`. The deterministic repair archived the original score tables
under `archive_sentence_count_pre_fix/` and changed derived length fields only.
Do not regenerate answers or rerun UQ for this correction. Complete results are
in `archehr_sebaseline/docs/summary_length_intervention.md`.

## Completed Gemma 3 4B model-scale result

The 4B scale point reuses the exact Phase-1 seed-31 1,000-question cohort and
changes only the generator checkpoint:

```text
5785064  six-question 2/2/2 smoke, COMPLETED 0:0, 00:03:37
5785065  full 1,000-question run, COMPLETED 0:0, 05:46:48
```

Submission used:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"

SMOKE_ID="$(sbatch --parsable --time=01:00:00 \
  --job-name=gemma3-4b-smoke \
  --export=ALL,SCALE_STAGE=smoke \
  scripts/run_gemma3_4b_phase1_scale_isambard.sbatch)"

FULL_ID="$(sbatch --parsable \
  --job-name=gemma3-4b-full \
  --dependency="afterok:${SMOKE_ID}" \
  --export=ALL,SCALE_STAGE=full \
  scripts/run_gemma3_4b_phase1_scale_isambard.sbatch)"
```

The entrypoint checks the source-derived cohort before loading 4B and checks
the saved `examples.jsonl` again after collection. Do not remove or bypass
`phase1_cohort_alignment.json`. The full output path is:

```bash
RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_gemma3_4b_1000x10_phase1_aligned_seed31_20260726"
```

The full run passed Level-4 health and exact cohort checks. The local Claude
batch plus one bounded retry produced 997/1,000 valid 4B labels. The final
paired comparison uses 990 questions valid for both 4B and 12B:
P(True)-minus-SE AUROC is -0.009 at 4B versus +0.047 at 12B, with paired
4B-minus-12B change -0.056 and 95% CI [-0.103,-0.010]. Full results are in
`archehr_sebaseline/docs/gemma3_model_scale_experiment.md`.

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
- `archehr_sebaseline/docs/summary_length_intervention.md`: completed paired
  summary-length protocol and results.
- `archehr_sebaseline/docs/experiment_runtime_log.md`: job provenance.
