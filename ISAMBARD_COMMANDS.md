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

## Completed Gemma 3 1B staged generation

The 1B run changes only the generator checkpoint. It first runs the same 2/2/2
smoke structure, then a separately authorized staged cohort containing all 200
summary questions and frozen 50-question factoid/list subsets:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"

SMOKE_ID="$(sbatch --parsable --time=01:00:00 \
  --job-name=gemma3-1b-smoke \
  --export=ALL,SCALE_STAGE=smoke \
  scripts/run_gemma3_1b_phase1_scale_isambard.sbatch)"

STAGED_ID="$(sbatch --parsable \
  --job-name=gemma3-1b-staged \
  --dependency="afterok:${SMOKE_ID}" \
  --export=ALL,SCALE_STAGE=staged \
  scripts/run_gemma3_1b_phase1_scale_isambard.sbatch)"

printf 'smoke=%s staged=%s\n' "$SMOKE_ID" "$STAGED_ID"
```

The staged job is submitted immediately but cannot start unless the smoke
finishes with exit code 0. The smoke script exits nonzero if its Level-4 health
check, Phase-1 subset check, or frozen-cohort postflight fails. No recurring or
continuous monitor is started.

Initial submission on 2026-07-27:

```text
5802163  gemma3-1b-smoke, FAILED 3:0 during uncached model loading
5802164  gemma3-1b-staged, never ran and cancelled after dependency failure
```

The full fixed 1B snapshot at revision
`dcc83ea841ab6100d6b47a070329e1ba4cf78752` was then downloaded and verified in
the shared Hugging Face cache. The first replacement exposed a separate code
issue: 1B is a text-only `Gemma3ForCausalLM`, while the shared generator had
routed all Gemma 3 checkpoints through the multimodal processor/model path.
That route is now selected from the saved model config.

```text
5807823  gemma3-1b-smoke-r1, FAILED 3:0 on the incorrect multimodal route
5807824  gemma3-1b-staged-r1, never ran and cancelled
5808905  gemma3-1b-smoke-r2, COMPLETED 0:0, 00:02:06
5808906  gemma3-1b-staged-r2, COMPLETED 0:0, 02:06:35, afterok:5808905
```

The staged output path is:

```bash
RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_gemma3_1b_300x10_phase1_aligned_seed31_20260727"
```

The 200 summary questions form the planned formal scale comparison. Factoid
and list accuracy on 50 questions each are feasibility gates only; do not
submit the remaining 430/270 questions without a separate decision.
Both final outputs passed health and cohort alignment. Correctness batch
`msgbatch_01LCGoC6Mh4EBNKHAq3XEB5U` returned 298/300 valid labels: factoid
8/50 correct and list 3/50 correct. One bounded two-summary retry,
`msgbatch_01AWNB4xt1h86wyfRdm6fq2F`, raised the final total to 299/300 valid
labels. The formal 196-summary analysis is complete; do not submit the
remaining factoid/list questions.

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

## Completed PubMedQA context-v2 Semantic Entropy completion

This is the only currently authorized PubMedQA addition. It reuses the exact
500 accepted Appendix-C v2 prompts and their saved official-decision
`incorrect` labels, and adds only ten high-temperature answers plus the
unchanged PubMedBERT bidirectional NLI clustering needed for discrete Semantic
Entropy. It does not regenerate the low-temperature answer or rerun Probes,
P(True), calibration, prompt selection, or correctness judging.

```text
5921808  three-question runnability smoke, COMPLETED 0:0, 00:02:28
5921809  formal 500-question job, COMPLETED 0:0, 04:59:51,
         afterok:5921808
source revision: d778553
```

Both jobs passed their postflight checks; do not resubmit them. The CUDA 12.6
environment was selected explicitly because the CUDA 13 environment lacked
the accepted analysis dependencies (`scipy` and `scikit-learn`). Accepted
source and destination paths are:

```bash
SOURCE_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"
SOURCE_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"
SMOKE_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_v2_se_smoke3_seed31_20260805"
FULL_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_v2_se_full500_seed31_20260805"
FULL_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_v2_se_full500_seed31_20260805"
```

The formal result is discrete-SE AUROC `0.5884` / AP `0.3674` on 500 frozen
labels. Acceptance confirmed 500 examples, 5,000 generations, matching source
hashes, zero empty answers, and zero generations at the token limit. The result
and artifact hashes are recorded in
`archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md` and
`archehr_sebaseline/docs/experiment_runtime_log.md`.

## Queued PubMedQA-v2 SE temperature sensitivity

This diagnostic reuses accepted `T=1.0` and adds only `T=0.7/1.3` on a fixed
200-question subset stratified by frozen `incorrect × yes/no/maybe`. Do not
submit another temperature or regenerate the baseline arm.

```text
5931810  six-question runnability smoke, pending
5931811  formal 200-question job, pending with afterok:5931810
source revision: 5a302c1
```

Accepted input and new output roots are:

```bash
SOURCE_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"
SOURCE_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722"
BASELINE_SE_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_v2_se_full500_seed31_20260805"
TEMP_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_v2_se_temperature_sensitivity_200_seed31_20260806"
TEMP_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_v2_se_temperature_sensitivity_200_seed31_20260806"
```

The batch creates the manifest once, runs both new temperature arms in separate
Python processes, reuses the matching baseline rows, applies 20,000 paired
bootstrap resamples, and fails unless both 200-question/2,000-generation arms
and the comparison summary pass postflight. Until then, retain the study as
queued rather than citing a temperature effect.

## Monitor jobs

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
sacct -j 5750742,5750745 \
  --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80

squeue -j 5921808,5921809 -o "%.18i %.30j %.2t %.12M %.20R %.24E"
sacct -j 5921808,5921809 \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,AllocTRES%80

squeue -j 5931810,5931811 -o "%.18i %.30j %.2t %.12M %.20R %.24E"
sacct -j 5931810,5931811 \
  --format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS,AllocTRES%80

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

SE_RUN="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/pubmedqa_context_v2_se_full500_seed31_20260805"
SE_ANALYSIS="$SCRATCHDIR/final_project/archehr_sebaseline/analysis_outputs/pubmedqa_context_v2_se_full500_seed31_20260805"

cat "$SE_RUN/run_timing.txt"
wc -l "$SE_RUN"/{examples,clusters}.jsonl \
  "$SE_RUN"/{generations,cleaned_generations}.jsonl
cat "$SE_ANALYSIS/metrics.csv"
sha256sum "$SE_RUN/se_scores.csv" "$SE_ANALYSIS"/{metrics.csv,summary.json}
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
