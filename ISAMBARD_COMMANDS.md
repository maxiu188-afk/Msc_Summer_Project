# Isambard Command Reference

This is the concise operational reference for the BioASQ SE/UQ work. It is
intentionally separate from `archehr_sebaseline/ISAMBARD_BIOASQ.md`: use this
file for common commands, and the package guide for a fresh experimental run.

## On the local machine

Authenticate and create/update the SSH configuration used by the Isambard
login alias:

```bash
clifton auth
clifton ssh-config write
ssh b6u.aip2.isambard
```

To upload a source archive or a data directory, run `scp` from the local
machine, not from an Isambard login shell:

```bash
scp <local-file-or-directory> b6u.aip2.isambard:~
scp -r <local-directory> b6u.aip2.isambard:~/bioasq_upload/
```

## Project location and environment

After logging in, the project and public BioASQ data live under scratch:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
source .venv_isambard/bin/activate
```

For a new environment, follow the setup section in
`archehr_sebaseline/ISAMBARD_BIOASQ.md`. Check the authenticated Hugging Face
account before the first model download:

```bash
hf auth whoami
```

If Gemma 3 reports a missing image processor, install the two processor
dependencies into this same virtual environment, then confirm that the
processor loads:

```bash
python -m pip install Pillow torchvision
python - <<'PY'
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("google/gemma-3-12b-it")
print(type(processor).__name__)
PY
```

Do not replace the existing PyTorch build while applying this repair.

## Submit the BioASQ Summary main run

Use the complete command in `archehr_sebaseline/ISAMBARD_BIOASQ.md`. Its
important settings are 100 questions, 10 generations per question, NLI SE,
reference-quality evaluation, verbalized confidence, and P(True). The command
returns a Slurm job ID:

```bash
sbatch scripts/run_bioasq_isambard.sbatch
```

## Active temperature-sensitivity repeat

The 2026-07-16 follow-up keeps the 100x10 workload, 192-token answer cap,
model, NLI clustering, and self-report stages unchanged. It changes only the
generation temperature from the historical default of `0.8` to `1.0` and runs
both established seeds. The batch script now records `temperature` and `top_p`
in `run_timing.txt`.

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json"

SEED=31 TEMPERATURE=1.0 TOP_P=0.9 MAX_NEW_TOKENS=192 \
OUTPUT_DIR="$PWD/outputs/bioasq_summary_gemma3_12b_100x10_temp1p0_seed31" \
DATA_PATH="$DATA_PATH" LOCAL_FILES_ONLY=1 NLI_LOCAL_FILES_ONLY=1 \
sbatch scripts/run_bioasq_isambard.sbatch

SEED=47 TEMPERATURE=1.0 TOP_P=0.9 MAX_NEW_TOKENS=192 \
OUTPUT_DIR="$PWD/outputs/bioasq_summary_gemma3_12b_100x10_temp1p0_seed47" \
DATA_PATH="$DATA_PATH" LOCAL_FILES_ONLY=1 NLI_LOCAL_FILES_ONLY=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

These jobs are currently pending result collection. Treat them as a controlled
UQ sensitivity check, not evidence of improvement until both seeds pass the
health check and are evaluated with the same citation-aware target.

## Monitor and inspect a job

Replace `<JOBID>` with the ID returned by `sbatch`.

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
sacct --starttime=today --format=JobID,JobName,State,ExitCode,Elapsed
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80

tail -f "bioasq-se-uq-<JOBID>.out"
cat "bioasq-se-uq-<JOBID>.err"
```

`PENDING` means the job is waiting for resources; `RUNNING` means the GPU job
is active. The batch script prints generation and NLI progress. A failed job
does not continue consuming a GPU because the script uses `set -euo pipefail`.

## Verify a completed run

```bash
OUT="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10"
cat "$OUT/health_check.txt"
cat "$OUT/bioasq_eval/bioasq_eval_summary.json"
cat "$OUT/run_timing.txt"
```

Keep the output directory, health check, quality summary, and Slurm logs as
experiment evidence. Formal BioASQ jobs write `run_timing.txt` on both success
and failure; add completed runs to
`archehr_sebaseline/docs/experiment_runtime_log.md` for future estimates. Model
caches and `.venv_isambard/` are rebuildable local state and should not be
included in result archives or Git commits.

## Validate quality and repeat the baseline

The reference validation jobs have completed:

```text
5660345  bioasq-judge  COMPLETED  00:20:56
5660346  bioasq-se-uq  COMPLETED  03:18:31
```

Their result summary is
`archehr_sebaseline/docs/bioasq_isambard_results_20260715.md`. The commands
below are retained for provenance and future explicitly planned repeats.

The package guide contains the full commands for the fixed 30-question Qwen
judge and the matched seed-47 repeat. In short, submit:

```bash
RUN_DIR="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10" \
sbatch scripts/run_bioasq_llm_judge.sbatch

SEED=47 \
OUTPUT_DIR="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10_seed47" \
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
LOCAL_FILES_ONLY=1 NLI_LOCAL_FILES_ONLY=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

The repeat otherwise inherits the same 100-example, 10-sample, Gemma 3 12B,
NLI, self-report, CUDA, and bfloat16 settings.
