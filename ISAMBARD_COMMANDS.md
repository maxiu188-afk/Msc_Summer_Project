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
```

Keep the output directory, health check, quality summary, and Slurm logs as
experiment evidence. Model caches and `.venv_isambard/` are rebuildable local
state and should not be included in result archives or Git commits.
