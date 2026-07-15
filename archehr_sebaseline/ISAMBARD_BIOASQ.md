# Isambard BioASQ First-Run Guide

This is the active launch guide for a fresh Isambard BioASQ run. It is not a
rerun of the completed RunPod batch. The job generates a new BioASQ run, checks
its Level 4 artifacts, evaluates reference quality, and (by default) computes
the two remaining simple UQ baselines: verbalized confidence and P(True).

## Upload payload and data

Upload the curated source archive to `$SCRATCHDIR/final_project/` and extract
it there. The archive deliberately excludes model caches, outputs, virtual
environments, and historical RunPod results.

The local workspace currently has no public BioASQ data directory, so upload
the public data separately and keep this layout:

```text
$SCRATCHDIR/final_project/
├── archehr_sebaseline/
└── data/
    ├── BioASQ-training13b/training13b.json
    └── Task13BGoldenEnriched/
```

For the first run, begin with the training summary slice. The Golden directory
is needed later for summary/factoid/list comparisons.

## First-time setup

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
module load cray-python || true
python -m venv .venv_isambard
source .venv_isambard/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
hf auth login
hf auth whoami
python -m unittest discover tests
```

Before submitting a GPU job, confirm that the installed PyTorch build can see
the allocated GPU from a short interactive allocation or a site-supported GPU
test command. Do not replace a CUDA build with CPU PyTorch if this check fails.

## Submit the first BioASQ run

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"

DATASET=bioasq_summary \
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
SPLIT=train13b \
OUTPUT_DIR="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10" \
MAX_EXAMPLES=100 \
NUM_SAMPLES=10 \
LOCAL_FILES_ONLY=0 \
NLI_LOCAL_FILES_ONLY=0 \
RUN_SELF_REPORT_UQ=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

`LOCAL_FILES_ONLY=0` permits the first authenticated download of Gemma and the
open NLI model. After the cache is populated, use `1` for both variables on
later runs. The batch script prints Level 4 generation/NLI progress and stops
on any failure. It writes the reference UQ results first, then the two
model-backed baselines, and finally refreshes the common BioASQ comparison.

## Verify and retrieve results

```bash
squeue -u "$USER" -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80

OUT="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10"
cat "$OUT/health_check.txt"
cat "$OUT/bioasq_eval/bioasq_eval_summary.json"
cat "$OUT/run_timing.txt"
```

The final comparison includes discrete/weighted SE, token log-probability,
token entropy, sequence NLL, verbalized confidence, and P(True). Keep outputs,
health checks, `run_timing.txt`, and Slurm logs. Record each completed formal run
in `docs/experiment_runtime_log.md`; do not archive model caches or virtual
environments as experiment evidence.

## Independent judge validation and matched repeat

The completed 100x10 run can be validated on a fixed, quality-stratified
30-question subset with the cached independent Qwen judge. All 300 prompts,
raw responses, rubric scores, and comparison statistics are retained:

```bash
RUN_DIR="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10" \
JUDGE_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
sbatch scripts/run_bioasq_llm_judge.sbatch
```

For an independently sampled repeat, keep every setting unchanged except the
generation seed and output directory:

```bash
DATASET=bioasq_summary \
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
SPLIT=train13b \
OUTPUT_DIR="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10_seed47" \
MAX_EXAMPLES=100 \
NUM_SAMPLES=10 \
SEED=47 \
LOCAL_FILES_ONLY=1 \
NLI_LOCAL_FILES_ONLY=1 \
RUN_SELF_REPORT_UQ=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

Both scripts write their own `run_timing.txt`. The repeat is matched to the
original seed-31 configuration; changing the model, data order, precision,
sample count, answer length, or evaluation stages would no longer be a pure
seed replication.
