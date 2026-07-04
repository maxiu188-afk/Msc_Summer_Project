# Server Run Guide

This guide covers the maintained server workflow for Level 3/4 runs on Isambard. Use Slurm for model inference; do not run generation on login nodes.

## Upload And Environment

Upload the `archehr_sebaseline` directory to:

```bash
$SCRATCHDIR/final_project/archehr_sebaseline
```

Create the environment:

```bash
cd $SCRATCHDIR/final_project/archehr_sebaseline
module load cray-python || true
python -m venv .venv_level2
source .venv_level2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Cache directories are set by the Slurm scripts:

```bash
HF_HOME=$SCRATCHDIR/final_project/hf_cache
TRANSFORMERS_CACHE=$SCRATCHDIR/final_project/hf_cache
TORCHINDUCTOR_CACHE_DIR=$SCRATCHDIR/final_project/torch_cache
```

Use `LOCAL_FILES_ONLY=0` for first-time downloads. Switch to `LOCAL_FILES_ONLY=1` once the model is cached for reproducible reruns.

## Level 4 Qwen2.5-7B NLI Run

```bash
cd $SCRATCHDIR/final_project/archehr_sebaseline

DATASET=pubmedqa \
DATA_PATH=$SCRATCHDIR/final_project/data/pubmedqa/ori_pqal.json \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_qwen25_7b_nli_50x5 \
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
MAX_EXAMPLES=50 \
NUM_SAMPLES=5 \
MAX_NEW_TOKENS=128 \
DEVICE=cuda \
TORCH_DTYPE=float16 \
LOCAL_FILES_ONLY=1 \
CLUSTERING_METHOD=nli \
NLI_MODEL_NAME=microsoft/deberta-v2-xlarge-mnli \
NLI_LOCAL_FILES_ONLY=1 \
sbatch --time=01:00:00 scripts/run_level4.sbatch
```

Check queue status:

```bash
squeue -u $USER -o "%.18i %.9P %.30j %.2t %.12M %.12l %R"
```

Inspect a completed job:

```bash
sacct -j <JOBID> --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,ReqMem,AllocTRES%80
```

## Health Check

```bash
OUT=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_qwen25_7b_nli_50x5

python scripts/check_level4_outputs.py \
  $OUT \
  --expected_examples 50 \
  --expected_num_samples 5 \
  --expected_generations 250 \
  --require_cuda \
  --require_nli \
  --write_report $OUT/health_check.txt
```

Expected summary fields:

```text
dataset: pubmedqa
examples: 50
generations: 250
device: cuda
requested_clustering_method: nli
clustering_method: nli_bidirectional_entailment
token_scores: True
```
