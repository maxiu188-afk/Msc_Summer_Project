#!/usr/bin/env bash
#SBATCH --job-name=bioasq-judge
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${RUN_DIR:?Set RUN_DIR to the completed BioASQ run directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${WORKDIR}/archehr_sebaseline"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv_isambard}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/llm_judge}"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
SELECTION_SEED="${SELECTION_SEED:-20260715}"

module load cray-python || true
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

RUN_STARTED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
RUN_STARTED_EPOCH="$(date +%s)"

record_run_timing() {
  local exit_code=$?
  local ended_at_utc ended_epoch elapsed_seconds elapsed_hms
  trap - EXIT
  ended_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  ended_epoch="$(date +%s)"
  elapsed_seconds=$((ended_epoch - RUN_STARTED_EPOCH))
  printf -v elapsed_hms '%02d:%02d:%02d' $((elapsed_seconds / 3600)) $(((elapsed_seconds % 3600) / 60)) $((elapsed_seconds % 60))
  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "started_at_utc: ${RUN_STARTED_AT_UTC}" \
    "ended_at_utc: ${ended_at_utc}" \
    "elapsed_seconds: ${elapsed_seconds}" \
    "elapsed_hms: ${elapsed_hms}" \
    "exit_code: ${exit_code}" \
    "judge_model_name: ${JUDGE_MODEL_NAME}" \
    "selection_seed: ${SELECTION_SEED}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_run_timing EXIT

cd "${PROJECT_DIR}"
python scripts/run_bioasq_llm_judge.py \
  --run_dir "${RUN_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --judge_model_name "${JUDGE_MODEL_NAME}" \
  --sample_size 30 \
  --selection_seed "${SELECTION_SEED}" \
  --device cuda \
  --torch_dtype bfloat16 \
  --local_files_only \
  --overwrite
