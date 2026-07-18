#!/usr/bin/env bash
#SBATCH --job-name=bioasq-self-report
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Recompute paper-protocol self-report UQ on existing BioASQ artifacts.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${RUN_DIR:?Set RUN_DIR to an existing BioASQ SE result directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"

module load cray-python || true
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"
GPU_DRIVER_MAJOR="${GPU_DRIVER_VERSION%%.*}"
if [[ ! "${GPU_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine an NVIDIA driver version." >&2
  exit 3
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  VENV_SELECTION="override"
elif (( GPU_DRIVER_MAJOR >= 580 )); then
  VENV_PATH="${VENV_PATH_ORIGINAL}"; VENV_SELECTION="original-cuda13"
else
  VENV_PATH="${VENV_PATH_CUDA127}"; VENV_SELECTION="cuda127-cu126"
fi
source "${VENV_PATH}/bin/activate"
export HF_HOME="${WORKDIR}/hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${WORKDIR}/torch_cache"
export PYTHONUNBUFFERED=1

cd "${PROJECT_DIR}"
python scripts/run_self_report_uq.py \
  --run_dir "${RUN_DIR}" --model_name "${MODEL_NAME}" \
  --device cuda --torch_dtype "${TORCH_DTYPE}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" --local_files_only --overwrite

printf '%s\n' \
  "model_name: ${MODEL_NAME}" "environment_selection: ${VENV_SELECTION}" \
  "self_report_answer_source: best_generation_low_temperature" \
  "p_true_high_temperature_context: cleaned_generations.jsonl" \
  > "${RUN_DIR}/self_report_timing.txt"
