#!/usr/bin/env bash
#SBATCH --job-name=bioasq-best
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Add paper-protocol low-temperature best generations to an existing SE run.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${RUN_DIR:?Set RUN_DIR to an existing BioASQ SE result directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-31}"
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
python scripts/generate_bioasq_best_generations.py \
  --run_dir "${RUN_DIR}" --model_name "${MODEL_NAME}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --temperature "${TEMPERATURE}" --top_p "${TOP_P}" --top_k "${TOP_K}" \
  --seed "${SEED}" --device cuda --torch_dtype "${TORCH_DTYPE}" --local_files_only --overwrite

printf '%s\n' \
  "temperature: ${TEMPERATURE}" "top_p: ${TOP_P}" "top_k: ${TOP_K}" \
  "seed: ${SEED}" "model_name: ${MODEL_NAME}" "environment_selection: ${VENV_SELECTION}" \
  > "${RUN_DIR}/best_generation_timing.txt"
