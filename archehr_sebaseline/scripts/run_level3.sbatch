#!/usr/bin/env bash
#SBATCH --job-name=archehr-se-l3
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Isambard should provide this variable.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${WORKDIR}/archehr_sebaseline"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/level3}"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"

cd "${WORKDIR}"

module load cray-python || true

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv_level2}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Python venv not found at: ${VENV_PATH}" >&2
  echo "Create it on Isambard with:" >&2
  echo "  cd ${PROJECT_DIR}" >&2
  echo "  module load cray-python || true" >&2
  echo "  python -m venv .venv_level2" >&2
  echo "  source .venv_level2/bin/activate" >&2
  echo "  python -m pip install --upgrade pip" >&2
  echo "  python -m pip install -r requirements.txt" >&2
  exit 2
fi
source "${VENV_PATH}/bin/activate"
python --version

DATASET="${DATASET:-fake}"
DATA_PATH="${DATA_PATH:-}"
SPLIT="${SPLIT:-dev}"
MODEL_NAME="${MODEL_NAME:-sshleifer/tiny-gpt2}"
MAX_EXAMPLES="${MAX_EXAMPLES:-10}"
NUM_SAMPLES="${NUM_SAMPLES:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-96}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

LOCAL_FILES_ARG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" || "${LOCAL_FILES_ONLY}" == "true" ]]; then
  LOCAL_FILES_ARG=(--local_files_only)
  if [[ "${MODEL_NAME}" != /* ]]; then
    echo "Warning: LOCAL_FILES_ONLY=1 and MODEL_NAME='${MODEL_NAME}' does not look like a local absolute path." >&2
    echo "This will only work if the HuggingFace model is already cached under HF_HOME=${HF_HOME}." >&2
  fi
fi

DATA_PATH_ARG=()
if [[ -n "${DATA_PATH}" ]]; then
  DATA_PATH_ARG=(--data_path "${DATA_PATH}")
fi

python "${PROJECT_DIR}/scripts/run_level3.py" \
  --dataset "${DATASET}" \
  "${DATA_PATH_ARG[@]}" \
  --split "${SPLIT}" \
  --model_name "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --device "${DEVICE}" \
  --torch_dtype "${TORCH_DTYPE}" \
  "${LOCAL_FILES_ARG[@]}" \
  --overwrite
