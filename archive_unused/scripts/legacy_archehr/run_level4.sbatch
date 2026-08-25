#!/usr/bin/env bash
#SBATCH --job-name=archehr-se-l4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00

set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Isambard should provide this variable.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${WORKDIR}/archehr_sebaseline"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/level4}"
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

DATASET="${DATASET:-pubmedqa}"
DATA_PATH="${DATA_PATH:-}"
SPLIT="${SPLIT:-dev}"
MODEL_NAME="${MODEL_NAME:-sshleifer/tiny-gpt2}"
MAX_EXAMPLES="${MAX_EXAMPLES:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-5}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE="${DEVICE:-cuda}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-nli}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-microsoft/deberta-v2-xlarge-mnli}"
NLI_DEVICE="${NLI_DEVICE:-${DEVICE}}"
NLI_MAX_INPUT_TOKENS="${NLI_MAX_INPUT_TOKENS:-512}"
NLI_TORCH_DTYPE="${NLI_TORCH_DTYPE:-}"
NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY}}"
NLI_TRUST_REMOTE_CODE="${NLI_TRUST_REMOTE_CODE:-0}"
STRICT_ENTAILMENT="${STRICT_ENTAILMENT:-0}"
CONDITION_ON_QUESTION="${CONDITION_ON_QUESTION:-1}"

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

NLI_LOCAL_FILES_ARG=()
if [[ "${NLI_LOCAL_FILES_ONLY}" == "1" || "${NLI_LOCAL_FILES_ONLY}" == "true" ]]; then
  NLI_LOCAL_FILES_ARG=(--nli_local_files_only)
  if [[ "${NLI_MODEL_NAME}" != /* ]]; then
    echo "Warning: NLI_LOCAL_FILES_ONLY=1 and NLI_MODEL_NAME='${NLI_MODEL_NAME}' does not look like a local absolute path." >&2
    echo "This will only work if the NLI model is already cached under HF_HOME=${HF_HOME}." >&2
  fi
fi

NLI_DTYPE_ARG=()
if [[ -n "${NLI_TORCH_DTYPE}" ]]; then
  NLI_DTYPE_ARG=(--nli_torch_dtype "${NLI_TORCH_DTYPE}")
fi

NLI_TRUST_REMOTE_CODE_ARG=()
if [[ "${NLI_TRUST_REMOTE_CODE}" == "1" || "${NLI_TRUST_REMOTE_CODE}" == "true" ]]; then
  NLI_TRUST_REMOTE_CODE_ARG=(--nli_trust_remote_code)
fi

STRICT_ENTAILMENT_ARG=()
if [[ "${STRICT_ENTAILMENT}" == "1" || "${STRICT_ENTAILMENT}" == "true" ]]; then
  STRICT_ENTAILMENT_ARG=(--strict_entailment)
fi

QUESTION_CONDITION_ARG=()
if [[ "${CONDITION_ON_QUESTION}" == "0" || "${CONDITION_ON_QUESTION}" == "false" ]]; then
  QUESTION_CONDITION_ARG=(--no_condition_on_question)
fi

python "${PROJECT_DIR}/scripts/run_level4.py" \
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
  --clustering_method "${CLUSTERING_METHOD}" \
  --nli_model_name "${NLI_MODEL_NAME}" \
  --nli_device "${NLI_DEVICE}" \
  --nli_max_input_tokens "${NLI_MAX_INPUT_TOKENS}" \
  "${NLI_DTYPE_ARG[@]}" \
  "${NLI_LOCAL_FILES_ARG[@]}" \
  "${NLI_TRUST_REMOTE_CODE_ARG[@]}" \
  "${STRICT_ENTAILMENT_ARG[@]}" \
  "${QUESTION_CONDITION_ARG[@]}" \
  --overwrite
