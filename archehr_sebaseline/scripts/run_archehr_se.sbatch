#!/usr/bin/env bash
#SBATCH --job-name=archehr-se
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00

set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Isambard should provide this variable.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${WORKDIR}/archehr_sebaseline"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/archehr_se}"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"

mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"

cd "${PROJECT_DIR}"

module load cray-python || true

VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv_level2}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Python venv not found at: ${VENV_PATH}" >&2
  echo "Create it on the server with:" >&2
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

DATA_PATH="${DATA_PATH:-}"
if [[ -z "${DATA_PATH}" ]]; then
  echo "DATA_PATH is required. Point it to a local ArchEHR-QA-style JSON/JSONL file." >&2
  exit 2
fi

SPLIT="${SPLIT:-dev}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
SEED="${SEED:-13}"
DEVICE="${DEVICE:-cuda}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-1024}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
CLUSTERING_METHOD="${CLUSTERING_METHOD:-nli}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-microsoft/deberta-v2-xlarge-mnli}"
NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY}}"
NLI_TORCH_DTYPE="${NLI_TORCH_DTYPE:-}"
NLI_STRICT_ENTAILMENT="${NLI_STRICT_ENTAILMENT:-0}"

LOCAL_FILES_ARG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" || "${LOCAL_FILES_ONLY}" == "true" ]]; then
  LOCAL_FILES_ARG=(--local_files_only)
  if [[ "${MODEL_NAME}" != /* ]]; then
    echo "Warning: LOCAL_FILES_ONLY=1 and MODEL_NAME='${MODEL_NAME}' does not look like a local absolute path." >&2
    echo "This requires the model to already be cached under HF_HOME=${HF_HOME}." >&2
  fi
fi

TRUST_REMOTE_CODE_ARG=()
if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  TRUST_REMOTE_CODE_ARG=(--trust_remote_code)
fi

MAX_EXAMPLES_ARG=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  MAX_EXAMPLES_ARG=(--max_examples "${MAX_EXAMPLES}")
fi

NLI_LOCAL_FILES_ARG=()
if [[ "${NLI_LOCAL_FILES_ONLY}" == "1" || "${NLI_LOCAL_FILES_ONLY}" == "true" ]]; then
  NLI_LOCAL_FILES_ARG=(--nli_local_files_only)
  if [[ "${NLI_MODEL_NAME}" != /* ]]; then
    echo "Warning: NLI_LOCAL_FILES_ONLY=1 and NLI_MODEL_NAME='${NLI_MODEL_NAME}' does not look like a local absolute path." >&2
    echo "This requires the NLI model to already be cached under HF_HOME=${HF_HOME}." >&2
  fi
fi

NLI_DTYPE_ARG=()
if [[ -n "${NLI_TORCH_DTYPE}" ]]; then
  NLI_DTYPE_ARG=(--nli_torch_dtype "${NLI_TORCH_DTYPE}")
fi

NLI_STRICT_ARG=()
if [[ "${NLI_STRICT_ENTAILMENT}" == "1" || "${NLI_STRICT_ENTAILMENT}" == "true" ]]; then
  NLI_STRICT_ARG=(--nli_strict_entailment)
fi

python "${PROJECT_DIR}/scripts/run_archehr_se.py" \
  --data_path "${DATA_PATH}" \
  --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --num_samples "${NUM_SAMPLES}" \
  "${MAX_EXAMPLES_ARG[@]}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --torch_dtype "${TORCH_DTYPE}" \
  "${LOCAL_FILES_ARG[@]}" \
  "${TRUST_REMOTE_CODE_ARG[@]}" \
  --clustering_method "${CLUSTERING_METHOD}" \
  --nli_model_name "${NLI_MODEL_NAME}" \
  "${NLI_LOCAL_FILES_ARG[@]}" \
  "${NLI_DTYPE_ARG[@]}" \
  "${NLI_STRICT_ARG[@]}" \
  --overwrite

