#!/usr/bin/env bash
#SBATCH --job-name=bioasq-se-uq
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Full BioASQ Level 4 generation plus reference UQ and optional self-report UQ.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${DATA_PATH:?Set DATA_PATH to the BioASQ JSON file or Golden directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${WORKDIR}/archehr_sebaseline"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv_isambard}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_summary_gemma3_12b_100x10}"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"

module load cray-python || true
source "${VENV_PATH}/bin/activate"
mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

DATASET="${DATASET:-bioasq_summary}"
SPLIT="${SPLIT:-train13b}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-microsoft/deberta-v2-xlarge-mnli}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY}}"
RUN_SELF_REPORT_UQ="${RUN_SELF_REPORT_UQ:-1}"

if [[ ! -e "${DATA_PATH}" ]]; then
  echo "BioASQ data is missing: ${DATA_PATH}" >&2
  exit 2
fi

LOCAL_FILES_ARG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARG=(--local_files_only)
fi
NLI_LOCAL_FILES_ARG=()
if [[ "${NLI_LOCAL_FILES_ONLY}" == "1" ]]; then
  NLI_LOCAL_FILES_ARG=(--nli_local_files_only)
fi

cd "${PROJECT_DIR}"
python scripts/run_level4.py \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --device cuda \
  --torch_dtype "${TORCH_DTYPE}" \
  "${LOCAL_FILES_ARG[@]}" \
  --clustering_method nli \
  --nli_model_name "${NLI_MODEL_NAME}" \
  --nli_device cuda \
  "${NLI_LOCAL_FILES_ARG[@]}" \
  --show_progress \
  --overwrite

ACTUAL_EXAMPLES="$(awk 'END { print NR }' "${OUTPUT_DIR}/examples.jsonl")"
ACTUAL_GENERATIONS="$(awk 'END { print NR }' "${OUTPUT_DIR}/generations.jsonl")"
python scripts/check_level4_outputs.py "${OUTPUT_DIR}" \
  --expected_examples "${ACTUAL_EXAMPLES}" \
  --expected_num_samples "${NUM_SAMPLES}" \
  --expected_generations "${ACTUAL_GENERATIONS}" \
  --require_cuda \
  --require_nli \
  --write_report "${OUTPUT_DIR}/health_check.txt"

python scripts/evaluate_bioasq_quality.py \
  --run_dir "${OUTPUT_DIR}" \
  --quality_mode reference \
  --bootstrap_samples 1000 \
  --overwrite

if [[ "${RUN_SELF_REPORT_UQ}" == "1" ]]; then
  python scripts/run_self_report_uq.py \
    --run_dir "${OUTPUT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --device cuda \
    --torch_dtype "${TORCH_DTYPE}" \
    --max_input_tokens "${MAX_INPUT_TOKENS}" \
    --overwrite
  python scripts/evaluate_bioasq_quality.py \
    --run_dir "${OUTPUT_DIR}" \
    --quality_mode reference \
    --bootstrap_samples 1000 \
    --overwrite
fi
