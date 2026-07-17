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
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_summary_gemma3_12b_100x10}"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"

module load cray-python || true

# CUDA 13 PyTorch requires an NVIDIA R580-or-newer driver.  Older CUDA 12.x
# nodes use the separate cu126 build, which is compatible with a CUDA 12.7
# driver.  VENV_PATH remains an explicit escape hatch for diagnostics.
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"
GPU_DRIVER_MAJOR="${GPU_DRIVER_VERSION%%.*}"
if [[ ! "${GPU_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine an NVIDIA driver version from nvidia-smi: ${GPU_DRIVER_VERSION:-<empty>}" >&2
  exit 3
fi

if [[ -n "${VENV_PATH:-}" ]]; then
  VENV_SELECTION="override"
elif (( GPU_DRIVER_MAJOR >= 580 )); then
  VENV_PATH="${VENV_PATH_ORIGINAL}"
  VENV_SELECTION="original-cuda13"
else
  VENV_PATH="${VENV_PATH_CUDA127}"
  VENV_SELECTION="cuda127-cu126"
fi

if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Selected virtual environment is missing: ${VENV_PATH}" >&2
  exit 3
fi
source "${VENV_PATH}/bin/activate"
TORCH_VERSION="$(python -c 'import torch; print(torch.__version__)')"
TORCH_CUDA_VERSION="$(python -c 'import torch; print(torch.version.cuda or "none")')"
if ! python -c 'import torch, sys; print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Selected environment cannot access the allocated GPU." >&2
  exit 3
fi
echo "[environment] driver=${GPU_DRIVER_VERSION} selection=${VENV_SELECTION} venv=${VENV_PATH} torch=${TORCH_VERSION} torch_cuda=${TORCH_CUDA_VERSION}"
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
TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
BEST_GENERATION_TEMPERATURE="${BEST_GENERATION_TEMPERATURE:-0.1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-microsoft/deberta-v2-xlarge-mnli}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
NLI_LOCAL_FILES_ONLY="${NLI_LOCAL_FILES_ONLY:-${LOCAL_FILES_ONLY}}"
RUN_SELF_REPORT_UQ="${RUN_SELF_REPORT_UQ:-1}"
SEED="${SEED:-31}"
QUALITY_MODE="${QUALITY_MODE:-three_axis}"
REFERENCE_NLI_MODEL="${REFERENCE_NLI_MODEL:-${NLI_MODEL_NAME}}"
REFERENCE_NLI_DEVICE="${REFERENCE_NLI_DEVICE:-cuda}"
REFERENCE_NLI_LOCAL_FILES_ONLY="${REFERENCE_NLI_LOCAL_FILES_ONLY:-${NLI_LOCAL_FILES_ONLY}}"
INCLUDE_EVIDENCE="${INCLUDE_EVIDENCE:-1}"
RELATIVE_RISK_FRACTION="${RELATIVE_RISK_FRACTION:-0.30}"

RUN_STARTED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
RUN_STARTED_EPOCH="$(date +%s)"

record_run_timing() {
  local exit_code=$?
  local ended_at_utc
  local ended_epoch
  local elapsed_seconds
  local elapsed_hms

  trap - EXIT
  ended_at_utc="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  ended_epoch="$(date +%s)"
  elapsed_seconds=$((ended_epoch - RUN_STARTED_EPOCH))
  printf -v elapsed_hms '%02d:%02d:%02d' \
    $((elapsed_seconds / 3600)) \
    $(((elapsed_seconds % 3600) / 60)) \
    $((elapsed_seconds % 60))

  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "slurm_job_name: ${SLURM_JOB_NAME:-bioasq-se-uq}" \
    "host: $(hostname)" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    "environment_selection: ${VENV_SELECTION}" \
    "environment_path: ${VENV_PATH}" \
    "torch_version: ${TORCH_VERSION}" \
    "torch_cuda_version: ${TORCH_CUDA_VERSION}" \
    "started_at_utc: ${RUN_STARTED_AT_UTC}" \
    "ended_at_utc: ${ended_at_utc}" \
    "elapsed_seconds: ${elapsed_seconds}" \
    "elapsed_hms: ${elapsed_hms}" \
    "exit_code: ${exit_code}" \
    "dataset: ${DATASET}" \
    "split: ${SPLIT}" \
    "model_name: ${MODEL_NAME}" \
    "max_examples: ${MAX_EXAMPLES}" \
    "num_samples: ${NUM_SAMPLES}" \
    "max_new_tokens: ${MAX_NEW_TOKENS}" \
    "temperature: ${TEMPERATURE}" \
    "top_p: ${TOP_P}" \
    "top_k: ${TOP_K}" \
    "best_generation_temperature: ${BEST_GENERATION_TEMPERATURE}" \
    "include_evidence: ${INCLUDE_EVIDENCE}" \
    "relative_risk_fraction: ${RELATIVE_RISK_FRACTION}" \
    "seed: ${SEED}" \
    "clustering_method: nli" \
    "run_self_report_uq: ${RUN_SELF_REPORT_UQ}" \
    > "${OUTPUT_DIR}/run_timing.txt"

  exit "${exit_code}"
}

trap record_run_timing EXIT

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
PROMPT_EVIDENCE_ARGS=()
if [[ "${INCLUDE_EVIDENCE}" == "0" ]]; then
  PROMPT_EVIDENCE_ARGS=(--without_evidence)
elif [[ "${INCLUDE_EVIDENCE}" != "1" ]]; then
  echo "INCLUDE_EVIDENCE must be 0 or 1, got: ${INCLUDE_EVIDENCE}" >&2
  exit 2
fi
QUALITY_ARGS=(
  --quality_mode "${QUALITY_MODE}"
  --relative_risk_fraction "${RELATIVE_RISK_FRACTION}"
)
if [[ "${QUALITY_MODE}" == "three_axis" ]]; then
  QUALITY_ARGS+=(
    --reference_nli_model "${REFERENCE_NLI_MODEL}"
    --reference_nli_device "${REFERENCE_NLI_DEVICE}"
  )
  if [[ "${REFERENCE_NLI_LOCAL_FILES_ONLY}" == "1" ]]; then
    QUALITY_ARGS+=(--reference_nli_local_files_only)
  fi
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
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --best_generation_temperature "${BEST_GENERATION_TEMPERATURE}" \
  "${PROMPT_EVIDENCE_ARGS[@]}" \
  --seed "${SEED}" \
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
  "${QUALITY_ARGS[@]}" \
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
    "${QUALITY_ARGS[@]}" \
    --bootstrap_samples 1000 \
    --overwrite
fi
