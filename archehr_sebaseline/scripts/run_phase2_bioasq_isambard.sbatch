#!/usr/bin/env bash
#SBATCH --job-name=bioasq-p2-artifacts
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Phase 2: one T=0.1 main answer, blind P(True), single-answer token UQ, and
# fixed hidden states.  This script intentionally never starts high-T sampling
# or NLI/semantic-entropy clustering.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${DATA_PATH:?Set DATA_PATH to BioASQ training13b.json.}"
: "${SPLIT_MANIFEST:?Set SPLIT_MANIFEST to phase2_bioasq_split_manifest.jsonl.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_phase2_gemma3_12b_single_answer_${STAMP}}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-31}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_EXAMPLES_PER_SPLIT="${MAX_EXAMPLES_PER_SPLIT:-}"

module load cray-python || true
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"
GPU_DRIVER_MAJOR="${GPU_DRIVER_VERSION%%.*}"
if [[ ! "${GPU_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine an NVIDIA driver version from nvidia-smi." >&2
  exit 3
fi
if [[ -n "${VENV_PATH:-}" ]]; then
  VENV_SELECTION="override"
elif (( GPU_DRIVER_MAJOR >= 580 )); then
  VENV_PATH="${VENV_PATH_ORIGINAL}"; VENV_SELECTION="original-cuda13"
else
  VENV_PATH="${VENV_PATH_CUDA127}"; VENV_SELECTION="cuda127-cu126"
fi
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Selected virtual environment is missing: ${VENV_PATH}" >&2
  exit 3
fi
source "${VENV_PATH}/bin/activate"
if ! python -c 'import torch, sys; print(f"cuda_available={torch.cuda.is_available()}"); sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "Selected environment cannot access the allocated GPU." >&2
  exit 3
fi
if [[ ! -f "${DATA_PATH}" || ! -f "${SPLIT_MANIFEST}" ]]; then
  echo "Missing data or split manifest." >&2
  exit 2
fi
mkdir -p "${OUTPUT_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"
record_timing() {
  local exit_code=$?
  trap - EXIT
  local ended_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local elapsed_seconds=$(( $(date +%s) - STARTED_EPOCH ))
  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "started_at_utc: ${STARTED_AT_UTC}" \
    "ended_at_utc: ${ended_at_utc}" \
    "elapsed_seconds: ${elapsed_seconds}" \
    "exit_code: ${exit_code}" \
    "model_name: ${MODEL_NAME}" \
    "temperature: ${TEMPERATURE}" \
    "top_p: ${TOP_P}" \
    "top_k: ${TOP_K}" \
    "num_high_temperature_samples: 0" \
    "uq: blind_p_true+verbal_confidence+single_answer_token_scores" \
    "hidden_state_grid: blocks=24,32,40,48;positions=TBG,SLT,LT" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    "environment_selection: ${VENV_SELECTION}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
LOCAL_FILES_ARG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARG=(--local_files_only)
fi
SMOKE_LIMIT_ARG=()
if [[ -n "${MAX_EXAMPLES_PER_SPLIT}" ]]; then
  if [[ ! "${MAX_EXAMPLES_PER_SPLIT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EXAMPLES_PER_SPLIT must be a positive integer when set." >&2
    exit 2
  fi
  SMOKE_LIMIT_ARG=(--max_examples_per_split "${MAX_EXAMPLES_PER_SPLIT}")
fi
python scripts/run_phase2_bioasq_artifacts.py \
  --data_path "${DATA_PATH}" \
  --split_manifest "${SPLIT_MANIFEST}" \
  --output_dir "${OUTPUT_DIR}" \
  --stage all \
  --model_name "${MODEL_NAME}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --max_input_tokens "${MAX_INPUT_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --seed "${SEED}" \
  --device cuda \
  --torch_dtype "${TORCH_DTYPE}" \
  "${SMOKE_LIMIT_ARG[@]}" \
  "${LOCAL_FILES_ARG[@]}"

ACTUAL_EXAMPLES="$(awk 'END { print NR }' "${OUTPUT_DIR}/examples.jsonl")"
if [[ -z "${MAX_EXAMPLES_PER_SPLIT}" && "${ACTUAL_EXAMPLES}" != "3930" ]]; then
  echo "Full Phase-2 run expected 3930 examples, got ${ACTUAL_EXAMPLES}." >&2
  exit 3
fi
python scripts/check_phase2_bioasq_artifacts.py "${OUTPUT_DIR}" \
  --expected_examples "${ACTUAL_EXAMPLES}" \
  --expected_hidden_dimension 3840 \
  --write_report "${OUTPUT_DIR}/health_check.txt"
