#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-probe-xfer
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Zero-shot PubMedQA transfer. The Probe bundle was fitted and selected on
# BioASQ only; this job never fits, recalibrates, or selects a Probe.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${PUBMEDQA_DATA_PATH:?Set PUBMEDQA_DATA_PATH to ori_pqal.json.}"
: "${PUBMEDQA_GROUND_TRUTH_PATH:?Set PUBMEDQA_GROUND_TRUTH_PATH to test_ground_truth.json.}"
: "${FROZEN_PROBE_BUNDLE:?Set FROZEN_PROBE_BUNDLE to frozen_probe_bundle.json.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/pubmedqa_frozen_probe_transfer_${STAMP}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${PROJECT_DIR}/analysis_outputs/pubmedqa_frozen_probe_transfer_${STAMP}}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.1}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-31}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
INCLUDE_CONTEXT="${INCLUDE_CONTEXT:-0}"

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
python -c 'import numpy, scipy, sklearn, torch; assert torch.cuda.is_available()'

for required_path in "${PUBMEDQA_DATA_PATH}" "${PUBMEDQA_GROUND_TRUTH_PATH}" "${FROZEN_PROBE_BUNDLE}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required transfer input is missing: ${required_path}" >&2
    exit 2
  fi
done
mkdir -p "${OUTPUT_DIR}" "${ANALYSIS_DIR}" "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"
record_timing() {
  local exit_code=$?
  trap - EXIT
  local ended_at_utc
  local elapsed_seconds
  ended_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  elapsed_seconds=$(( $(date +%s) - STARTED_EPOCH ))
  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "started_at_utc: ${STARTED_AT_UTC}" \
    "ended_at_utc: ${ended_at_utc}" \
    "elapsed_seconds: ${elapsed_seconds}" \
    "exit_code: ${exit_code}" \
    "model_name: ${MODEL_NAME}" \
    "probe_policy: frozen BioASQ Probes; no PubMedQA fitting or tuning" \
    "num_high_temperature_samples: 0" \
    "answer_contract: yes/no/maybe plus explanation" \
    "evidence_mode: $([[ "${INCLUDE_CONTEXT}" == "1" ]] && printf provided_pubmed_abstract_context || printf none)" \
    "environment_selection: ${VENV_SELECTION}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
LOCAL_FILES_ARG=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  LOCAL_FILES_ARG=(--local-files-only)
fi
LIMIT_ARG=()
if [[ -n "${MAX_EXAMPLES}" ]]; then
  if [[ ! "${MAX_EXAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EXAMPLES must be a positive integer when set." >&2
    exit 2
  fi
  LIMIT_ARG=(--max-examples "${MAX_EXAMPLES}")
fi
CONTEXT_ARG=()
if [[ "${INCLUDE_CONTEXT}" == "1" ]]; then
  CONTEXT_ARG=(--include-context)
elif [[ "${INCLUDE_CONTEXT}" != "0" ]]; then
  echo "INCLUDE_CONTEXT must be 0 or 1." >&2
  exit 2
fi

python scripts/run_pubmedqa_frozen_probe_transfer.py \
  --data-path "${PUBMEDQA_DATA_PATH}" \
  --ground-truth-path "${PUBMEDQA_GROUND_TRUTH_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --stage all \
  --model-name "${MODEL_NAME}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-input-tokens "${MAX_INPUT_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --seed "${SEED}" \
  --device cuda \
  --torch-dtype "${TORCH_DTYPE}" \
  "${CONTEXT_ARG[@]}" \
  "${LIMIT_ARG[@]}" \
  "${LOCAL_FILES_ARG[@]}"

ACTUAL_EXAMPLES="$(awk 'END { print NR }' "${OUTPUT_DIR}/examples.jsonl")"
if [[ -z "${MAX_EXAMPLES}" && "${ACTUAL_EXAMPLES}" != "500" ]]; then
  echo "Full PubMedQA transfer expected 500 examples, got ${ACTUAL_EXAMPLES}." >&2
  exit 3
fi

python scripts/evaluate_pubmedqa_frozen_probe_transfer.py \
  --run-dir "${OUTPUT_DIR}" \
  --frozen-probe-bundle "${FROZEN_PROBE_BUNDLE}" \
  --output-dir "${ANALYSIS_DIR}"
