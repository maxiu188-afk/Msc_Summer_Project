#!/usr/bin/env bash
#SBATCH --job-name=bioasq-ref-nli
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Re-evaluate completed BioASQ outputs with three-axis ideal-answer NLI.
# This deliberately does not regenerate model answers or rerun self-report UQ.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to an existing completed BioASQ run directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
REFERENCE_NLI_MODEL="${REFERENCE_NLI_MODEL:-microsoft/deberta-v2-xlarge-mnli}"
REFERENCE_NLI_LOCAL_FILES_ONLY="${REFERENCE_NLI_LOCAL_FILES_ONLY:-1}"
REFERENCE_NLI_MAX_INPUT_TOKENS="${REFERENCE_NLI_MAX_INPUT_TOKENS:-512}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
TIMING_PATH="${TIMING_PATH:-${OUTPUT_DIR}/reference_nli_timing.txt}"

module load cray-python || true

# CUDA 13 PyTorch needs an R580-or-newer driver; use the cu126 environment on
# older CUDA 12.x nodes. VENV_PATH remains available as an explicit override.
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"
GPU_DRIVER_MAJOR="${GPU_DRIVER_VERSION%%.*}"
if [[ ! "${GPU_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine the NVIDIA driver version: ${GPU_DRIVER_VERSION:-<empty>}" >&2
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

if [[ ! -f "${OUTPUT_DIR}/examples.jsonl" || ! -f "${OUTPUT_DIR}/generations.jsonl" ]]; then
  echo "Completed BioASQ artifacts are missing from: ${OUTPUT_DIR}" >&2
  exit 2
fi

mkdir -p "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

STARTED_AT_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
STARTED_EPOCH="$(date +%s)"
record_timing() {
  local exit_code=$?
  local ended_epoch elapsed_seconds
  trap - EXIT
  ended_epoch="$(date +%s)"
  elapsed_seconds=$((ended_epoch - STARTED_EPOCH))
  cat > "${TIMING_PATH}" <<EOF
slurm_job_id: ${SLURM_JOB_ID:-manual}
slurm_job_name: ${SLURM_JOB_NAME:-bioasq-ref-nli}
host: $(hostname)
gpu_driver_version: ${GPU_DRIVER_VERSION}
environment_selection: ${VENV_SELECTION}
environment_path: ${VENV_PATH}
torch_version: ${TORCH_VERSION}
torch_cuda_version: ${TORCH_CUDA_VERSION}
reference_nli_model: ${REFERENCE_NLI_MODEL}
reference_nli_local_files_only: ${REFERENCE_NLI_LOCAL_FILES_ONLY}
reference_nli_max_input_tokens: ${REFERENCE_NLI_MAX_INPUT_TOKENS}
quality_mode: three_axis
started_at_utc: ${STARTED_AT_UTC}
ended_at_utc: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
elapsed_seconds: ${elapsed_seconds}
exit_code: ${exit_code}
EOF
  exit "${exit_code}"
}
trap record_timing EXIT

ARGS=(
  --run_dir "${OUTPUT_DIR}"
  --quality_mode three_axis
  --reference_nli_model "${REFERENCE_NLI_MODEL}"
  --reference_nli_device cuda
  --reference_nli_max_input_tokens "${REFERENCE_NLI_MAX_INPUT_TOKENS}"
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
  --overwrite
)
if [[ "${REFERENCE_NLI_LOCAL_FILES_ONLY}" == "1" ]]; then
  ARGS+=(--reference_nli_local_files_only)
fi

cd "${PROJECT_DIR}"
python scripts/evaluate_bioasq_quality.py "${ARGS[@]}"
