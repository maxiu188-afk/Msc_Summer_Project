#!/usr/bin/env bash
#SBATCH --job-name=bioasq-summary-length
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${PHASE2_RUN_DIR:?Set PHASE2_RUN_DIR to the accepted Phase-2 run.}"
: "${LABELS_PATH:?Set LABELS_PATH to the accepted Claude labels.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new summary-length output directory.}"

PROJECT_DIR="${PROJECT_DIR:-${SCRATCHDIR}/final_project/archehr_sebaseline}"
WORKDIR="${SCRATCHDIR}/final_project"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
JOB_TMP_DIR="${SCRATCHDIR}/tmp/summary_length_${SLURM_JOB_ID:-manual_$$}"
EXPECTED_EXAMPLES="${EXPECTED_EXAMPLES:-123}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"

mkdir -p "${JOB_TMP_DIR}"
chmod 700 "${JOB_TMP_DIR}"
export TMPDIR="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"

module load cray-python || true
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"
GPU_DRIVER_MAJOR="${GPU_DRIVER_VERSION%%.*}"
if [[ ! "${GPU_DRIVER_MAJOR}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine NVIDIA driver version." >&2
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
python -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'

if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to use non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
mkdir -p "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

STARTED_AT_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
STARTED_EPOCH="$(date +%s)"
record_timing() {
  local exit_code=$?
  trap - EXIT
  mkdir -p "${OUTPUT_DIR}"
  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "started_at_utc: ${STARTED_AT_UTC}" \
    "ended_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "elapsed_seconds: $(( $(date +%s) - STARTED_EPOCH ))" \
    "exit_code: ${exit_code}" \
    "expected_examples: ${EXPECTED_EXAMPLES}" \
    "max_examples: ${MAX_EXAMPLES}" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    "environment_selection: ${VENV_SELECTION}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
ARGS=(
  --phase2-run-dir "${PHASE2_RUN_DIR}"
  --labels-path "${LABELS_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --expected-examples "${EXPECTED_EXAMPLES}"
  --device cuda
)
if [[ -n "${MAX_EXAMPLES}" ]]; then
  [[ "${MAX_EXAMPLES}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_EXAMPLES must be a positive integer." >&2
    exit 2
  }
  ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  ARGS+=(--local-files-only)
fi
python scripts/run_summary_length_intervention.py "${ARGS[@]}"

python - "${OUTPUT_DIR}" "${MAX_EXAMPLES:-${EXPECTED_EXAMPLES}}" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = int(sys.argv[2])
status = json.loads((root / "status.json").read_text())
if status.get("status") != "complete" or int(status.get("completed", -1)) != expected:
    raise SystemExit(f"Incomplete status: {status}")
for condition in ("current", "short"):
    condition_dir = root / condition
    for name, count in (
        ("examples.jsonl", expected),
        ("prompts.jsonl", expected),
        ("best_generations.jsonl", expected),
        ("generations.jsonl", expected * 10),
        ("cleaned_generations.jsonl", expected * 10),
        ("clusters.jsonl", expected),
    ):
        observed = sum(1 for line in (condition_dir / name).open() if line.strip())
        if observed != count:
            raise SystemExit(f"{condition}/{name}: expected {count}, got {observed}")
    with (condition_dir / "condition_scores.csv").open(newline="") as infile:
        if len(list(csv.DictReader(infile))) != expected:
            raise SystemExit(f"{condition}/condition_scores.csv row mismatch")
print(f"health_check=PASS examples={expected} conditions=2")
PY
