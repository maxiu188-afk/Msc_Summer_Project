#!/usr/bin/env bash
#SBATCH --job-name=bioasq-p2-uq-eff
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Incremental UQ-cost benchmark on the Claude-labelled Phase-2 test examples.
# The saved main answers are inputs and are never regenerated.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${PHASE2_RUN_DIR:?Set PHASE2_RUN_DIR to the completed Phase-2 run.}"
: "${LABELS_PATH:?Set LABELS_PATH to claude_generation_labels.csv.}"
: "${FROZEN_PROBE_BUNDLE:?Set FROZEN_PROBE_BUNDLE to frozen_probe_bundle.json.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new benchmark directory.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-pritamdeka/PubMedBERT-MNLI-MedNLI}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-50}"
SEED="${SEED:-31}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
INCLUDE_SE="${INCLUDE_SE:-1}"
EXPECTED_EXAMPLES="${EXPECTED_EXAMPLES:-384}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"

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
for required_path in \
  "${PHASE2_RUN_DIR}/examples.jsonl" \
  "${PHASE2_RUN_DIR}/prompts.jsonl" \
  "${PHASE2_RUN_DIR}/best_generations.jsonl" \
  "${LABELS_PATH}" \
  "${FROZEN_PROBE_BUNDLE}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing required input: ${required_path}" >&2
    exit 2
  fi
done
if [[ -e "${OUTPUT_DIR}" && -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Refusing to use non-empty output directory: ${OUTPUT_DIR}" >&2
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
    "expected_examples: ${EXPECTED_EXAMPLES}" \
    "max_examples: ${MAX_EXAMPLES}" \
    "include_semantic_entropy: ${INCLUDE_SE}" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    "environment_selection: ${VENV_SELECTION}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
ARGS=(
  --run-dir "${PHASE2_RUN_DIR}"
  --labels-path "${LABELS_PATH}"
  --frozen-probe-bundle "${FROZEN_PROBE_BUNDLE}"
  --output-dir "${OUTPUT_DIR}"
  --model-name "${MODEL_NAME}"
  --nli-model-name "${NLI_MODEL_NAME}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --num-samples "${NUM_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --seed "${SEED}"
  --device cuda
  --torch-dtype "${TORCH_DTYPE}"
  --expected-examples "${EXPECTED_EXAMPLES}"
)
if [[ -n "${MAX_EXAMPLES}" ]]; then
  if [[ ! "${MAX_EXAMPLES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MAX_EXAMPLES must be a positive integer when set." >&2
    exit 2
  fi
  ARGS+=(--max-examples "${MAX_EXAMPLES}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  ARGS+=(--local-files-only)
fi
if [[ "${INCLUDE_SE}" == "1" ]]; then
  ARGS+=(--include-semantic-entropy)
fi

python scripts/benchmark_phase2_uq_efficiency.py "${ARGS[@]}"

python - "${OUTPUT_DIR}" "${EXPECTED_EXAMPLES}" "${INCLUDE_SE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected = int(sys.argv[2])
include_se = sys.argv[3] == "1"
status = json.loads((output_dir / "benchmark_status.json").read_text())
if status.get("status") != "complete" or int(status.get("completed_examples", -1)) != expected:
    raise SystemExit(f"Incomplete benchmark status: {status}")
with (output_dir / "efficiency_summary.csv").open(newline="") as infile:
    methods = {row["method"] for row in csv.DictReader(infile)}
required = {
    "p_true_probe",
    "accuracy_probe",
    "both_probes_joint",
    "blind_p_true",
    "normalized_nll_10_samples",
}
if include_se:
    required |= {"discrete_semantic_entropy", "num_clusters"}
if not required <= methods:
    raise SystemExit(f"Missing efficiency methods: {sorted(required - methods)}")
with (output_dir / "uq_scores.csv").open(newline="") as infile:
    rows = list(csv.DictReader(infile))
if len(rows) != expected:
    raise SystemExit(f"Expected {expected} UQ score rows, got {len(rows)}")
print(f"health_check=PASS examples={expected} methods={len(methods)}")
PY
