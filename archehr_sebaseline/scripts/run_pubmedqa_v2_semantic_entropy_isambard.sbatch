#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-v2-se
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# One bounded addition: ten-sample Semantic Entropy for the already accepted
# PubMedQA Appendix-C context-v2 condition. This job does not regenerate the
# low-temperature answer or rerun Probes, P(True), calibration, or judging.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${PUBMEDQA_V2_RUN_DIR:?Set PUBMEDQA_V2_RUN_DIR to the accepted v2 run.}"
: "${PUBMEDQA_V2_ANALYSIS_DIR:?Set PUBMEDQA_V2_ANALYSIS_DIR to its accepted analysis.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new SE output directory.}"
: "${ANALYSIS_DIR:?Set ANALYSIS_DIR to a new SE analysis directory.}"

PROJECT_DIR="${PROJECT_DIR:-${SCRATCHDIR}/final_project/archehr_sebaseline}"
WORKDIR="${SCRATCHDIR}/final_project"
VENV_PATH_ORIGINAL="${PROJECT_DIR}/.venv_isambard"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
JOB_TMP_DIR="${SCRATCHDIR}/tmp/pubmedqa_v2_se_${SLURM_JOB_ID:-manual_$$}"
EXPECTED_EXAMPLES="${EXPECTED_EXAMPLES:-500}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-pritamdeka/PubMedBERT-MNLI-MedNLI}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SOURCE_GIT_REVISION="${SOURCE_GIT_REVISION:-unknown}"

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
python -c 'import numpy, scipy, sklearn, torch; assert torch.cuda.is_available()'

for required_path in \
  "${PUBMEDQA_V2_RUN_DIR}/pubmedqa_transfer_run_metadata.json" \
  "${PUBMEDQA_V2_RUN_DIR}/examples.jsonl" \
  "${PUBMEDQA_V2_RUN_DIR}/prompts.jsonl" \
  "${PUBMEDQA_V2_ANALYSIS_DIR}/transfer_predictions.csv"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required accepted v2 artifact is missing: ${required_path}" >&2
    exit 2
  fi
done
for new_dir in "${OUTPUT_DIR}" "${ANALYSIS_DIR}"; do
  if [[ -e "${new_dir}" && -n "$(find "${new_dir}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to use non-empty output directory: ${new_dir}" >&2
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
  printf '%s\n' \
    "slurm_job_id: ${SLURM_JOB_ID:-manual}" \
    "started_at_utc: ${STARTED_AT_UTC}" \
    "ended_at_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "elapsed_seconds: $(( $(date +%s) - STARTED_EPOCH ))" \
    "exit_code: ${exit_code}" \
    "source_git_revision: ${SOURCE_GIT_REVISION}" \
    "runner_sha256: $(sha256sum "${PROJECT_DIR}/scripts/run_pubmedqa_v2_semantic_entropy.py" | awk '{print $1}')" \
    "batch_sha256: $(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" \
    "expected_examples: ${EXPECTED_EXAMPLES}" \
    "max_examples: ${MAX_EXAMPLES}" \
    "model_name: ${MODEL_NAME}" \
    "nli_model_name: ${NLI_MODEL_NAME}" \
    "num_samples: 10" \
    "temperature: 1.0" \
    "top_p: 0.9" \
    "top_k: 50" \
    "seed: 31" \
    "prompt_version: pubmedqa_context_explanation_v2" \
    "environment_selection: ${VENV_SELECTION}" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    > "${OUTPUT_DIR}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
ARGS=(
  --source-run-dir "${PUBMEDQA_V2_RUN_DIR}"
  --source-analysis-dir "${PUBMEDQA_V2_ANALYSIS_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --analysis-dir "${ANALYSIS_DIR}"
  --expected-examples "${EXPECTED_EXAMPLES}"
  --model-name "${MODEL_NAME}"
  --nli-model-name "${NLI_MODEL_NAME}"
  --device cuda
  --torch-dtype bfloat16
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
elif [[ "${LOCAL_FILES_ONLY}" != "0" ]]; then
  echo "LOCAL_FILES_ONLY must be 0 or 1." >&2
  exit 2
fi

python scripts/run_pubmedqa_v2_semantic_entropy.py "${ARGS[@]}"

python - "${OUTPUT_DIR}" "${ANALYSIS_DIR}" "${MAX_EXAMPLES:-${EXPECTED_EXAMPLES}}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
analysis = Path(sys.argv[2])
expected = int(sys.argv[3])
for name, count in (
    ("examples.jsonl", expected),
    ("prompts.jsonl", expected),
    ("generations.jsonl", expected * 10),
    ("cleaned_generations.jsonl", expected * 10),
    ("clusters.jsonl", expected),
):
    observed = sum(1 for line in (output / name).open() if line.strip())
    if observed != count:
        raise SystemExit(f"{name}: expected {count}, got {observed}")
with (output / "se_scores.csv").open(newline="") as infile:
    if len(list(csv.DictReader(infile))) != expected:
        raise SystemExit("se_scores.csv row mismatch")
summary = json.loads((analysis / "summary.json").read_text())
if summary.get("status") != "complete" or summary.get("rows", {}).get("analysis_predictions") != expected:
    raise SystemExit(f"Incomplete analysis summary: {summary}")
print(f"health_check=PASS examples={expected} generations={expected * 10}")
PY
