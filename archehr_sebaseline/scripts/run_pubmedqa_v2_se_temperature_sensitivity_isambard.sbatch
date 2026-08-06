#!/usr/bin/env bash
#SBATCH --job-name=pubmedqa-se-temp
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Paired temperature diagnostic: reuse the accepted T=1.0 result and add only
# T=0.7 and T=1.3 on one fixed incorrect x gold-label stratified cohort.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
: "${PUBMEDQA_V2_RUN_DIR:?Set PUBMEDQA_V2_RUN_DIR to the accepted v2 run.}"
: "${PUBMEDQA_V2_ANALYSIS_DIR:?Set PUBMEDQA_V2_ANALYSIS_DIR to its accepted analysis.}"
: "${BASELINE_SE_ANALYSIS_DIR:?Set BASELINE_SE_ANALYSIS_DIR to accepted T=1.0 analysis.}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new temperature-study output root.}"
: "${ANALYSIS_ROOT:?Set ANALYSIS_ROOT to a new temperature-study analysis root.}"

PROJECT_DIR="${PROJECT_DIR:-${SCRATCHDIR}/final_project/archehr_sebaseline}"
WORKDIR="${SCRATCHDIR}/final_project"
VENV_PATH_CUDA127="${PROJECT_DIR}/.venv_isambard_cuda127"
HF_CACHE_DIR="${WORKDIR}/hf_cache"
TORCH_CACHE_DIR="${WORKDIR}/torch_cache"
JOB_TMP_DIR="${SCRATCHDIR}/tmp/pubmedqa_se_temp_${SLURM_JOB_ID:-manual_$$}"
STUDY_SIZE="${STUDY_SIZE:-200}"
SELECTION_SEED="${SELECTION_SEED:-20260806}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-20260806}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-20000}"
EXPECTED_EXAMPLES="${EXPECTED_EXAMPLES:-500}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-pritamdeka/PubMedBERT-MNLI-MedNLI}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SOURCE_GIT_REVISION="${SOURCE_GIT_REVISION:-unknown}"
LOW_TEMPERATURE="0.7"
HIGH_TEMPERATURE="1.3"

for integer_value in "${STUDY_SIZE}" "${SELECTION_SEED}" "${BOOTSTRAP_SEED}" "${BOOTSTRAP_SAMPLES}"; do
  [[ "${integer_value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "Study sizes and seeds must be positive integers." >&2
    exit 2
  }
done
if (( STUDY_SIZE > EXPECTED_EXAMPLES )); then
  echo "STUDY_SIZE cannot exceed EXPECTED_EXAMPLES." >&2
  exit 2
fi

LOW_OUTPUT_DIR="${OUTPUT_ROOT}/temperature_0p7"
HIGH_OUTPUT_DIR="${OUTPUT_ROOT}/temperature_1p3"
LOW_ANALYSIS_DIR="${ANALYSIS_ROOT}/temperature_0p7"
HIGH_ANALYSIS_DIR="${ANALYSIS_ROOT}/temperature_1p3"
COMPARISON_DIR="${ANALYSIS_ROOT}/comparison"
MANIFEST_PATH="${ANALYSIS_ROOT}/cohort_manifest.csv"
MANIFEST_SUMMARY_PATH="${ANALYSIS_ROOT}/cohort_manifest_summary.json"

for new_root in "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}"; do
  if [[ -e "${new_root}" && -n "$(find "${new_root}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "Refusing to use non-empty study root: ${new_root}" >&2
    exit 2
  fi
done
for required_path in \
  "${PUBMEDQA_V2_RUN_DIR}/pubmedqa_transfer_run_metadata.json" \
  "${PUBMEDQA_V2_RUN_DIR}/examples.jsonl" \
  "${PUBMEDQA_V2_RUN_DIR}/prompts.jsonl" \
  "${PUBMEDQA_V2_ANALYSIS_DIR}/transfer_predictions.csv" \
  "${BASELINE_SE_ANALYSIS_DIR}/predictions.csv" \
  "${BASELINE_SE_ANALYSIS_DIR}/summary.json"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Required accepted artifact is missing: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}" "${JOB_TMP_DIR}" \
  "${HF_CACHE_DIR}" "${TORCH_CACHE_DIR}"
chmod 700 "${JOB_TMP_DIR}"
export TMPDIR="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${TORCH_CACHE_DIR}"
export PYTHONUNBUFFERED=1

module load cray-python || true
VENV_PATH="${VENV_PATH:-${VENV_PATH_CUDA127}}"
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  echo "Selected virtual environment is missing: ${VENV_PATH}" >&2
  exit 3
fi
source "${VENV_PATH}/bin/activate"
python -c 'import numpy, scipy, sklearn, torch; assert torch.cuda.is_available()'
GPU_DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | sed -n '1p')"

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
    "study_sha256: $(sha256sum "${PROJECT_DIR}/scripts/pubmedqa_v2_se_temperature_sensitivity.py" | awk '{print $1}')" \
    "batch_sha256: $(sha256sum "${BASH_SOURCE[0]}" | awk '{print $1}')" \
    "study_size: ${STUDY_SIZE}" \
    "selection_seed: ${SELECTION_SEED}" \
    "bootstrap_seed: ${BOOTSTRAP_SEED}" \
    "bootstrap_samples: ${BOOTSTRAP_SAMPLES}" \
    "temperatures: ${LOW_TEMPERATURE},1.0,${HIGH_TEMPERATURE}" \
    "new_temperature_arms: ${LOW_TEMPERATURE},${HIGH_TEMPERATURE}" \
    "num_samples_per_question: 10" \
    "generation_seed: 31" \
    "top_p: 0.9" \
    "top_k: 50" \
    "model_name: ${MODEL_NAME}" \
    "nli_model_name: ${NLI_MODEL_NAME}" \
    "prompt_version: pubmedqa_context_explanation_v2" \
    "venv_path: ${VENV_PATH}" \
    "gpu_driver_version: ${GPU_DRIVER_VERSION}" \
    > "${OUTPUT_ROOT}/run_timing.txt"
  exit "${exit_code}"
}
trap record_timing EXIT

cd "${PROJECT_DIR}"
python scripts/pubmedqa_v2_se_temperature_sensitivity.py prepare \
  --source-analysis-dir "${PUBMEDQA_V2_ANALYSIS_DIR}" \
  --output-manifest "${MANIFEST_PATH}" \
  --output-summary "${MANIFEST_SUMMARY_PATH}" \
  --sample-size "${STUDY_SIZE}" \
  --selection-seed "${SELECTION_SEED}"

COMMON_ARGS=(
  --source-run-dir "${PUBMEDQA_V2_RUN_DIR}"
  --source-analysis-dir "${PUBMEDQA_V2_ANALYSIS_DIR}"
  --expected-examples "${EXPECTED_EXAMPLES}"
  --selected-ids-path "${MANIFEST_PATH}"
  --model-name "${MODEL_NAME}"
  --nli-model-name "${NLI_MODEL_NAME}"
  --device cuda
  --torch-dtype bfloat16
)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  COMMON_ARGS+=(--local-files-only)
elif [[ "${LOCAL_FILES_ONLY}" != "0" ]]; then
  echo "LOCAL_FILES_ONLY must be 0 or 1." >&2
  exit 2
fi

python scripts/run_pubmedqa_v2_semantic_entropy.py \
  "${COMMON_ARGS[@]}" \
  --temperature "${LOW_TEMPERATURE}" \
  --output-dir "${LOW_OUTPUT_DIR}" \
  --analysis-dir "${LOW_ANALYSIS_DIR}"

python scripts/run_pubmedqa_v2_semantic_entropy.py \
  "${COMMON_ARGS[@]}" \
  --temperature "${HIGH_TEMPERATURE}" \
  --output-dir "${HIGH_OUTPUT_DIR}" \
  --analysis-dir "${HIGH_ANALYSIS_DIR}"

python scripts/pubmedqa_v2_se_temperature_sensitivity.py analyze \
  --manifest "${MANIFEST_PATH}" \
  --baseline-analysis-dir "${BASELINE_SE_ANALYSIS_DIR}" \
  --low-analysis-dir "${LOW_ANALYSIS_DIR}" \
  --high-analysis-dir "${HIGH_ANALYSIS_DIR}" \
  --output-dir "${COMPARISON_DIR}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES}" \
  --bootstrap-seed "${BOOTSTRAP_SEED}"

python - "${OUTPUT_ROOT}" "${ANALYSIS_ROOT}" "${STUDY_SIZE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
analysis_root = Path(sys.argv[2])
expected = int(sys.argv[3])
with (analysis_root / "cohort_manifest.csv").open(newline="") as infile:
    if len(list(csv.DictReader(infile))) != expected:
        raise SystemExit("Temperature manifest row mismatch.")
for slug in ("temperature_0p7", "temperature_1p3"):
    output = output_root / slug
    analysis = analysis_root / slug
    expected_counts = {
        "examples.jsonl": expected,
        "prompts.jsonl": expected,
        "generations.jsonl": expected * 10,
        "cleaned_generations.jsonl": expected * 10,
        "clusters.jsonl": expected,
    }
    for name, count in expected_counts.items():
        observed = sum(1 for line in (output / name).open() if line.strip())
        if observed != count:
            raise SystemExit(f"{slug}/{name}: expected {count}, got {observed}")
    summary = json.loads((analysis / "summary.json").read_text())
    if summary.get("status") != "complete" or summary.get("rows", {}).get("analysis_predictions") != expected:
        raise SystemExit(f"Incomplete temperature arm: {slug}")
comparison = json.loads((analysis_root / "comparison" / "summary.json").read_text())
if comparison.get("status") != "complete" or comparison.get("num_examples") != expected:
    raise SystemExit("Incomplete paired temperature comparison.")
print(f"temperature_health_check=PASS examples={expected} new_generations={expected * 20}")
PY
