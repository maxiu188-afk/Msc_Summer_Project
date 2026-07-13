#!/usr/bin/env bash
# Run the remaining BioASQ experiments in one unattended tmux session.
#
# Required jobs: Golden summary, Golden factoid, Golden list, and a summary
# seed-repeat. Each job runs the Level 4 health check through the pilot script,
# then the lightweight BioASQ evaluator. A failure stops the batch immediately.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VENV_PATH:-${PROJECT_DIR}/.venv_runpod}/bin/activate"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/batch_logs}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
RUN_OPTIONAL_QWEN="${RUN_OPTIONAL_QWEN:-0}"

mkdir -p "${LOG_DIR}"

is_completed() {
  local output_dir="$1"
  [[ -f "${output_dir}/health_check.txt" ]] &&
    grep -q "Level 4 output health check: PASS" "${output_dir}/health_check.txt" &&
    [[ -f "${output_dir}/bioasq_eval/bioasq_eval_summary.json" ]]
}

run_job() {
  local name="$1"
  local dataset="$2"
  local data_path="$3"
  local split="$4"
  local max_examples="$5"
  local max_new_tokens="$6"
  local seed="$7"
  local output_dir="$8"
  local model_name="${9:-google/gemma-3-12b-it}"
  local log_path="${LOG_DIR}/${name}_$(date -u +%Y%m%dT%H%M%SZ).log"

  if [[ "${SKIP_COMPLETED}" == "1" ]] && is_completed "${output_dir}"; then
    echo "[batch] SKIP completed ${name}: ${output_dir}"
    return
  fi

  echo "[batch] START ${name}"
  echo "[batch] log: ${log_path}"
  (
    DATASET="${dataset}" \
    DATA_PATH="${data_path}" \
    SPLIT="${split}" \
    MAX_EXAMPLES="${max_examples}" \
    NUM_SAMPLES=10 \
    MAX_NEW_TOKENS="${max_new_tokens}" \
    SEED="${seed}" \
    OUTPUT_DIR="${output_dir}" \
    MODEL_NAME="${model_name}" \
    bash "${PROJECT_DIR}/scripts/runpod_bioasq_summary_pilot.sh"

    python "${PROJECT_DIR}/scripts/evaluate_bioasq_quality.py" \
      --run_dir "${output_dir}" \
      --quality_target mean \
      --quality_threshold 0.15 \
      --overwrite
  ) 2>&1 | tee "${log_path}"
  echo "[batch] COMPLETE ${name}"
}

GOLDEN_DIR="${PROJECT_DIR}/data/Task13BGoldenEnriched"
TRAINING_PATH="${PROJECT_DIR}/data/BioASQ-training13b/training13b.json"

run_job \
  golden_summary_gemma3_12b_100x10 \
  bioasq_summary "${GOLDEN_DIR}" golden13b 100 192 31 \
  "${OUTPUT_ROOT}/runpod_bioasq_golden_summary_gemma3_12b_100x10"

run_job \
  golden_factoid_gemma3_12b_50x10 \
  bioasq_factoid "${GOLDEN_DIR}" golden13b 50 96 31 \
  "${OUTPUT_ROOT}/runpod_bioasq_golden_factoid_gemma3_12b_50x10"

run_job \
  golden_list_gemma3_12b_50x10 \
  bioasq_list "${GOLDEN_DIR}" golden13b 50 128 31 \
  "${OUTPUT_ROOT}/runpod_bioasq_golden_list_gemma3_12b_50x10"

run_job \
  train_summary_gemma3_12b_50x10_seed47 \
  bioasq_summary "${TRAINING_PATH}" train13b 50 192 47 \
  "${OUTPUT_ROOT}/runpod_bioasq_train_summary_gemma3_12b_50x10_seed47"

if [[ "${RUN_OPTIONAL_QWEN}" == "1" ]]; then
  run_job \
    golden_summary_qwen25_7b_50x10 \
    bioasq_summary "${GOLDEN_DIR}" golden13b 50 192 31 \
    "${OUTPUT_ROOT}/runpod_bioasq_golden_summary_qwen25_7b_50x10" \
    Qwen/Qwen2.5-7B-Instruct
fi

echo "[batch] All requested BioASQ jobs completed."
