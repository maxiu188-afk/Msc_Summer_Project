#!/usr/bin/env bash
# Archive every required Runpod BioASQ result plus durable batch logs.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs}"
ARCHIVE_PATH="${1:-${OUTPUT_ROOT}/bioasq_se_runpod_required_results_$(date -u +%Y%m%dT%H%M%SZ).tar.gz}"

required_runs=(
  "runpod_bioasq_golden_summary_gemma3_12b_100x10"
  "runpod_bioasq_golden_factoid_gemma3_12b_50x10"
  "runpod_bioasq_golden_list_gemma3_12b_50x10"
  "runpod_bioasq_train_summary_gemma3_12b_50x10_seed47"
)
archive_items=()

for run_name in "${required_runs[@]}"; do
  run_dir="${OUTPUT_ROOT}/${run_name}"
  if [[ ! -d "${run_dir}" ]]; then
    echo "[package] missing required run directory: ${run_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${run_dir}/health_check.txt" ]] || ! grep -q "Level 4 output health check: PASS" "${run_dir}/health_check.txt"; then
    echo "[package] required run does not have a passing health check: ${run_dir}" >&2
    exit 1
  fi
  if [[ ! -f "${run_dir}/bioasq_eval/bioasq_eval_summary.json" ]]; then
    echo "[package] missing BioASQ evaluation summary: ${run_dir}" >&2
    exit 1
  fi
  archive_items+=("outputs/${run_name}")
done

# The Qwen comparison is optional, but preserve it whenever it was run.
optional_qwen="${OUTPUT_ROOT}/runpod_bioasq_golden_summary_qwen25_7b_50x10"
if [[ -d "${optional_qwen}" ]]; then
  archive_items+=("outputs/$(basename "${optional_qwen}")")
fi

if [[ -d "${OUTPUT_ROOT}/batch_logs" ]]; then
  archive_items+=("outputs/batch_logs")
fi

mkdir -p "$(dirname "${ARCHIVE_PATH}")"
cd "${PROJECT_DIR}"
tar -czf "${ARCHIVE_PATH}" "${archive_items[@]}"
tar -tzf "${ARCHIVE_PATH}" > /dev/null

echo "[package] archive verified: ${ARCHIVE_PATH}"
echo "[package] included: ${archive_items[*]}"
