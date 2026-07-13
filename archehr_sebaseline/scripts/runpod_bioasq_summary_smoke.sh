#!/usr/bin/env bash
# Fast model-backed check: data loading, CUDA generation, token scores, and outputs.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VENV_PATH:-${PROJECT_DIR}/.venv_runpod}/bin/activate"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PROJECT_DIR}/.cache/torchinductor}"
export PYTHONUNBUFFERED=1

DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/BioASQ-training13b/training13b.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/runpod_bioasq_summary_smoke}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"

python "${PROJECT_DIR}/scripts/run_level4.py" \
  --dataset bioasq_summary \
  --data_path "${DATA_PATH}" \
  --split train13b \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --num_samples 2 \
  --max_examples 2 \
  --max_new_tokens 96 \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed 31 \
  --device cuda \
  --torch_dtype float16 \
  --max_input_tokens 4096 \
  --clustering_method exact \
  --overwrite

python "${PROJECT_DIR}/scripts/check_level4_outputs.py" \
  "${OUTPUT_DIR}" \
  --expected_examples 2 \
  --expected_num_samples 2 \
  --expected_generations 4 \
  --require_cuda \
  --write_report "${OUTPUT_DIR}/health_check.txt"
