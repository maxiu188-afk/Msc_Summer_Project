#!/usr/bin/env bash
# Configurable BioASQ Semantic Entropy NLI run for a 48 GB GPU.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VENV_PATH:-${PROJECT_DIR}/.venv_runpod}/bin/activate"

export HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PROJECT_DIR}/.cache/torchinductor}"
export PYTHONUNBUFFERED=1

# Gemma access must have been accepted on Hugging Face. Before the first run,
# authenticate in the pod: huggingface-cli login (or export HF_TOKEN=...).
DATASET="${DATASET:-bioasq_summary}"
DATA_PATH="${DATA_PATH:-${PROJECT_DIR}/data/BioASQ-training13b/training13b.json}"
SPLIT="${SPLIT:-train13b}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/runpod_${DATASET}_gemma3_12b_100x10}"
MODEL_NAME="${MODEL_NAME:-google/gemma-3-12b-it}"
NLI_MODEL_NAME="${NLI_MODEL_NAME:-microsoft/deberta-v2-xlarge-mnli}"
MAX_EXAMPLES="${MAX_EXAMPLES:-100}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-192}"
SEED="${SEED:-31}"

python "${PROJECT_DIR}/scripts/run_level4.py" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --split "${SPLIT}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name "${MODEL_NAME}" \
  --num_samples "${NUM_SAMPLES}" \
  --max_examples "${MAX_EXAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.8 \
  --top_p 0.9 \
  --seed "${SEED}" \
  --device cuda \
  --torch_dtype bfloat16 \
  --max_input_tokens 4096 \
  --clustering_method nli \
  --nli_model_name "${NLI_MODEL_NAME}" \
  --nli_device cuda \
  --nli_max_input_tokens 512 \
  --show_progress \
  --overwrite

# MAX_EXAMPLES is an upper bound.  Some official BioASQ subsets contain fewer
# compatible questions (for example, Golden 13B summary has 80), so validate
# the artifacts against the number actually written rather than that cap.
ACTUAL_EXAMPLES="$(awk 'END { print NR }' "${OUTPUT_DIR}/examples.jsonl")"
if [[ "${ACTUAL_EXAMPLES}" -lt 1 ]]; then
  echo "[health] no examples were written to ${OUTPUT_DIR}/examples.jsonl" >&2
  exit 1
fi
EXPECTED_GENERATIONS="$((ACTUAL_EXAMPLES * NUM_SAMPLES))"

python "${PROJECT_DIR}/scripts/check_level4_outputs.py" \
  "${OUTPUT_DIR}" \
  --expected_examples "${ACTUAL_EXAMPLES}" \
  --expected_num_samples "${NUM_SAMPLES}" \
  --expected_generations "${EXPECTED_GENERATIONS}" \
  --require_cuda \
  --require_nli \
  --write_report "${OUTPUT_DIR}/health_check.txt"
