#!/usr/bin/env bash
# Prepare a fresh Runpod PyTorch pod for the BioASQ Semantic Entropy baseline.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${VENV_PATH:-${PROJECT_DIR}/.venv_runpod}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${PROJECT_DIR}/.cache/torchinductor}"

mkdir -p "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
export HF_HOME TORCHINDUCTOR_CACHE_DIR
export TRANSFORMERS_CACHE="${HF_HOME}"
export PYTHONUNBUFFERED=1

# --system-site-packages preserves the CUDA-enabled PyTorch supplied by the
# Runpod image. Use a Runpod PyTorch/CUDA template rather than a plain image.
if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
  python3 -m venv --system-site-packages "${VENV_PATH}"
fi
source "${VENV_PATH}/bin/activate"

python -m pip install --upgrade pip
python -m pip install --upgrade -r "${PROJECT_DIR}/requirements-runpod.txt"
hf --help >/dev/null

python - <<'PY'
import torch
import transformers

print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Select a Runpod PyTorch/CUDA image and attach the L40S before continuing.")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"gpu_memory_gb={torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}")
PY

python -m unittest discover "${PROJECT_DIR}/tests"

echo "Runpod environment is ready. Activate later with: source ${VENV_PATH}/bin/activate"
