#!/usr/bin/env bash
#SBATCH --job-name=gemma3-4b-scale
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Gemma 3 4B model-scale comparison aligned exactly to the Phase-1 cohort.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"

WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
DATA_PATH="${DATA_PATH:-${WORKDIR}/data/BioASQ-training13b/training13b.json}"
REFERENCE_EXAMPLES_PATH="${REFERENCE_EXAMPLES_PATH:-${PROJECT_DIR}/outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31/examples.jsonl}"
REFERENCE_SHA256="4c40280a6ab234d204ad1f60e9c7d06a588f8387ee5ee21a4d2c1028e907b7d0"
SCALE_STAGE="${SCALE_STAGE:-smoke}"

case "${SCALE_STAGE}" in
  smoke)
    MAX_EXAMPLES=6
    BIOASQ_TYPE_LIMITS="factoid=2,list=2,summary=2"
    ALIGNMENT_MODE="subset"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_gemma3_4b_phase1_aligned_smoke_seed31_job${SLURM_JOB_ID}}"
    ;;
  full)
    MAX_EXAMPLES=1000
    BIOASQ_TYPE_LIMITS="factoid=480,list=320,summary=200"
    ALIGNMENT_MODE="exact"
    OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_gemma3_4b_1000x10_phase1_aligned_seed31_20260726}"
    ;;
  *)
    echo "SCALE_STAGE must be smoke or full, got: ${SCALE_STAGE}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "BioASQ source is missing: ${DATA_PATH}" >&2
  exit 2
fi
if [[ ! -f "${REFERENCE_EXAMPLES_PATH}" ]]; then
  echo "Accepted Phase-1 reference examples are missing: ${REFERENCE_EXAMPLES_PATH}" >&2
  exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite an existing 4B model-scale output: ${OUTPUT_DIR}" >&2
  exit 2
fi

TMP_BASE="${WORKDIR}/tmp"
JOB_TMPDIR="${TMP_BASE}/gemma3_4b_scale_${SLURM_JOB_ID}"
mkdir -p "${JOB_TMPDIR}"
export TMPDIR="${JOB_TMPDIR}"
export TMP="${JOB_TMPDIR}"
export TEMP="${JOB_TMPDIR}"

module load cray-python || true

TYPE_LIMIT_ARGS=()
IFS=',' read -r -a TYPE_LIMIT_ENTRIES <<< "${BIOASQ_TYPE_LIMITS}"
for type_limit in "${TYPE_LIMIT_ENTRIES[@]}"; do
  TYPE_LIMIT_ARGS+=(--type-limit "${type_limit}")
done

cd "${PROJECT_DIR}"
python scripts/check_phase1_cohort_alignment.py \
  --reference-examples "${REFERENCE_EXAMPLES_PATH}" \
  --source-data-path "${DATA_PATH}" \
  --mode "${ALIGNMENT_MODE}" \
  "${TYPE_LIMIT_ARGS[@]}" \
  --selection-seed 20260718 \
  --split train13b \
  --expected-reference-sha256 "${REFERENCE_SHA256}"

export PROJECT_DIR DATA_PATH OUTPUT_DIR
export DATASET="bioasq_medical_uq"
export SPLIT="train13b"
export MODEL_NAME="google/gemma-3-4b-it"
export MAX_EXAMPLES BIOASQ_TYPE_LIMITS
export SELECTION_SEED="20260718"
export NUM_SAMPLES="10"
export MAX_NEW_TOKENS="192"
export MAX_INPUT_TOKENS="4096"
export TEMPERATURE="1.0"
export TOP_P="0.9"
export TOP_K="50"
export BEST_GENERATION_TEMPERATURE="0.1"
export TORCH_DTYPE="bfloat16"
export NLI_MODEL_NAME="pritamdeka/PubMedBERT-MNLI-MedNLI"
export RUN_SELF_REPORT_UQ="1"
export INCLUDE_EVIDENCE="0"
export SEED="31"

bash "${PROJECT_DIR}/scripts/run_bioasq_isambard.sbatch"

python scripts/check_phase1_cohort_alignment.py \
  --reference-examples "${REFERENCE_EXAMPLES_PATH}" \
  --candidate-examples "${OUTPUT_DIR}/examples.jsonl" \
  --mode "${ALIGNMENT_MODE}" \
  "${TYPE_LIMIT_ARGS[@]}" \
  --selection-seed 20260718 \
  --split train13b \
  --expected-reference-sha256 "${REFERENCE_SHA256}" \
  --write-report "${OUTPUT_DIR}/phase1_cohort_alignment.json"

printf '%s\n' \
  "status: complete" \
  "scale_stage: ${SCALE_STAGE}" \
  "model_name: ${MODEL_NAME}" \
  "phase1_reference_sha256: ${REFERENCE_SHA256}" \
  "phase1_alignment_mode: ${ALIGNMENT_MODE}" \
  "phase1_alignment_report: ${OUTPUT_DIR}/phase1_cohort_alignment.json" \
  > "${OUTPUT_DIR}/model_scale_status.txt"
