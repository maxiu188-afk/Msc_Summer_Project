#!/usr/bin/env bash
#SBATCH --job-name=bioasq-uq-smoke
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00

# Exercise generation, PubMedBERT NLI set clustering, and blind P(True).
# Claude is deliberately offline and is not called by a server smoke test.
set -euo pipefail

: "${SCRATCHDIR:?SCRATCHDIR is not set. Submit this on Isambard.}"
WORKDIR="${SCRATCHDIR}/final_project"
PROJECT_DIR="${PROJECT_DIR:-${WORKDIR}/archehr_sebaseline}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to the BioASQ JSON file or Golden directory.}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

export PROJECT_DIR DATA_PATH
export DATASET="${DATASET:-bioasq_medical_uq}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs/bioasq_medical_uq_smoke_${STAMP}}"
export MAX_EXAMPLES="${MAX_EXAMPLES:-3}"
export BIOASQ_TYPE_LIMITS="${BIOASQ_TYPE_LIMITS:-}"
export NUM_SAMPLES="${NUM_SAMPLES:-3}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-48}"
export MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-2048}"
export RUN_SELF_REPORT_UQ=1
export INCLUDE_EVIDENCE=0

exec bash "${PROJECT_DIR}/scripts/run_bioasq_isambard.sbatch"
