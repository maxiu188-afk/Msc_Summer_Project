# Isambard BioASQ medical-UQ run guide

The active experiment is BioASQ medical QA uncertainty estimation. It tests
two hypotheses in one matched pipeline: whether Semantic Entropy is useful for
medical answers, and whether P(True) improves when it sees high-temperature
alternatives.

The active data mix is `bioasq_medical_uq`: factoid, list, and summary only.
Yes/no questions are excluded. Generation is no-evidence for all three types;
summary stays no-evidence even in a future mixed evidence configuration.

## Project and data

```text
$SCRATCHDIR/final_project/
├── archehr_sebaseline/
└── data/BioASQ-training13b/training13b.json
```

The batch scripts select `.venv_isambard` on R580+ CUDA nodes and
`.venv_isambard_cuda127` otherwise. Do not replace either environment's
PyTorch build to diagnose an allocation issue.

## Server smoke first

The smoke uses three questions, three high-temperature samples, PubMedBERT
NLI, a low-temperature target answer, and both P(True) variants. It does not
call Claude because the binary judge is a separate local API stage.

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
LOCAL_FILES_ONLY=0 NLI_LOCAL_FILES_ONLY=0 \
sbatch scripts/run_bioasq_medical_uq_smoke_isambard.sbatch
```

After the first model downloads complete, set both cache flags to `1`. Confirm
the job succeeded before any full run:

```bash
OUT="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_medical_uq_smoke_<timestamp>"
cat "$OUT/health_check.txt"
cat "$OUT/run_timing.txt"
head -n 2 "$OUT/uq_baselines/self_report_examples.csv"
```

The self-report CSV must contain `p_true_blind_uncertainty` and
`p_true_with_samples_uncertainty`; the latter is the P(True)-10 condition.

## Full paired experiment

After a successful smoke, use the same command with the full workload:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
OUTPUT_DIR="$PWD/outputs/bioasq_medical_uq_gemma3_12b_100x10_seed31" \
MAX_EXAMPLES=100 NUM_SAMPLES=10 SEED=31 TEMPERATURE=1.0 TOP_P=0.9 \
LOCAL_FILES_ONLY=1 NLI_LOCAL_FILES_ONLY=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

The batch script writes generation/UQ artifacts, a health check, timing, and
paired P(True) results. It intentionally does not run the archived
deterministic BioASQ quality evaluator.

## Local binary Claude stage

Download a complete run, then submit one judge request for every
low-temperature `best_generations.jsonl` answer. The judge returns exactly
`correct` or `incorrect`; `incorrect` is the positive risk class. Factoid and
list use `exact_answers`; summary uses `ideal_answers`.

```bash
cd archehr_sebaseline
python scripts/run_bioasq_claude_judge.py submit --run_dir <run-dir>
python scripts/run_bioasq_claude_judge.py status --run_dir <run-dir>
python scripts/run_bioasq_claude_judge.py download --run_dir <run-dir>
python scripts/evaluate_bioasq_claude_judge.py --run_dir <run-dir>
```

The resulting `claude_binary_main_answer_judge/claude_uq_*.csv` compares SE,
token UQ, P(True)-blind, and P(True)-10 against the same binary labels.

## Archived historical protocol

The earlier summary/evidence runs, generic DeBERTa NLI, three-axis deterministic
quality score, and three-level Claude labels are retained only as archived
diagnostics under `docs/archived_low_usability/`. They are not comparable to
the active protocol and must not be extended as the main experiment.
