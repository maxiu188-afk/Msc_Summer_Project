# Isambard BioASQ medical-UQ run guide

The active experiment is the Phase-2 BioASQ hidden-state collection. Phase 1
is complete: it tested Semantic Entropy and blind P(True) on a shared
stratified question set. P(True)-10 is retired and is not part of any active
command.

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

## Phase 2: single-answer P(True), token UQ, and hidden states

The Phase-2 job uses the frozen 3,930-question manifest, not the historical
1,000-question Phase-1 cohort. It generates exactly one `T=0.1` no-evidence
main answer per question and records:

- original blind P(True) and verbalized-confidence UQ for that main answer;
- single-answer token UQ (`sequence_nll`, `normalized_nll`, mean/max token
  entropy); and
- bf16 hidden-state vectors from blocks `24/32/40/48` at `TBG/SLT/LT`.

It deliberately does **not** invoke NLI, Semantic Entropy, cluster count,
sample disagreement, P(True)-10, or any high-temperature answer generation.
The raw hidden tensors are split into train/validation/test `.pt` files and
have layout `[example, 4 blocks, 3 token positions, 3840 dimensions]`.

### Completed Phase-2 run and local Probe follow-up

Smoke job `5719092` completed in 27 seconds; full job `5719110` completed in
3:44:58 with a passing health check for all 3,930 answers, blind P(True) rows,
and `3,144/393/393` hidden-state rows. The copied local result is then consumed
by the CPU-only linear-Probe entry point; it never reloads Gemma or regenerates
answers:

```bash
PYTHONPATH=src python scripts/train_phase2_linear_probes.py \
  --run-dir outputs/bioasq_phase2_gemma3_12b_single_answer_seed31 \
  --output-dir analysis_outputs/bioasq_phase2_linear_probes_seed31 \
  --include-accuracy-probe --allow-incomplete-accuracy-labels
```

The command fits only train rows, uses validation to freeze the Probe
type/layer/token, and writes test metrics only for that selection. The first
pass fixes both final Probes at block 24/LT with L2 logistic regression:
P(True)-Probe uses the even train threshold and Accuracy-Probe uses Claude
incorrect labels. It is a local analysis step, not a new Isambard allocation.
See the root `PHASE2_PROBE_PLAN.md` for results and the cross-dataset next step.

The ignored manifest must be copied to the server before submission. From the
local machine, after the project source itself has been synchronized:

```bash
scp archehr_sebaseline/artifacts/phase2_bioasq_training13b_split/phase2_bioasq_split_manifest.jsonl \
  b6u.aip2.isambard:~/phase2_bioasq_split_manifest.jsonl
ssh b6u.aip2.isambard \
  'mkdir -p "$SCRATCHDIR/final_project/phase2_inputs" && mv ~/phase2_bioasq_split_manifest.jsonl "$SCRATCHDIR/final_project/phase2_inputs/"'
```

First submit the three-question smoke (one question from each frozen split).
It validates the Gemma hidden-state API and final artifact shape before the
full collection:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
SPLIT_MANIFEST="$SCRATCHDIR/final_project/phase2_inputs/phase2_bioasq_split_manifest.jsonl" \
LOCAL_FILES_ONLY=1 MAX_EXAMPLES_PER_SPLIT=1 \
sbatch --time=01:00:00 --job-name=bioasq-p2-smoke scripts/run_phase2_bioasq_isambard.sbatch
```

After the smoke's `health_check.txt` reports `PASS`, submit the full
collection. It requests 24 hours because it includes the post-hoc P(True) and
one-pass hidden-state replay for all 3,930 questions:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
SPLIT_MANIFEST="$SCRATCHDIR/final_project/phase2_inputs/phase2_bioasq_split_manifest.jsonl" \
OUTPUT_DIR="$PWD/outputs/bioasq_phase2_gemma3_12b_single_answer_seed31" \
LOCAL_FILES_ONLY=1 SEED=31 \
sbatch scripts/run_phase2_bioasq_isambard.sbatch
```

Check the completed run without inferring any accuracy result before Claude
labels exist:

```bash
OUT="$SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_phase2_gemma3_12b_single_answer_seed31"
cat "$OUT/health_check.txt"
cat "$OUT/run_timing.txt"
head -n 2 "$OUT/uq_baselines/self_report_examples.csv"
```

For a completed run, append reproducibility provenance without re-running the
model. Run this in the same cached model environment, then download the small
JSON sidecar with the other result metadata:

```bash
python scripts/write_phase2_provenance_sidecar.py \
  --run_dir "$OUT" \
  --model_name google/gemma-3-12b-it \
  --hf_cache_dir "$SCRATCHDIR/final_project/hf_cache" \
  --source_git_commit 4f26b9d
```

## Historical Phase-1 server smoke

The smoke uses three questions, three high-temperature samples, PubMedBERT
NLI, a low-temperature target answer, and blind P(True). It does not call
Claude because the binary judge is a separate local API stage.

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

The self-report CSV must contain `p_true_blind_uncertainty`; it must not
contain a P(True)-10 / `p_true_with_samples_uncertainty` column.

## Historical Phase-1 closing / P(True)-Probe baseline

After a successful smoke, use the same command with the full workload:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
OUTPUT_DIR="$PWD/outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31" \
MAX_EXAMPLES=1000 BIOASQ_TYPE_LIMITS="factoid=480,list=320,summary=200" \
SELECTION_SEED=20260718 NUM_SAMPLES=10 SEED=31 TEMPERATURE=1.0 TOP_P=0.9 \
LOCAL_FILES_ONLY=1 NLI_LOCAL_FILES_ONLY=1 \
sbatch scripts/run_bioasq_isambard.sbatch
```

The batch script writes all SE variants, token UQ, verbal confidence, blind
P(True), a health check, and timing. It intentionally does not run the archived
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

The resulting `claude_binary_main_answer_judge/claude_uq_*.csv` compares all
SE variants, token UQ, verbal confidence, and blind P(True) against the same
binary labels.

## Completed Phase-1 reference run (2026-07-19)

Jobs 5706186/5706187 completed the 1,000x10 no-evidence protocol at seeds
31/47 in 8:00:08/7:14:24. Both full artifacts passed the repaired health check;
jobs 5715701/5715703 added blind P(True) and verbal confidence. Claude retained
991 labels per seed after one bounded retry. P(True)-blind is strongest overall
(0.811/0.821), while list cluster count is 0.849/0.884. See
`docs/bioasq_medical_uq_results_20260718.md`.

## Superseded pilot (2026-07-18)

Jobs 5702970 (seed 31) and 5702980 (seed 47) completed the full 100x10
no-evidence protocol in 49:14 and 53:25. Both health checks passed. Local
Claude binary batches supplied 98 valid labels per seed; see
`docs/bioasq_medical_uq_protocol.md` for the type-stratified AUROC results.
The result established the blind P(True) protocol and retired P(True)-10. Its
local raw download is archived; do not extend it as the main experiment.

## Archived historical protocol

The earlier summary/evidence runs, generic DeBERTa NLI, three-axis deterministic
quality score, and three-level Claude labels are retained only as archived
diagnostics under `../archive_unused/docs/historical_results/`. They are not comparable to
the active protocol and must not be extended as the main experiment.
