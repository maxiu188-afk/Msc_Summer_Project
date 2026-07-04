# ArchEHR-QA UQ Handoff: Level 4 Baseline

Last updated: 2026-07-04

## Current State

The active implementation is:

```text
D:\work\FinalProject\code\archehr_sebaseline
```

The codebase has been cleaned so the maintained path now starts from Level 3/4.
Old Level 0-2 code, scripts, tests, fake data, old packages, and obsolete
handoff docs have been removed.

Current maintained entry points:

```text
archehr_sebaseline/scripts/run_level3.py
archehr_sebaseline/scripts/run_level3.sbatch
archehr_sebaseline/scripts/run_level4.py
archehr_sebaseline/scripts/run_level4.sbatch
archehr_sebaseline/scripts/check_level4_outputs.py
archehr_sebaseline/scripts/evaluate_pubmedqa_labels.py
```

Dependencies are now listed in:

```text
archehr_sebaseline/requirements.txt
```

Local validation after cleanup:

```text
python -m unittest discover archehr_sebaseline\tests
Ran 23 tests
OK
```

## Level 4 Server Baseline

The cleaned code has been validated on Isambard with a full Level 4 pilot:

```text
dataset: pubmedqa
examples: 50
generations: 250
model_name: Qwen/Qwen2.5-7B-Instruct
num_samples: 5
device: cuda
torch_dtype: float16
requested_clustering_method: nli
clustering_method: nli_bidirectional_entailment
nli_model_name: microsoft/deberta-v2-xlarge-mnli
token_scores: True
```

The run completed in about 14 minutes and passed the structured health check.

## Local Result Location

Downloaded server results are stored at:

```text
D:\work\FinalProject\code\server_results\level4_qwen25_7b_nli_50x5
```

Important files:

```text
summary.txt
health_check.txt
analysis_report.md
label_heuristic_eval.md
pubmedqa_label_predictions.csv
pubmedqa_eval_summary.json
rejection_curve.csv
auroc_bar.svg
rejection_curve.svg
examples.jsonl
generations.jsonl
cleaned_generations.jsonl
clusters.jsonl
se_scores.csv
generation_uq.csv
example_uq.csv
```

## Initial Findings

The 50x5 Qwen2.5-7B PubMedQA pilot is meaningful enough for baseline analysis.

Main observations:

- Output artifacts are structurally sound: complete row counts, valid JSONL/CSV,
  no `NaN` or `Infinity`, and token-score fields populated.
- NLI semantic clustering found variation in most examples:
  - 12/50 examples had one semantic cluster.
  - 38/50 examples split into 2-4 semantic clusters.
- Normalized discrete SE mean is about `0.396`.
- Likelihood-weighted SE almost exactly tracks discrete SE in this run.
- Token-level uncertainty correlates only weakly with semantic entropy, which
  supports treating SE as an answer-level uncertainty signal.
- A simple PubMedQA yes/no/maybe majority heuristic gives `33/50 = 0.66`
  accuracy.
- The maintained Level 5 evaluator is more conservative: ties are marked
  `unknown`, giving `28/50 = 0.560` majority accuracy on the same pilot.
- Level 5 AUROCs for detecting incorrect majority answers are currently modest:
  normalized discrete SE about `0.547`, normalized weighted SE about `0.551`,
  predictive entropy / normalized NLL about `0.570`, and mean token entropy
  about `0.557`.
- Important caveat: SE captures semantic disagreement, but not all mistakes.
  Some answers are stable and wrong.

## Server Commands

Use the cleaned project directory on the server:

```bash
cd $SCRATCHDIR/final_project/archehr_sebaseline
source .venv_level2/bin/activate
```

If rebuilding the environment:

```bash
module load cray-python || true
python -m venv .venv_level2
source .venv_level2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the NVIDIA driver complains that it is too old, reinstall a compatible
PyTorch wheel, for example:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
python -m pip install transformers sentencepiece protobuf tiktoken
```

Run a cleaned-code Level 4 smoke:

```bash
DATASET=pubmedqa \
DATA_PATH=$SCRATCHDIR/final_project/data/pubmedqa/ori_pqal.json \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_clean_qwen25_nli_1x1 \
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
MAX_EXAMPLES=1 \
NUM_SAMPLES=1 \
MAX_NEW_TOKENS=32 \
DEVICE=cuda \
TORCH_DTYPE=float16 \
LOCAL_FILES_ONLY=1 \
CLUSTERING_METHOD=nli \
NLI_MODEL_NAME=microsoft/deberta-v2-xlarge-mnli \
NLI_LOCAL_FILES_ONLY=1 \
sbatch --time=00:20:00 scripts/run_level4.sbatch
```

Run the 50x5 pilot:

```bash
DATASET=pubmedqa \
DATA_PATH=$SCRATCHDIR/final_project/data/pubmedqa/ori_pqal.json \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_clean_qwen25_nli_50x5 \
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
MAX_EXAMPLES=50 \
NUM_SAMPLES=5 \
MAX_NEW_TOKENS=128 \
DEVICE=cuda \
TORCH_DTYPE=float16 \
LOCAL_FILES_ONLY=1 \
CLUSTERING_METHOD=nli \
NLI_MODEL_NAME=microsoft/deberta-v2-xlarge-mnli \
NLI_LOCAL_FILES_ONLY=1 \
sbatch --time=01:00:00 scripts/run_level4.sbatch
```

Health check:

```bash
OUT=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_clean_qwen25_nli_50x5

python scripts/check_level4_outputs.py \
  $OUT \
  --expected_examples 50 \
  --expected_num_samples 5 \
  --expected_generations 250 \
  --require_cuda \
  --require_nli \
  --write_report $OUT/health_check.txt
```

If health check reports missing files, first verify that `OUT` points to the
actual output directory:

```bash
find $SCRATCHDIR/final_project/archehr_sebaseline/outputs -name summary.txt -printf "%TY-%Tm-%Td %TH:%TM %p\n" | sort | tail -n 10
```

## Cleanup Notes

Server old backup directories can be removed after the cleaned-code health
check passes:

```bash
cd $SCRATCHDIR/final_project
rm -rf archehr_sebaseline_before_cleanup_*
```

Do not delete the active cleaned project directory:

```text
$SCRATCHDIR/final_project/archehr_sebaseline
```

Do not delete model caches unless intentionally freeing space:

```text
$SCRATCHDIR/final_project/hf_cache
$SCRATCHDIR/final_project/torch_cache
```

## Level 5 Evaluation

The PubMedQA yes/no/maybe heuristic has been promoted into a maintained module
and CLI script:

```text
archehr_sebaseline/src/archehr_sebaseline/evaluation/pubmedqa_labels.py
archehr_sebaseline/scripts/evaluate_pubmedqa_labels.py
archehr_sebaseline/tests/test_pubmedqa_label_eval.py
```

Run it on a Level 4 output directory:

```bash
python scripts/evaluate_pubmedqa_labels.py \
  $OUT \
  --overwrite
```

It writes:

```text
pubmedqa_label_predictions.csv
pubmedqa_eval_summary.json
rejection_curve.csv
auroc_bar.svg
rejection_curve.svg
```

## Next Recommended Work

Review whether the maintained conservative label heuristic is acceptable for
the report, then either run a larger `N=10` baseline for smoother SE estimates
or start SEP target generation and hidden-state extraction.
