# ArchEHR-QA Semantic Entropy Baseline

This project contains the current Level 3/4 Semantic Entropy baseline for grounded clinical QA experiments. Earlier local-only smoke levels have been removed from the active code path; the maintained workflow now starts from the common dataset schema and the Level 4 token-score/NLI pipeline.

## Current Status

- Level 3: common-schema stability pipeline.
- Level 4: Qwen2.5-7B PubMedQA pilot with token-score baselines, likelihood-weighted Semantic Entropy, and NLI bidirectional-entailment clustering.
- Latest server pilot: 50 examples x 5 samples, CUDA, Qwen2.5-7B-Instruct, NLI clustering, health check PASS.

## Install

```powershell
python -m venv archehr_sebaseline\.venv
archehr_sebaseline\.venv\Scripts\python.exe -m pip install -r archehr_sebaseline\requirements.txt
```

On Isambard:

```bash
cd $SCRATCHDIR/final_project/archehr_sebaseline
module load cray-python || true
python -m venv .venv_level2
source .venv_level2/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Tests

From `D:\work\FinalProject\code`:

```powershell
python -m unittest discover archehr_sebaseline\tests
```

## Level 3

Level 3 runs the common-schema pipeline with exact cleaned-answer clustering.

```powershell
python archehr_sebaseline\scripts\run_level3.py --dataset fake --output_dir archehr_sebaseline\outputs\level3 --num_samples 3 --max_examples 10 --overwrite
```

PubMedQA:

```powershell
python archehr_sebaseline\scripts\run_level3.py --dataset pubmedqa --data_path D:\path\to\ori_pqal.json --split pqal --output_dir archehr_sebaseline\outputs\level3_pubmedqa --num_samples 3 --max_examples 10 --overwrite
```

## Level 4

Level 4 records token log probabilities, token entropies, sequence NLL, per-example UQ summaries, discrete SE, likelihood-weighted SE, and NLI semantic clusters.

Local exact-clustering smoke:

```powershell
python archehr_sebaseline\scripts\run_level4.py --dataset pubmedqa --data_path D:\path\to\ori_pqal.json --split pqal --output_dir archehr_sebaseline\outputs\level4_pubmedqa --model_name sshleifer/tiny-gpt2 --num_samples 1 --max_examples 1 --max_new_tokens 16 --device cpu --clustering_method exact --overwrite
```

Server Qwen2.5-7B NLI pilot:

```bash
DATASET=pubmedqa \
DATA_PATH=$SCRATCHDIR/final_project/data/pubmedqa/ori_pqal.json \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/level4_qwen25_7b_nli_50x5 \
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

## Health Check

```bash
python scripts/check_level4_outputs.py \
  $OUT \
  --expected_examples 50 \
  --expected_num_samples 5 \
  --expected_generations 250 \
  --require_cuda \
  --require_nli \
  --write_report $OUT/health_check.txt
```

The check validates required artifacts, row counts, summary consistency, JSONL/CSV parsing, token-score fields, cluster-size consistency, entropy ranges, and non-finite text markers.

## Level 5 PubMedQA Evaluation

Level 5 evaluates Level 4 uncertainty scores against a PubMedQA yes/no/maybe
answer-quality proxy. The maintained heuristic extracts explicit generated
labels, uses conservative majority voting, and marks tied known votes as
`unknown`.

See [docs/level5_evaluation_method.md](docs/level5_evaluation_method.md) for
the full evaluation definition and limitations.

```powershell
python archehr_sebaseline\scripts\evaluate_pubmedqa_labels.py server_results\level4_qwen25_7b_nli_50x5 --overwrite
```

The script writes:

```text
pubmedqa_label_predictions.csv
pubmedqa_eval_summary.json
rejection_curve.csv
auroc_bar.svg
rejection_curve.svg
```

For the local 50x5 Qwen2.5-7B pilot, this maintained evaluator reports
`28/50 = 0.560` majority accuracy and AUROC values around `0.55-0.57` for the
current uncertainty scores. Open the SVG files in a browser for a quick visual
summary of the AUROC table and selective-prediction curve.

## Data Boundary

Restricted clinical datasets are not downloaded by this project. Put manually approved data under explicit local paths and pass them through `--data_path`.
