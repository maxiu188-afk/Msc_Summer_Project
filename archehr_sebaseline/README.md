# ArchEHR-QA Semantic Entropy Baseline

This is the active project package for the grounded clinical QA uncertainty baseline.

The current maintained research path is a simple ArchEHR-QA Semantic Entropy baseline:

```text
ArchEHR-QA XML/JSON/JSONL
-> structured grounded prompt
-> multi-sample cited answer generation
-> answer-level Semantic Entropy
-> citation-set uncertainty
-> lightweight analysis report
```

The goal is to evaluate uncertainty in LLM answers, not to build a complex answer-improvement or leaderboard-optimized ArchEHR-QA system.

## Package Layout

```text
archehr_sebaseline/
  scripts/                  CLI and Slurm entry points
  src/archehr_sebaseline/   Python package
  tests/                    Unit tests and smoke fixtures
  docs/                     Plans, handoffs, method notes, literature extracts
  requirements.txt          Minimal Python dependencies
  SERVER_RUN.md             Historical server run guide
```

## Important Modules

Data loading:

```text
src/archehr_sebaseline/dataset_adapters.py
src/archehr_sebaseline/common_schema.py
src/archehr_sebaseline/data_io.py
```

Prompting and parsing:

```text
src/archehr_sebaseline/prompting.py
src/archehr_sebaseline/answer_parsing.py
```

Generation and token scores:

```text
src/archehr_sebaseline/generation.py
src/archehr_sebaseline/baseline_uq.py
```

Semantic Entropy and clustering:

```text
src/archehr_sebaseline/entropy.py
src/archehr_sebaseline/simple_clustering.py
src/archehr_sebaseline/nli_clustering.py
src/archehr_sebaseline/semantic_scores.py
```

ArchEHR-QA SE baseline:

```text
src/archehr_sebaseline/pipeline_archehr_se.py
src/archehr_sebaseline/citation_uq.py
```

Historical PubMedQA/Level 4 path:

```text
src/archehr_sebaseline/pipeline_level3.py
src/archehr_sebaseline/pipeline_level4.py
src/archehr_sebaseline/evaluation/pubmedqa_labels.py
```

## Entry Points

Main ArchEHR-QA SE baseline:

```text
scripts/run_archehr_se.py
scripts/run_archehr_se.sbatch
```

Historical Level 3/4 scripts:

```text
scripts/run_level3.py
scripts/run_level3.sbatch
scripts/run_level4.py
scripts/run_level4.sbatch
scripts/check_level4_outputs.py
scripts/evaluate_pubmedqa_labels.py
```

## Local Tests

From `D:\work\FinalProject\code`:

```powershell
python -m unittest discover archehr_sebaseline\tests
```

Current expected result:

```text
Ran 30 tests
OK
```

## ArchEHR-QA SE Baseline

The main script reads ArchEHR-QA-style XML, JSON, or JSONL files.

Local smoke example:

```powershell
python archehr_sebaseline\scripts\run_archehr_se.py `
  --data_path D:\path\to\archehr-qa.xml `
  --output_dir archehr_sebaseline\outputs\archehr_se_smoke `
  --model_name sshleifer/tiny-gpt2 `
  --num_samples 2 `
  --max_examples 1 `
  --clustering_method exact `
  --overwrite
```

Server-style NLI run:

```bash
python scripts/run_archehr_se.py \
  --data_path $SCRATCHDIR/final_project/data/archehr_qa/dev/archehr-qa.xml \
  --output_dir $SCRATCHDIR/final_project/archehr_sebaseline/outputs/archehr_se_qwen25_nli \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --num_samples 10 \
  --max_new_tokens 256 \
  --device cuda \
  --torch_dtype float16 \
  --local_files_only \
  --clustering_method nli \
  --nli_model_name microsoft/deberta-v2-xlarge-mnli \
  --nli_local_files_only \
  --overwrite
```

Slurm version:

```bash
DATA_PATH=$SCRATCHDIR/final_project/data/archehr_qa/dev/archehr-qa.xml \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/archehr_se_dev_n10 \
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct \
MAX_EXAMPLES=20 \
NUM_SAMPLES=10 \
MAX_NEW_TOKENS=256 \
DEVICE=cuda \
TORCH_DTYPE=float16 \
LOCAL_FILES_ONLY=1 \
CLUSTERING_METHOD=nli \
NLI_MODEL_NAME=microsoft/deberta-v2-xlarge-mnli \
NLI_LOCAL_FILES_ONLY=1 \
sbatch --time=03:00:00 scripts/run_archehr_se.sbatch
```

## ArchEHR-QA SE Outputs

```text
examples.jsonl
prompts.jsonl
generations.jsonl
parsed_generations.jsonl
cleaned_generations.jsonl
answer_clusters.jsonl
answer_se_scores.csv
generation_uq.csv
citation_uq.csv
analysis_report.md
summary.txt
```

## Documentation

Key docs:

```text
docs/archehr_se_baseline_plan.md
docs/level5_evaluation_method.md
docs/progress_level4.md
docs/handoff_guide.md
docs/literature_notes.md
docs/extracted_literature_text/
```

## Data Boundary

Restricted clinical datasets are not downloaded by this project. Place approved data manually and pass it through explicit paths such as `--data_path`.
