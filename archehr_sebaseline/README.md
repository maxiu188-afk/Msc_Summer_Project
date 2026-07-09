# ArchEHR-QA Semantic Entropy Baseline

This is the active project package for grounded clinical QA uncertainty baselines.

The implemented ArchEHR-QA path is:

```text
ArchEHR-QA XML/JSON/JSONL
-> structured grounded prompt
-> multi-sample cited answer generation
-> answer-level Semantic Entropy
-> citation-set uncertainty
-> lightweight answer-quality evaluation
-> lightweight analysis report
```

The goal is to evaluate uncertainty in LLM answers, not to build a complex answer-improvement or leaderboard-optimized QA system.

## Current Research Status

ArchEHR-QA is no longer the planned final evaluation/training dataset for this project. The pipeline is implemented and useful as an engineering baseline, but the available test key contains only clinician reference answers:

```text
case_id
clinician_answer
```

It does not contain sentence relevance labels, citation labels, or answer-quality labels. This means:

- SE can be computed on ArchEHR-QA.
- Generation, parsing, NLI clustering, answer SE, citation entropy, and token UQ can be inspected.
- Dev can be evaluated when sentence relevance labels are available.
- Test can only be evaluated in reference-only mode.
- ArchEHR-QA should not be used as the main SEP training target or final AUROC/ECE benchmark unless stronger gold labels are added.

The next research step is to choose a replacement dataset with usable answer-quality labels while reusing this package's generation, clustering, UQ, and evaluation infrastructure.

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

Evaluation:

```text
src/archehr_sebaseline/evaluation/archehr_answer_quality.py
src/archehr_sebaseline/evaluation/uncertainty_metrics.py
src/archehr_sebaseline/evaluation/pubmedqa_labels.py
```

Historical PubMedQA/Level 4 path:

```text
src/archehr_sebaseline/pipeline_level3.py
src/archehr_sebaseline/pipeline_level4.py
```

## Entry Points

Main ArchEHR-QA SE baseline:

```text
scripts/run_archehr_se.py
scripts/run_archehr_se.sbatch
scripts/evaluate_archehr_se.py
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
Ran 39 tests
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
  --output_dir $SCRATCHDIR/final_project/archehr_sebaseline/outputs/archehr_se_gemma3_12b_nli \
  --model_name google/gemma-3-12b-it \
  --num_samples 10 \
  --max_new_tokens 256 \
  --device cuda \
  --max_input_tokens 4096 \
  --torch_dtype bfloat16 \
  --local_files_only \
  --clustering_method nli \
  --nli_model_name microsoft/deberta-v2-xlarge-mnli \
  --nli_local_files_only \
  --overwrite
```

Slurm version:

```bash
DATA_PATH=$SCRATCHDIR/final_project/data/archehr_qa/dev/archehr-qa.xml \
OUTPUT_DIR=$SCRATCHDIR/final_project/archehr_sebaseline/outputs/archehr_se_gemma3_12b_dev20x10 \
MODEL_NAME=google/gemma-3-12b-it \
MAX_EXAMPLES=20 \
NUM_SAMPLES=10 \
MAX_NEW_TOKENS=256 \
MAX_INPUT_TOKENS=4096 \
DEVICE=cuda \
TORCH_DTYPE=bfloat16 \
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

## ArchEHR-QA Evaluation

The evaluation script is a second-stage pipeline. It does not regenerate answers. It reads an existing SE run directory plus the ArchEHR-QA key JSON:

```bash
python scripts/evaluate_archehr_se.py \
  --run_dir $SCRATCHDIR/final_project/archehr_sebaseline/outputs/archehr_se_gemma3_12b_dev20x10 \
  --key_path $SCRATCHDIR/final_project/data/archehr_qa/dev/archehr-qa_key.json \
  --overwrite
```

It writes `RUN_DIR/eval/` by default:

```text
answer_quality_generations.csv
answer_quality_examples.csv
se_eval_examples.csv
se_auroc.csv
se_ece.csv
se_reliability_bins.csv
se_rejection_curve.csv
evaluation_summary.json
evaluation_report.md
auroc_bar.svg
rejection_curve.svg
reliability_diagram.svg
```

Lightweight answer quality is currently:

```text
citation_score = max(strict citation F1, lenient citation F1)
answer_coverage = 0.5 * token recall + 0.5 * ROUGE-L recall
lexical_similarity = 0.5 * token F1 + 0.5 * ROUGE-L F1
relevance_score = 0.75 * answer_coverage + 0.25 * lexical_similarity
quality_score = 0.75 * citation_score + 0.25 * relevance_score
```

`is_low_quality` is then used as the target for SE evaluation. The default threshold is `0.4`. AUROC, ECE, and rejection curves are computed for answer SE, citation SE, token entropy, NLL, and cluster count.

If the key file has clinician reference answers but no gold evidence labels, the evaluator switches to reference-only mode. In that case `quality_score = relevance_score` and the default low-quality threshold is `0.2`.

Reference-only mode is diagnostic only. It cannot evaluate factuality, citation correctness, or evidence support, so it should not be treated as the final project benchmark.

## Documentation

Key docs:

```text
docs/archehr_se_baseline_plan.md
docs/archehr_evaluation_architecture.md
docs/dataset_pivot_status.md
docs/level5_evaluation_method.md
docs/progress_level4.md
docs/handoff_guide.md
docs/literature_notes.md
docs/extracted_literature_text/
```

## Data Boundary

Restricted clinical datasets are not downloaded by this project. Place approved data manually and pass it through explicit paths such as `--data_path`.
