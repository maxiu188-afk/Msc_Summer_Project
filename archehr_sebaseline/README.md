# ArchEHR-QA Semantic Entropy Baseline

This is the active project package for grounded biomedical/clinical QA
uncertainty baselines.

The current main replacement dataset path is BioASQ Task B:

```text
BioASQ questions + PubMed snippets
-> grounded biomedical prompt
-> multi-sample answer generation
-> answer-level Semantic Entropy
-> token-level UQ
-> dataset-specific quality evaluation
```

ArchEHR-QA remains implemented as an engineering diagnostic baseline.

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

The replacement dataset direction is now BioASQ. PubMedQA remains available as
a short-answer engineering smoke path, but it is not the preferred main SE
dataset.

### Latest BioASQ baseline evidence

The baseline now includes two matched Isambard training-summary runs with Gemma
3 12B, 100 questions, ten samples per question, token scores, self-report UQ,
and bidirectional-entailment NLI clustering. Both health checks passed. Mean
quality is 0.2238 versus 0.2245 and paired example-quality rho is 0.993, but
discrete-SE AUROC changes from 0.687 to 0.609. The repeat confidence interval
includes 0.5, and token entropy/P(True) outperform SE in seed 47. The baseline
therefore supports a positive but modest uncertainty-quality relationship, not
a stable SE-superiority claim.

A fixed 30-question Qwen judge also completed. Its mean score was 0.871, but it
assigned no example below the preregistered 0.5 threshold, making judge-label
AUROC undefined. A larger closed-model judge remains possible but is deferred
until conventional SE failure analysis is complete.

The raw required-batch archive and a cautious result table are documented in
`docs/bioasq_runpod_results_20260713.md`. Future experiments run on Isambard;
`RUNPOD.md` is retained only as a reproducibility record for this batch.
For a fresh BioASQ launch on Isambard, follow `ISAMBARD_BIOASQ.md`.
The latest Isambard results and interpretation boundary are documented in
`docs/bioasq_isambard_results_20260715.md`; runtimes are in
`docs/experiment_runtime_log.md`.

The paired Isambard temperature-sensitivity runs completed on 2026-07-16. They
keep the 100-question, ten-sample, 192-token configuration and the
seed-31/seed-47 pair, but explicitly use `temperature=1.0` and `top_p=0.9`.
Jobs 5679663 and 5679664 both passed Level 4 health checks. Under a common
citation-aware reference target, mean answer quality was unchanged from the
temperature-0.8 pair, while within-question semantic diversity increased
slightly. The three-axis ideal-answer NLI re-evaluation then completed in
01:31 and 01:26, with mean three-axis quality of 0.1968 and 0.1864. A matched
no-evidence direct-answer pair (jobs 5684358 and 5684360) is submitted with
the same temperature, seeds, generation settings, NLI clustering, and
self-report UQ. Its citation axis is intentionally not applicable, so its
final quality target uses the evaluator's ROUGE + ideal-answer-NLI fallback.

The main-answer protocol was corrected on 2026-07-17 to match the Semantic
Entropy reference implementation. Ten samples at `T=1.0`, `top_p=0.9`, and
`top_k=50` are reserved for SE and other UQ scores. A separately generated
single `T=0.1` answer is the only answer used to establish quality. Four
supplemental Isambard jobs (5692776, 5692777, 5692779, and 5692780) are adding
these files to the completed temperature-1.0 evidence/no-evidence pair. The
final label source will be a local Claude Sonnet 5 three-way text comparison;
`poor` is the binary error label. The prior 4,000 high-temperature Claude
labels remain an auxiliary, non-primary analysis. Details and commands are in
`docs/semantic_entropy_generation_protocol.md`.

The active reference evaluator now retains answer-reference coverage and adds
document-level overlap between cited snippet documents and BioASQ standard
documents. It combines the two by geometric mean when gold document metadata
is available. Its deterministic comparator uses the lowest 30% of quality
scores within a BioASQ question type, avoiding a score-scale-dependent fixed
cutoff. The fixed threshold remains a diagnostic and must be calibrated against
reviewed examples; the final UQ label is the low-temperature Claude `poor`
classification described below.

New BioASQ-main-track artifacts use BioASQ/`bioasq_se` names rather than new
`archehr` prefixes. This does not rename the package, Python imports, or
historical ArchEHR-QA files. See `docs/naming_policy.md`.

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
src/archehr_sebaseline/evaluation/bioasq_llm_judge.py
src/archehr_sebaseline/evaluation/bioasq_quality.py
src/archehr_sebaseline/evaluation/uncertainty_metrics.py
src/archehr_sebaseline/evaluation/pubmedqa_labels.py
```

Historical PubMedQA/Level 4 path:

```text
src/archehr_sebaseline/pipeline_level3.py
src/archehr_sebaseline/pipeline_level4.py
```

The Level 4 path now also supports BioASQ through the common-schema adapter.

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
scripts/evaluate_bioasq_quality.py
scripts/evaluate_bioasq_nli_isambard.sbatch
scripts/run_bioasq_llm_judge.py
scripts/run_bioasq_llm_judge.sbatch
scripts/run_bioasq_isambard.sbatch
scripts/runpod_remaining_experiments.sh
```

## BioASQ SE Baseline

BioASQ is loaded through the common Level 4 pipeline. The adapter accepts either
a single BioASQ JSON file or a directory of JSON files such as the golden
enriched batches.

Supported dataset aliases:

```text
bioasq
bioasq_summary
bioasq_factoid
bioasq_list
bioasq_yesno
```

Recommended first main run is `bioasq_summary`, because summary answers are
long enough for answer-level SE to be meaningful:

```bash
python scripts/run_level4.py \
  --dataset bioasq_summary \
  --data_path $SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json \
  --split train13b \
  --output_dir $SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_summary_gemma3_12b_100x10 \
  --model_name google/gemma-3-12b-it \
  --num_samples 10 \
  --max_examples 100 \
  --max_new_tokens 192 \
  --device cuda \
  --max_input_tokens 4096 \
  --torch_dtype bfloat16 \
  --local_files_only \
  --clustering_method nli \
  --nli_model_name microsoft/deberta-v2-xlarge-mnli \
  --nli_local_files_only \
  --overwrite
```

The golden enriched batches can be passed as a directory:

```bash
python scripts/run_level4.py \
  --dataset bioasq_summary \
  --data_path $SCRATCHDIR/final_project/data/Task13BGoldenEnriched \
  --split golden13b \
  --output_dir $SCRATCHDIR/final_project/archehr_sebaseline/outputs/bioasq_golden_summary_gemma3_12b \
  --model_name google/gemma-3-12b-it \
  --num_samples 10 \
  --max_new_tokens 192 \
  --device cuda \
  --max_input_tokens 4096 \
  --torch_dtype bfloat16 \
  --local_files_only \
  --clustering_method nli \
  --nli_model_name microsoft/deberta-v2-xlarge-mnli \
  --nli_local_files_only \
  --overwrite
```

For stricter correctness supervision, use `bioasq_factoid` or `bioasq_list`;
their normalized examples preserve `exact_answers` for a follow-up evaluator.

### Lightweight BioASQ evaluation

After a Level 4 BioASQ run, reference mode evaluates existing artifacts without
loading a generation or NLI model:

```bash
python scripts/evaluate_bioasq_quality.py \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10 \
  --quality_target mean \
  --quality_threshold 0.15 \
  --relative_risk_fraction 0.30 \
  --bootstrap_samples 1000
```

This is a transparent local approximation of the official metric families,
not the official BioASQ service. Summary answers use unstemmed ROUGE-2 and
ROUGE-SU4 F1; yes/no uses accuracy; factoid records strict, answer-first, and
lenient exact-answer diagnostics; list uses normalized set precision/recall/F1.
Citation IDs are checked against the provided snippet IDs, but this is not a
claim-entailment check. The evaluator uses the per-type bottom 30% as a
deterministic comparison label and retains the fixed threshold as a diagnostic.
It writes bootstrap intervals, Spearman correlation with continuous risk
(`1 - quality`), and a coverage-risk summary. These targets are not BioASQ pass
marks and do not replace the low-temperature Claude outcome for final UQ
results.

For the primary grounded-quality analysis, use the already-maintained open NLI
model to test whether each answer claim is entailed by the supplied gold
snippets. This is post-processing only; it does not regenerate any answer:

```bash
python scripts/evaluate_bioasq_quality.py \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10 \
  --output_dir outputs/bioasq_summary_gemma3_12b_100x10/bioasq_eval_grounded \
  --quality_mode grounded \
  --grounding_nli_model microsoft/deberta-v2-xlarge-mnli \
  --grounding_nli_device cuda \
  --grounding_max_claims 4 \
  --grounding_max_evidence_sentences 10 \
  --bootstrap_samples 1000 \
  --overwrite
```

Grounded mode preserves the reference-quality, entailment, contradiction, and
cited-claim diagnostics in separate columns. Its composite is the geometric
mean of reference quality and net evidence support, so a reference-like answer
that contradicts the supplied snippets receives a low score. It remains an
open-NLI proxy rather than clinical expert adjudication; validate it on a
reviewed subset before using it as SEP supervision.

### Claude main-answer evaluation

After the low-temperature `best_generations.jsonl` file has been downloaded,
Claude is run locally as a separate post-processing stage. It does not alter
the server pipeline or its `bioasq_eval/` artifacts. The judge compares each
candidate only with BioASQ ideal answers and emits exactly one of `good`,
`partial`, or `poor`; its system instruction explicitly establishes this as
offline academic annotation rather than medical advice. Use low reasoning
effort and a 32-token response cap:

```bash
python scripts/run_bioasq_claude_judge.py submit \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10_temp1p0_seed31 \
  --model claude-sonnet-5 --effort low --max_tokens 32

python scripts/run_bioasq_claude_judge.py download \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10_temp1p0_seed31

python scripts/evaluate_bioasq_claude_judge.py \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10_temp1p0_seed31 \
  --overwrite
```

The final command writes isolated `claude_judge/claude_uq_*.csv` artifacts;
only `poor` is positive for AUROC and AURAC. Do not feed Claude labels into the
earlier deterministic quality evaluator or use the historical high-temperature
sample labels as a substitute for the low-temperature main answer.

### Simple UQ baselines

The Level 4 token-score artifacts already provide average token log-probability,
average token entropy, and sequence NLL. The BioASQ evaluator automatically
compares all three with SE; negative average token log-probability is the same
quantity as normalized NLL, and sequence NLL is retained separately to expose
its answer-length sensitivity.

Verbalized confidence and P(True) are implemented as one optional model-backed
post-processing pass. Run it once for a completed output, then re-run the
BioASQ evaluator to include both fields automatically:

```bash
python scripts/run_self_report_uq.py \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10 \
  --model_name google/gemma-3-12b-it \
  --device cuda \
  --torch_dtype bfloat16 \
  --overwrite

python scripts/evaluate_bioasq_quality.py \
  --run_dir outputs/bioasq_summary_gemma3_12b_100x10 \
  --bootstrap_samples 1000 \
  --overwrite
```

The post-processing script scores every sampled answer, then averages the ten
values per question for a fair comparison with answer-level SE. It uses neither
retrieval nor NLI, but it does load the answer model once.

## Local Tests

From the repository root on macOS/Linux:

```bash
python -m unittest discover archehr_sebaseline/tests
```

From the repository root on Windows PowerShell:

```powershell
python -m unittest discover archehr_sebaseline/tests
```

Current expected result:

```text
Ran 61 tests
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
