# ArchEHR-QA Semantic Entropy Baseline

This is the active project package for biomedical/clinical QA uncertainty
baselines.

The current main replacement dataset path is BioASQ Task B:

```text
BioASQ factoid/list questions
-> no-evidence biomedical prompt
-> multi-sample answer generation
-> answer-level Semantic Entropy
-> token-level UQ
-> binary correct/incorrect evaluation
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

The replacement dataset direction is now BioASQ. PubMedQA is not the main
research dataset, but its official 500-question PQA-L test subset is the fixed
external dataset for zero-shot evaluation of the two frozen Probe models.

### Current BioASQ decision

The completed BioASQ summary runs are archived as low-usability diagnostics.
They are not current benchmark results because they combine summary prompts,
evidence-conditioned generation, generic NLI clustering, and non-binary quality
targets. Their reports and raw outputs remain available for provenance under
`../archive_unused/docs/historical_results/` and
`../archive_unused/results/`.

Phase 1 is complete. The 1,000-question stratified BioASQ baseline uses free
biomedical set-aware NLI, binary `correct`/`incorrect` judging of the
low-temperature main answer, all SE/token UQ baselines, verbal confidence, and
blind P(True). With 991 valid labels per seed, P(True)-blind is strongest
overall (0.811/0.821 AUROC); SE is retained as a strong list-specific baseline
(0.845/0.881 discrete SE). P(True)-10 is retired. See
`docs/bioasq_medical_uq_results_20260718.md` for the final table and
`docs/bioasq_medical_uq_protocol.md` for the protocol.

Phase 2 uses the full eligible BioASQ training13b corpus rather than treating
the Phase-1 1,000 questions as its complete dataset. The observed Phase-1 IDs
are train-side reference examples; an exact-question-grouped, type-stratified
manifest creates the remaining train/validation/test split. See
`docs/phase2_bioasq_dataset_split.md`.

The active research plan is the root `../PHASE2_PROBE_PLAN.md`: P(True)-Probe
approximates direct blind P(True), while Accuracy-Probe predicts the binary
Claude correctness label. The initial collection contract is fixed at blocks
`24/32/40/48` and positions `TBG/SLT/LT`; high-temperature multi-sample UQ
remains an optional later comparison. The Isambard smoke/full collection has
completed and the initial local linear-Probe pass is frozen: fit on train,
select type/layer/token on validation, and evaluate once on test. P(True)-Probe
uses hard-threshold-even L2 logistic regression at block 24/LT (P(True)-target
AUROC 0.9026); Accuracy-Probe uses L2 logistic regression at block 24/LT
(Claude-incorrect AUROC 0.8058, AP 0.8884). The 72 blank Claude labels are
excluded only from Accuracy-Probe (3,090/384/384 valid train/validation/test
rows). The current PubMedQA Appendix-C context v2 result reaches 72.4% strict
accuracy, compared with 59.0% for context v1, and improves Macro-F1 from 0.5211
to 0.5659. Frozen P(True)-Probe is the strongest v2 error-ranking score
(AUROC 0.6839, AP 0.4519), followed by blind P(True) (0.6490 /
0.4210), verbalized confidence (0.6402 / 0.4164), and frozen Accuracy-Probe
(0.5901 / 0.3916). Both context versions use the same 500 IDs, model settings,
and frozen Probes without target fitting. The question-only result is archived
because it omitted the article evidence defining the official labels. See
`docs/pubmedqa_frozen_probe_transfer.md`, `../PHASE2_PROBE_PLAN.md`, and
`docs/experiment_runtime_log.md`.

The PubMedQA follow-up is fixed error analysis of the completed v2 result,
especially its 12.7% recall on the minority `maybe` label. This does not reopen
the prompt or authorize repeated tuning.

The in-domain efficiency benchmark is complete on the 384 valid-labelled
Phase-2 test questions. Smoke job `5761273` passed; full job `5761275` failed
before model loading because of a nonexistent node-local temporary directory,
and replacement job `5773786` then completed in `03:31:33` with its health
check passing. One block-24/LT Probe takes about `60.5 ms/question`, both
jointly `60.54 ms`, blind P(True) `154.4 ms`, ten-sample normalized NLL
`30.55 s`, and discrete SE/cluster count `32.75 s`. Both Probes generate zero
tokens; blind P(True) also generates no free-running tokens but scores fixed
`True` and `False` continuations in two LM calls. The full AUROC/AP, GPU time,
token count, and throughput tables are in
`docs/phase2_uq_efficiency_benchmark.md`.

Phase-2 is now complete. On the same 384 test questions, Accuracy-Probe minus
blind P(True) has AUROC difference `+0.01584` with paired-bootstrap 95% CI
`[-0.03890, 0.07173]`, so the point-estimate advantage is not statistically
resolved. The one validation-fitted two-Probe fusion reaches test AUROC
`0.8125`, AP `0.8871`, and Brier `0.1688`; its small mixed change versus
Accuracy-Probe remains exploratory. The frozen protocol and complete tables
are in `docs/phase2_probe_completion_statistics.md`.

The post-Simpson main line now studies the conditions under which Semantic
Entropy or blind P(True) performs better. After the current
efficiency/bootstrap/fusion completion, the next experiment is a paired
BioASQ-summary prompt intervention. The shorter condition adds only:

```text
Keep the answer to one or two brief but complete sentences.
Include only information needed to answer the question, without extra background.
```

It has no word/token cap or hard truncation, and all other generation, UQ, NLI,
and judging settings stay fixed. A later Claude-versus-NLI reclustering stage
is conditional and diagnostic only: it tests whether NLI is a major SE
bottleneck, not whether Claude-SE should be deployed. Any optional model-scale
work compares SE and blind P(True) without training additional Probes. See
`../PHASE2_PROBE_PLAN.md`.

New BioASQ-main-track artifacts use BioASQ/`bioasq_se` names rather than new
`archehr` prefixes. This does not rename the package, Python imports, or
historical ArchEHR-QA files.

## Package Layout

```text
archehr_sebaseline/
  scripts/                  CLI and Slurm entry points
  src/archehr_sebaseline/   Python package
  tests/                    Unit tests and smoke fixtures
  docs/                     Active protocols, results, and runtime records
  requirements.txt          Minimal Python dependencies
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
scripts/benchmark_phase2_uq_efficiency.py
scripts/run_phase2_uq_efficiency_isambard.sbatch
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
the server pipeline. The judge emits exactly `correct` or `incorrect`; factoid
and list use `exact_answers`, while summary uses `ideal_answers`. Its system
instruction explicitly establishes this as offline academic annotation rather
than medical advice. First create a reviewable manifest, then submit it:

```bash
python scripts/run_bioasq_claude_judge.py prepare \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31

python scripts/run_bioasq_claude_judge.py submit \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31 \
  --model claude-sonnet-5 --effort low --max_tokens 32

python scripts/run_bioasq_claude_judge.py download \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31

python scripts/evaluate_bioasq_claude_judge.py \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31 \
  --allow_incomplete_labels \
  --overwrite
```

The final command writes `claude_binary_main_answer_judge/claude_uq_*.csv` plus
independent `by_type/factoid`, `by_type/list`, and `by_type/summary` artifacts.
`incorrect` is always the risk-positive label. Do not feed Claude labels into
the archived deterministic evaluator or use high-temperature sample labels as
a substitute for the low-temperature main answer.

### Simple UQ baselines

The Level 4 token-score artifacts already provide average token log-probability,
average token entropy, and sequence NLL. The BioASQ evaluator automatically
compares all three with SE; negative average token log-probability is the same
quantity as normalized NLL, and sequence NLL is retained separately to expose
its answer-length sensitivity.

Verbalized confidence and blind P(True) are implemented as one optional
model-backed post-processing pass over each low-temperature main answer. The
binary Claude evaluator merges them with SE and token UQ:

```bash
python scripts/run_self_report_uq.py \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31 \
  --model_name google/gemma-3-12b-it \
  --device cuda \
  --torch_dtype bfloat16 \
  --overwrite

python scripts/evaluate_bioasq_claude_judge.py \
  --run_dir outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31 \
  --allow_incomplete_labels \
  --overwrite
```

The post-processing script uses the `T=0.1` answer as the object under
evaluation. P(True) receives only the question and proposed answer (plus
evidence if that generation prompt used evidence). It uses neither retrieval
nor NLI, but it does load the answer model once.

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
docs/README.md
docs/bioasq_medical_uq_protocol.md
docs/bioasq_medical_uq_results_20260718.md
docs/phase2_bioasq_dataset_split.md
docs/phase2_uq_efficiency_benchmark.md
docs/pubmedqa_frozen_probe_transfer.md
docs/semantic_entropy_generation_protocol.md
docs/experiment_runtime_log.md
```

Superseded ArchEHR/Level-4 plans, handoff notes, Runpod guides, extracted paper
text, and low-usability result reports are consolidated under
`../archive_unused/`.

## Data Boundary

Restricted clinical datasets are not downloaded by this project. Place approved data manually and pass it through explicit paths such as `--data_path`.
