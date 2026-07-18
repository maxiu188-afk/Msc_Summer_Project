# ArchEHR-QA Semantic Entropy Baseline Plan

Last updated: 2026-07-18

## Active BioASQ result (2026-07-18)

The new no-evidence BioASQ run is complete: 100 questions x 10 samples at
seeds 31/47, PubMedBERT-MNLI-MedNLI, binary Claude labels, and per-type UQ
evaluation. Discrete SE supports factoid/list but not summary, while
P(True)-blind is stronger than P(True) with ten high-temperature answers.
The project is paused for a careful direction review; do not treat SEP or a
P(True)-probe as the committed next step.

This document preserves the historical ArchEHR-QA engineering-baseline plan.
The historical BioASQ result reports are in `archived_low_usability/`; the
current no-evidence medical-UQ result is in `bioasq_medical_uq_protocol.md`
and the root `SE_BASELINE_LEVEL_PLAN.md`.

## Goal

The next project stage should focus on a clean Semantic Entropy baseline for
ArchEHR-QA-style grounded long-form clinical QA.

The goal is not to build the strongest possible ArchEHR-QA answering system.
The goal is to evaluate uncertainty in LLM answers.

Status update: this ArchEHR-QA SE baseline has now been implemented and run on
dev/test. It should be treated as an engineering baseline and diagnostic run,
not as the final project benchmark, because the available test key does not
contain gold evidence labels or answer-quality labels.

Therefore, we should avoid complex answer-improvement systems such as:

- multi-model ensembling,
- fine-tuned evidence selectors,
- answer reformulation loops,
- agentic prompt optimization,
- test-time scaling for answer quality,
- complex RAG pipelines beyond using the provided note sentences.

CogStack-KCL-UCL is useful mainly as task-design guidance: it shows that
ArchEHR-QA benefits from separating evidence grounding from answer generation
and from structured sentence-level citations. For this project, we should keep
that insight but implement only the simplest version needed for uncertainty
measurement.

## Research Question

The immediate research question is:

```text
Can Semantic Entropy quantify instability in grounded clinical answers to
ArchEHR-QA-style questions?
```

Useful uncertainty signals may include:

- variation in the generated answer text,
- variation in the semantic meaning of generated answers,
- variation in cited evidence sentences,
- disagreement between answer meaning and citation sets.

## Boundary

PubMedQA is no longer part of the main research path. It may remain useful for
engineering smoke tests, but it should not be used for the main ArchEHR-QA SE
baseline or for later probe targets.

The main pipeline should use ArchEHR-QA-style examples only.

Current dataset limitation:

- Dev has clinician answers and sentence relevance labels.
- Test has clinician answers only.
- Test therefore supports reference-only diagnostics, not factuality/citation
  evaluation.
- ArchEHR-QA should not be used as the main SEP target unless additional labels
  are added.

Restricted clinical data rules remain unchanged:

- Codex should not read real restricted ArchEHR-QA data.
- Real data should be placed manually by the user and passed through explicit
  config or command-line paths.
- Local automated tests should use fake or sanitized examples.

## Minimal ArchEHR-QA Example Schema

Normalize ArchEHR-QA-style inputs to:

```python
{
    "dataset": "archehr_qa",
    "id": str,
    "split": str,
    "patient_question": str,
    "clinician_question": str | None,
    "evidence_sentences": [
        {"sentence_id": str, "text": str}
    ],
    "gold_answer": str | None,
    "gold_relevant_sentence_ids": list[str] | None,
}
```

For the SE baseline, the most important fields are:

```text
patient_question
clinician_question
evidence_sentences
gold_answer
gold_relevant_sentence_ids
```

The pipeline should support running without `gold_answer` or
`gold_relevant_sentence_ids`, because SE itself does not require labels.

## Prompting Principle

Use one simple answer-generation prompt. Do not build a multi-stage answer
optimizer.

The prompt should:

1. Provide the patient question.
2. Provide the clinician question if available.
3. Provide all available note/evidence sentences with IDs.
4. Ask the model to answer only using the provided evidence.
5. Require sentence-level citations.
6. Prefer structured output that is easy to parse.

Recommended output format:

```json
[
  {
    "statement": "A concise answer statement.",
    "citation": "S3"
  },
  {
    "statement": "Another supported answer statement.",
    "citation": "S7"
  }
]
```

This format is inspired by the CogStack-KCL-UCL structured citation prompt, but
without their full two-stage answer-improvement pipeline.

## Generation Setup

For each ArchEHR-QA example:

1. Build one prompt.
2. Sample `N` answers from the same model.
3. Save raw outputs, parsed statements, citation IDs, token scores, and cleaned
   answer text.

Use `N=10` when compute allows, because the Semantic Entropy paper used ten
generations for sentence-length QA. Smaller `N` is acceptable only for smoke
tests.

Recommended generation outputs:

```text
examples.jsonl
prompts.jsonl
generations.jsonl
parsed_generations.jsonl
cleaned_generations.jsonl
generation_uq.csv
summary.txt
```

## Semantic Entropy Outputs

Compute at least three uncertainty views.

### 1. Answer Semantic Entropy

Cluster full generated answers by semantic equivalence using open-weight NLI.

Output:

```text
answer_clusters.jsonl
answer_se_scores.csv
```

This is the closest analogue to the original Semantic Entropy baseline.

### 2. Citation Entropy

Extract the set of cited sentence IDs from each generation and measure how
unstable the cited evidence is across samples.

Useful scores:

```text
citation_set_entropy
mean_pairwise_citation_jaccard
citation_vote_distribution
```

Output:

```text
citation_uq.csv
```

This is important for ArchEHR-QA because grounded-answer uncertainty may appear
as instability over evidence, even when answer wording is similar.

### 3. Joint Answer-Citation View

Record answer semantic clusters together with citation sets.

This does not need a complicated model. A simple artifact is enough:

```text
example_id
sample_id
answer_cluster_id
citation_ids
clean_answer
```

This lets us inspect cases such as:

- same answer meaning, different citations,
- different answer meanings, same citations,
- high answer SE and high citation entropy,
- low answer SE but unstable citations.

## Evaluation

Evaluation should be separate from SE computation.

SE can be computed without labels. If gold annotations are available, evaluate
whether uncertainty correlates with quality.

Preferred lightweight ArchEHR-QA quality targets:

1. Citation precision, recall, and F1 against relevant sentence IDs.
2. Essential evidence recall if essential/supplementary labels are available.
3. Answer-reference similarity against clinician-authored answer if available.
4. Manual review of a small number of high-SE and low-SE cases.

Avoid making PubMedQA-style `yes/no/maybe` labels part of the ArchEHR-QA main
evaluation.

## What Not To Do Yet

Do not implement SEP yet.

Do not train probes yet.

Do not add complex answer-improvement machinery before the SE baseline is
stable.

Do not optimize for ArchEHR-QA shared-task leaderboard score. That is a
different project objective.

## Implementation Steps

### Step 1: ArchEHR-QA Loader

Status: initial implementation complete.

Add a loader for local ArchEHR-QA-style XML, JSON, or JSONL files.

Deliverables:

```text
dataset adapter
fake/sanitized fixture
loader tests
```

### Step 2: Structured Prompt And Parser

Status: initial implementation complete.

Add a simple ArchEHR-QA prompt and JSON output parser.

Deliverables:

```text
prompt template
parsed statement/citation records
parser tests
```

### Step 3: ArchEHR SE Run Script

Status: initial implementation complete.

Add a script that runs multi-sample generation and computes answer SE.

Deliverables:

```text
scripts/run_archehr_se.py
answer_clusters.jsonl
answer_se_scores.csv
summary.txt
```

### Step 4: Citation UQ

Status: initial implementation complete.

Add citation-set uncertainty metrics.

Deliverables:

```text
citation_uq.csv
tests for citation entropy/Jaccard metrics
```

### Step 5: ArchEHR SE Analysis Report

Status: initial implementation complete.

Generate a lightweight report with:

```text
SE distribution
citation entropy distribution
examples with highest answer SE
examples with lowest answer SE
examples with stable answer but unstable citations
examples with unstable answer but stable citations
```

This completed the immediate ArchEHR-QA SE engineering baseline.

## Final ArchEHR-QA Status

Completed:

- official-style XML loading,
- structured cited answer prompting,
- JSON/fallback answer parsing,
- Gemma 3 12B generation support,
- NLI answer clustering,
- answer-level SE,
- citation-set UQ,
- token-level UQ,
- lightweight evaluation artifacts and SVG plots.

Latest test run:

```text
examples: 100
generations: 1000
num_samples: 10
parse_status: json for 1000/1000 generations
```

Key finding:

```text
ArchEHR-QA test lacks gold evidence/quality labels, so it cannot serve as the
final answer-quality benchmark for SE or SEP.
```

The next project step is dataset replacement, not additional ArchEHR-QA answer
optimization.

## Implemented Files

```text
src/archehr_sebaseline/answer_parsing.py
src/archehr_sebaseline/citation_uq.py
src/archehr_sebaseline/pipeline_archehr_se.py
scripts/run_archehr_se.py
tests/test_archehr_se.py
```

The existing dataset adapter and prompt module now also support the ArchEHR-QA
SE baseline:

```text
src/archehr_sebaseline/dataset_adapters.py
src/archehr_sebaseline/prompting.py
```

## Current Smoke Validation

Local fake/sanitized tests cover:

- ArchEHR-QA-style JSONL loading.
- Official-style ArchEHR-QA XML loading.
- Sentence ID preservation.
- Structured JSON answer parsing.
- Citation regex fallback parsing.
- Citation-set entropy and Jaccard.
- Full static-generator ArchEHR SE pipeline.

Current local validation:

```text
python -m unittest discover archehr_sebaseline\tests
Ran 43 tests
OK
```
