# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
Current main dataset direction: test BioASQ Task B
Parallel research advice: continue comparing credible replacement datasets
Evaluation experiment: compare deterministic metrics with an LLM-based judge
Following stage: SEP after the dataset and evaluation target are stable
```

ArchEHR-QA remains useful for testing grounded long-form generation, citation
parsing, answer Semantic Entropy, and citation uncertainty. It should not be the
main SEP training or final SE-evaluation dataset unless additional gold
answer-quality/evidence labels become available.

PubMedQA remains available only as an engineering smoke-test and historical
baseline path. It is not the preferred main research target because its short
yes/no/maybe labels do not match open long-form clinical generation.

BioASQ Task B is the current main experimental direction. The immediate work is
to test its summary, factoid, list, and yes/no paths using the existing
generation and uncertainty pipeline. Simpson's recommendation to continue
exploring candidate datasets is retained as a parallel validation activity, not
as a reversal of the BioASQ direction.

The current answer-quality evaluators are also treated as provisional. Their
dataset-specific thresholds, heuristic parsing, and manually weighted scores
should be adapted and validated for any replacement dataset. An LLM-as-a-judge
path is planned as a flexible comparison for long-form answer evaluation, with
fixed prompts/configuration and validation against labels or manual review where
possible.

## Folder Structure

```text
code/
  archehr_sebaseline/        Active maintained SE/UQ package and dataset adapters
  semantic_uncertainty/      Reference implementation from the Semantic Entropy work
  server_results/            Downloaded or copied server outputs and analysis artifacts
  .agents/                   Local Codex/agent state
  .git/                      Repository metadata
```

## Root-Level Planning Files

These root-level files are intentionally kept at the workspace level:

```text
codex_task_overview_archehr_uq.txt
SE_BASELINE_LEVEL_PLAN.md
```

Their roles are:

- `codex_task_overview_archehr_uq.txt`: original project overview and broad
  requirements.
- `SE_BASELINE_LEVEL_PLAN.md`: active stage plan, current status, and dataset
  pivot notes.

## Active Project

The active maintained code lives in:

```text
archehr_sebaseline/
```

See:

```text
archehr_sebaseline/README.md
```

for code layout, script entry points, tests, and server run commands.

The package-level docs under `archehr_sebaseline/docs/` now separate:

- implemented ArchEHR-QA engineering baseline,
- current dataset limitation,
- evaluation method notes,
- handoff and next-step guidance.

## Reference Code

The directory:

```text
semantic_uncertainty/
```

contains reference code for the original Semantic Entropy project. Treat it as
reference material, not as the active codebase.

## Notes And Literature Extracts

Project notes, literature-derived notes, and extracted paper text should live
under:

```text
archehr_sebaseline/docs/
```

Extracted PDF text snapshots are organized under:

```text
archehr_sebaseline/docs/extracted_literature_text/
```

The original PDF papers remain outside this code workspace under:

```text
D:\work\FinalProject\literature
```

## Server Packages

Server upload archives may appear at the root, for example:

```text
archehr_sebaseline_server_*.zip
archehr_sebaseline_server_*.tar.gz
```

These are transport artifacts, not source code. Regenerate them from
`archehr_sebaseline/` when the code changes.
