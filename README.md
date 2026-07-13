# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
BioASQ Task B: grounded Gemma 3 12B / NLI baseline completed on summary, factoid, and list
Current compute host: Isambard (Runpod was a completed temporary recovery path)
Next evaluation work: validate deterministic quality scores against a reviewed or LLM-judged subset
Following stage: SEP only after the evaluation target is stable
```

ArchEHR-QA remains useful for testing grounded long-form generation, citation
parsing, answer Semantic Entropy, and citation uncertainty. It should not be the
main SEP training or final SE-evaluation dataset unless additional gold
answer-quality/evidence labels become available.

PubMedQA remains available only as an engineering smoke-test and historical
baseline path. It is not the preferred main research target because its short
yes/no/maybe labels do not match open long-form clinical generation.

BioASQ Task B is the current main experimental direction. The immediate work is
to validate its answer-quality target before SEP work. A first grounded Gemma 3
12B / 10-sample / NLI batch has completed on Golden summary (80 questions),
Golden factoid (50), Golden list (50), and a training-summary repeat (50).
Simpson's recommendation to continue exploring candidate datasets is retained
as a parallel validation activity, not as a reversal of the BioASQ direction.
See `archehr_sebaseline/docs/bioasq_runpod_results_20260713.md` for the
results and their interpretation boundary.

## Active Naming Policy

New BioASQ-main-track artifacts use BioASQ/`bioasq_se` names rather than new
`archehr` prefixes. This is forward-only: the `archehr_sebaseline/` package,
existing imports, historical ArchEHR-QA artifacts, and completed server
results remain unchanged to preserve working paths and provenance. See
`archehr_sebaseline/docs/naming_policy.md` for the exact rules.

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
bioasq_se_runpod_*.tar.gz
```

These are transport artifacts, not source code. Regenerate them from
`archehr_sebaseline/` when the code changes. Existing legacy archive names are
not renamed retroactively.
