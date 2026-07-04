# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

The current active research path is:

```text
ArchEHR-QA-style grounded long-form QA
-> multi-sample answer generation
-> Semantic Entropy over generated answers
-> citation-set uncertainty
```

PubMedQA remains available only as an engineering smoke-test and historical
baseline path. It is no longer the main research target.

## Folder Structure

```text
code/
  archehr_sebaseline/        Active project code for the ArchEHR-QA SE baseline
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
HANDOFF_LEVEL4_BASELINE.md
```

Their roles are:

- `codex_task_overview_archehr_uq.txt`: original project overview and broad
  requirements.
- `SE_BASELINE_LEVEL_PLAN.md`: active stage plan and current direction.
- `HANDOFF_LEVEL4_BASELINE.md`: historical handoff for the Level 4 baseline.

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
archehr_sebaseline_clean_level4.tar.gz
```

These are transport artifacts, not source code. Regenerate them from
`archehr_sebaseline/` when the code changes.

