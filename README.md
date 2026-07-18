# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
BioASQ Task B: prior summary results archived as low-usability diagnostics
Current compute host: Isambard (Runpod was a completed temporary recovery path)
Active direction: no-evidence BioASQ medical-QA UQ, stratified by answer type
Latest validated run: 100x10 at seeds 31/47; PubMedBERT set-aware NLI and
  binary Claude labels completed for factoid/list/summary
Current decision: retain SE analysis for factoid/list; do not use P(True)-10
  or begin a P(True)-probe until the next direction is reviewed
```

ArchEHR-QA remains useful for testing grounded long-form generation, citation
parsing, answer Semantic Entropy, and citation uncertainty. It should not be the
main SEP training or final SE-evaluation dataset unless additional gold
answer-quality/evidence labels become available.

PubMedQA remains available only as an engineering smoke-test and historical
baseline path. It is not the preferred main research target because its short
yes/no/maybe labels do not match open long-form clinical generation.

BioASQ Task B remains the active dataset family, but all completed summary
results are now archived rather than treated as active UQ evidence. They mix
summary prompts, evidence-conditioned generation, a generic NLI clustering
model, and non-binary quality targets. The archived results are useful only for
provenance and artifact-format reference; they do not decide whether SE,
P(True), or a future probe is the preferred medical-QA UQ method.

The completed no-evidence BioASQ run uses a free biomedical, set-aware NLI
equivalence method; a low-temperature main answer judged only as `correct` or
`incorrect`; ten high-temperature answers for UQ; and paired P(True) with and
without those samples. Overall, normalized discrete SE is 0.729/0.759 AUROC
for seeds 31/47, while P(True)-blind is 0.781/0.824 and P(True)-10 is
0.717/0.758. These are descriptive two-seed results, not a final method choice.
See `archehr_sebaseline/docs/bioasq_medical_uq_protocol.md` for the full
per-type outcome and `archehr_sebaseline/docs/archived_low_usability/README.md`
for retired material.

## Active Naming Policy

New BioASQ-main-track artifacts use BioASQ/`bioasq_se` names rather than new
`archehr` prefixes. This is forward-only: the `archehr_sebaseline/` package,
existing imports, historical ArchEHR-QA artifacts, and completed server
results remain unchanged to preserve working paths and provenance. See
`archehr_sebaseline/docs/naming_policy.md` for the exact rules.

The BioASQ evaluator uses versioned post-processing with type-specific
quality diagnostics, threshold sensitivity, continuous-risk association, and
bootstrap intervals. It follows BioASQ metric families but is not the official
evaluator. A fixed Qwen judge was completed, but its ceiling effect produced no
low-quality labels and only weak continuous agreement. A stronger API judge is
deferred until conventional SE behaviour and the evaluation target are better
understood.

## Folder Structure

```text
code/
  archehr_sebaseline/        Active maintained SE/UQ package and dataset adapters
  semantic_uncertainty/      Reference implementation from the Semantic Entropy work
  server_results/            Active results; archived diagnostics are in archived_low_usability/
  .agents/                   Local Codex/agent state
  .git/                      Repository metadata
```

## Root-Level Planning Files

These root-level files are intentionally kept at the workspace level:

```text
codex_task_overview_archehr_uq.txt
SE_BASELINE_LEVEL_PLAN.md
ISAMBARD_COMMANDS.md
```

Their roles are:

- `codex_task_overview_archehr_uq.txt`: original project overview and broad
  requirements.
- `SE_BASELINE_LEVEL_PLAN.md`: active stage plan, current status, and dataset
  pivot notes.
- `ISAMBARD_COMMANDS.md`: common local and Isambard operational commands;
  use the package guide for the full BioASQ launch procedure.

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
