# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
BioASQ Task B: baseline complete; two matched Isambard summary 100x10 runs passed
Current compute host: Isambard (Runpod was a completed temporary recovery path)
Main result: historical lexical-quality scores replicate strongly, but SE AUROC is modest and seed-sensitive
Active run: no-evidence direct-answer temperature-1.0 paired repeat submitted; evidence-conditioned NLI evaluation complete
Next work: retrieve the direct-answer pair, compare it with the evidence-conditioned result, then calibrate the final target against manual review
Deferred work: larger closed-model judge study and SEP supervision
```

ArchEHR-QA remains useful for testing grounded long-form generation, citation
parsing, answer Semantic Entropy, and citation uncertainty. It should not be the
main SEP training or final SE-evaluation dataset unless additional gold
answer-quality/evidence labels become available.

PubMedQA remains available only as an engineering smoke-test and historical
baseline path. It is not the preferred main research target because its short
yes/no/maybe labels do not match open long-form clinical generation.

BioASQ Task B is the current main experimental direction. In addition to the
historical RunPod Golden summary/factoid/list batch, two matched Isambard
training-summary runs have completed with 100 questions and ten generations per
question. Their answer-quality scores correlate at 0.993 across seeds, while
discrete-SE AUROC changes from 0.687 to 0.609 and the repeat confidence interval
includes chance. This supports a cautious baseline result, not a robust claim
that SE is the strongest uncertainty method.
The previous ROUGE-only operational target is retained as historical evidence,
but the active local evaluator now reports three quality axes: lexical
reference-answer coverage, document-level overlap between answer citations and
BioASQ's standard documents, and a three-class NLI ideal-answer coverage score
(good/partial/poor = 1.0/0.5/0.0). New runs combine the available axes with a
geometric mean. Its fixed low-quality threshold must be calibrated against the
manually reviewed set rather than treated as a pass mark.

On 2026-07-16, the paired temperature-sensitivity repeat at `temperature=1.0`
and `top_p=0.9` completed for both seeds (jobs 5679663 and 5679664). Both
100x10 outputs passed their Level 4 health checks. A same-config local
citation-aware reference evaluation found virtually unchanged mean answer
quality versus temperature 0.8, while within-question semantic variation rose
slightly. The final three-axis NLI re-evaluation completed in 01:31 and 01:26
for seeds 31 and 47, giving mean three-axis scores of 0.1968 and 0.1864. A
matched no-evidence direct-answer ablation (jobs 5684358 and 5684360) is now
submitted with the same temperature, seeds, and generation settings. Do not
compare the direct-answer condition on the citation axis: its final target uses
the valid ROUGE + ideal-answer-NLI two-axis fallback.
Simpson's recommendation to continue exploring candidate datasets is retained
as a parallel validation activity, not as a reversal of the BioASQ direction.
See `archehr_sebaseline/docs/bioasq_isambard_results_20260715.md` for the latest
results and `archehr_sebaseline/docs/bioasq_runpod_results_20260713.md` for the
historical task-type batch.

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
  server_results/            Downloaded or copied server outputs and analysis artifacts
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
