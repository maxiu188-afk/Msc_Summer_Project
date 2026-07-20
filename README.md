# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
BioASQ Task B: prior summary results archived as low-usability diagnostics
Current compute host: Isambard (Runpod was a completed temporary recovery path)
Active direction: Phase-2 BioASQ hidden-state probes, stratified by answer type
Latest validated run: Phase-2 3,930-question single-answer collection and
  two frozen linear hidden-state Probes on Gemma 3 12B
Current Phase-2 tracks: P(True)-Probe (direct P(True) fidelity) and
  Claude-label Accuracy-Probe (answer correctness)
Current result: train-only fitting, validation-only feature selection, and one
  held-out test evaluation are complete; cross-dataset transfer is next
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

The completed Phase-1 no-evidence BioASQ baseline uses 1,000 fixed stratified
questions (480 factoid, 320 list, 200 summary), free biomedical set-aware NLI,
and a low-temperature main answer judged only as `correct` or `incorrect`.
With 991 valid Claude labels per seed, P(True)-blind is best overall (AUROC
0.811/0.821), while SE is especially strong on list questions (discrete SE
0.845/0.881). Those are Phase-1 comparators for the current P(True)-Probe and
Claude-label Accuracy-Probe; direct blind P(True), not P(True)-10, is the
P(True)-Probe target. See
`archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md` for the full
result and `archehr_sebaseline/docs/archived_low_usability/README.md` for
retired material.

Phase 2 uses the full eligible BioASQ training13b corpus. The Phase-1
1,000-question cohort is already observed, so it is retained only on the
training side; an exact-question-grouped and type-stratified manifest reserves
unobserved questions for validation and final test. The active plan is
`PHASE2_PROBE_PLAN.md`; the fixed data procedure is in
`archehr_sebaseline/docs/phase2_bioasq_dataset_split.md`.

The completed Phase-2 run preserves that separation: parameters are trained on
the 3,144-question train split, the Probe type/layer/token is selected on 393
validation questions, and the resulting configuration is evaluated once on the
393-question test split (384 valid Claude labels for the Accuracy track).
The final P(True)-Probe is a train-even-threshold L2 logistic model at block
24/LT; it reaches 0.9026 AUROC against its P(True) target. The final
Accuracy-Probe is also block 24/LT and reaches 0.8058 AUROC / 0.8884 AP for
Claude incorrect, ahead of direct blind P(True) at 0.7900 / 0.8383. No complex
Probe is justified by this first pass; the next experiment is frozen-model
cross-dataset transfer, where P(True)-Probe is the primary hypothesis. See
`PHASE2_PROBE_PLAN.md` for target definitions, missing-label handling, and the
full UQ comparison.

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
PHASE2_PROBE_PLAN.md
ISAMBARD_COMMANDS.md
```

Their roles are:

- `codex_task_overview_archehr_uq.txt`: original project overview and broad
  requirements.
- `PHASE2_PROBE_PLAN.md`: active two-track probe plan, frozen BioASQ split,
  collection gates, and evaluation boundaries.
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
