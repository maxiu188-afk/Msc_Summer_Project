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
  held-out test evaluation are complete; PubMedQA Appendix-C context v2 reaches
  72.4% decision accuracy and P(True)-Probe reaches 0.6839 error AUROC
Current follow-up: an incremental UQ-efficiency smoke/full pair is queued on
  the 384 valid-labelled BioASQ test questions; no efficiency result exists yet
```

ArchEHR-QA remains useful for testing grounded long-form generation, citation
parsing, answer Semantic Entropy, and citation uncertainty. It should not be the
main SEP training or final SE-evaluation dataset unless additional gold
answer-quality/evidence labels become available.

PubMedQA remains unsuitable as the main research target because its
yes/no/maybe task differs from BioASQ free-form QA. Its official 500-question
PQA-L test subset is now deliberately used as an external frozen-Probe transfer
test, with an explained decision output and no target-dataset fitting.

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
result and `archive_unused/` for retired material.

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
Probe is justified by this first pass. The completed frozen-model cross-dataset
transfer used P(True)-Probe as the primary hypothesis. The current PubMedQA
Appendix-C context v2 result reaches 72.4% strict accuracy, up from 59.0% for
context v1, with Macro-F1 improving from 0.5211 to 0.5659. P(True)-Probe is the
strongest v2 error-ranking score at 0.6839 AUROC / 0.4519 AP, followed by
blind P(True) at 0.6490 / 0.4210 and verbalized confidence at
0.6402 / 0.4164. The question-only PubMedQA result is archived because it
omitted the article evidence that defines the official decision. See
`PHASE2_PROBE_PLAN.md` and
`archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md` for the frozen
2x2 Probe-target comparison, complete v1/v2 tables, explanation contract, and
other UQ scores.

The current in-domain follow-up measures incremental UQ cost after the saved
main answer exists. Smoke job `5761273` and dependent full job `5761275`
compare the two block-24/LT Probes (one shared hidden-state replay) with blind
P(True), ten-sample normalized NLL, discrete SE, and cluster count on all 384
valid-labelled Phase-2 test questions. The jobs are queued, so no latency,
GPU-time, token, throughput, or new AUROC value is yet a result. See
`archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`.

## Active Naming Policy

New BioASQ-main-track artifacts use BioASQ/`bioasq_se` names rather than new
`archehr` prefixes. This is forward-only: the `archehr_sebaseline/` package,
existing imports and historical ArchEHR-QA artifact names remain unchanged to
preserve working paths and provenance.

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
  server_results/            Active server-result snapshots
  archive_unused/            Superseded docs, data inventories, and old results
  .agents/                   Local Codex/agent state
  .git/                      Repository metadata
```

## Root-Level Planning Files

These root-level files are intentionally kept at the workspace level:

```text
PHASE2_PROBE_PLAN.md
ISAMBARD_COMMANDS.md
SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md
```

Their roles are:

- `PHASE2_PROBE_PLAN.md`: active two-track probe plan, frozen BioASQ split,
  collection gates, and evaluation boundaries.
- `ISAMBARD_COMMANDS.md`: common local and Isambard operational commands;
  historical launch procedures are kept only under `archive_unused/`.
- `SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`: current meeting brief
  covering the completed Phase-2 BioASQ results, PubMedQA v2 transfer, and
  proposed remaining research direction.

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

Use `archehr_sebaseline/docs/README.md` as the active documentation index.
Superseded plans, handoff notes, legacy runbooks, and historical result reports
are consolidated under `archive_unused/`.

## Reference Code

The directory:

```text
semantic_uncertainty/
```

contains reference code for the original Semantic Entropy project. Treat it as
reference material, not as the active codebase.

## Archived Material

Material that is not part of the current workflow belongs under:

```text
archive_unused/
```

See `archive_unused/README.md` before moving obsolete documents, generated data,
or results. The local `literature/` directory is separate source material and
was not changed by this cleanup.

## Server Packages

Server upload archives may appear at the root, for example:

```text
bioasq_se_runpod_*.tar.gz
```

These are transport artifacts, not source code. Regenerate them from
`archehr_sebaseline/` when the code changes. Existing legacy archive names are
not renamed retroactively.
