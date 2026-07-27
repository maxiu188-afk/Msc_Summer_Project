# FinalProject Code Workspace

This folder contains the code-side workspace for the lightweight uncertainty
quantification project on grounded clinical question answering.

Current status:

```text
ArchEHR-QA SE baseline: implemented and tested as an engineering baseline
ArchEHR-QA final evaluation/training target: paused because no usable gold labels
BioASQ Task B: prior summary results archived as low-usability diagnostics
Current compute host: Isambard (Runpod was a completed temporary recovery path)
Active direction: study when SE or blind P(True) performs better; the paired
  BioASQ-summary answer-length intervention is complete
Latest validated run: Phase-2 3,930-question single-answer collection and
  two frozen linear hidden-state Probes on Gemma 3 12B
Current Phase-2 tracks: P(True)-Probe (direct P(True) fidelity) and
  Claude-label Accuracy-Probe (answer correctness)
Current result: train-only fitting, validation-only feature selection, and one
  held-out test evaluation are complete; PubMedQA Appendix-C context v2 reaches
  72.4% decision accuracy and P(True)-Probe reaches 0.6839 error AUROC
Current Phase-2 status: complete; the efficiency benchmark, paired bootstrap,
  and one validation-fitted two-Probe fusion diagnostic are finished
Current follow-up result: shortening from 82.88 to 52.11 mean words did not
  recover SE relative to blind P(True); paired length-effect CI includes zero
Current mechanism result: on 93 fixed-generation long-summary questions,
  Claude reclustering raises SE AUROC from 0.595 to 0.781, strongly
  implicating NLI clustering difficulty; Claude is diagnostic, not deployable
Current model-scale result: on 196 common valid summary questions,
  P(True)-minus-SE AUROC changes from +0.243 at 12B to +0.008 at 4B and
  -0.096 at 1B; the 1B-versus-4B incremental change is not resolved
Current 1B feasibility result: factoid accuracy 8/50 and list accuracy 3/50;
  neither remaining full type cohort is recommended
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
main answer exists. Six-question smoke job `5761273` completed and passed its
health check. The first full job `5761275` failed before model loading because
its node exposed a nonexistent local temporary directory; the batch script now
uses a job-specific `$SCRATCHDIR` temporary directory. Replacement full job
`5773786` completed all 384 questions in `03:31:33` and passed its health
check. One Probe takes about `60.5 ms/question`, both jointly `60.54 ms`,
blind P(True) `154.4 ms`, ten-sample normalized NLL `30.55 s`, and SE
`32.75 s`. Probes are about `2.55x` faster than blind P(True) and over `500x`
faster than the sampling methods. Blind P(True) has zero freely generated
tokens but still scores fixed `True` and `False` continuations in two LM calls.
See `archehr_sebaseline/docs/phase2_uq_efficiency_benchmark.md`.

The Phase-1 factoid/list/summary samples give a natural descriptive
answer-length/UQ contrast. Mean sampled-answer lengths are about `8/35/111`
generated tokens. Blind-P(True)-minus-discrete-SE AUROC is about
`+0.075`, `0.000/-0.029`, and `+0.243/+0.231` across the two seeds:
SE is competitive on structured lists but falls far behind on long summaries.
Because type, answer structure, prompt, and NLI rule also vary, this is an
operating-regime observation rather than a causal length effect.

Phase-2 statistical closure is also complete. Accuracy-Probe exceeds blind
P(True) by `0.01584` AUROC on the 384 test questions, but the 20,000-resample
paired-bootstrap 95% CI is `[-0.03890, 0.07173]`, so the improvement is not
statistically resolved. A validation-fitted two-Probe fusion reaches test
AUROC `0.8125` and AP `0.8871`, only `0.0067` AUROC above Accuracy-Probe while
slightly lowering AP. It remains an exploratory diagnostic rather than a new
main model. See
`archehr_sebaseline/docs/phase2_probe_completion_statistics.md`.

The paired BioASQ-summary answer-length intervention is complete. The short
prompt reduced mean main-answer length from 82.88 to 52.11 words and from 3.25
to 2.00 sentences, with 123/123 main answers satisfying the one-or-two-sentence
instruction. On 121 questions with valid labels in both conditions, accuracy
was unchanged at 56/121. Blind P(True) AUROC changed from 0.7948 to 0.8040,
while discrete SE changed from 0.5782 to 0.5468. The pre-declared relative
length effect was +0.0405 with paired-bootstrap 95% CI
`[-0.0713, 0.1531]`; shortening therefore did not establish an SE recovery
relative to blind P(True). See
`archehr_sebaseline/docs/summary_length_intervention.md`.

The fixed-generation Claude-versus-NLI mechanism diagnostic is complete on
93/96 long-summary questions: Claude-clustered SE reaches 0.781 AUROC versus
0.595 under the accepted NLI clusters, strongly implicating long-answer NLI
over-merging. Claude remains an expensive diagnostic rather than a deployment
proposal.

The exact-cohort Gemma 3 4B/12B comparison is also complete. On 990 paired
questions, P(True)-minus-SE AUROC changes from +0.047 at 12B to -0.009 at 4B;
the paired change is -0.056 with 95% CI [-0.103,-0.010]. The strongest scale
effect is on summary, where P(True)'s 12B advantage disappears at 4B. No
model-scale Probe is trained.

The staged Gemma 3 1B generation is complete. Initial smoke `5802163` passed its
cohort and CUDA preflights but failed before generation because the 1B weights
were absent from the shared cache; dependent staged job `5802164` never ran and
was cancelled. The fixed 1B revision was then downloaded and checksum-verified.
Replacement smoke `5807823` still failed because the shared generator
incorrectly treated the text-only 1B checkpoint as multimodal. That route now
uses the saved model configuration: 1B selects
`AutoTokenizer`/`AutoModelForCausalLM`, while 4B/12B retain their existing
multimodal path. Dependent job `5807824` never ran and was cancelled. Second
replacement smoke `5808905` completed `0:0` in `00:02:06`; staged job
`5808906`, held by `afterok:5808905`, then completed `0:0` in `02:06:35`.
Both passed Level-4 health and exact frozen-cohort checks. The staged output
contains all 200 summary questions aligned to the existing 4B/12B cohort, plus
frozen 50-question factoid and list subsets. The 300-request protocol-matched
Claude correctness batch `msgbatch_01LCGoC6Mh4EBNKHAq3XEB5U` completed with
300/300 API successes and 298 valid labels. Its single protocol-matched
64-token retry `msgbatch_01AWNB4xt1h86wyfRdm6fq2F` raised this to 299/300;
the one remaining invalid summary response is excluded without another retry.
Factoid accuracy is 8/50 (16%) and list accuracy is 3/50 (6%), so neither full
type expansion is recommended. On 196 summary questions with valid labels for
all three models, 1B accuracy is 12.2%. Blind P(True), discrete SE, and
normalized-NLL AUROC are 0.501, 0.597, and 0.789. P(True)-minus-SE is
`-0.096 [-0.244,+0.060]` at 1B versus `+0.008 [-0.081,+0.096]` at 4B and
`+0.243 [+0.171,+0.312]` at 12B. The point estimate continues downward, but
the 1B-minus-4B change `-0.104 [-0.267,+0.059]` is not statistically resolved.
No further PubMedQA error analysis or prompt tuning is planned. The canonical
design and remaining boundaries are in `PHASE2_PROBE_PLAN.md`.

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
  collection gates, completed results, and the post-Simpson SE/P(True)
  operating-regime follow-up.
- `ISAMBARD_COMMANDS.md`: common local and Isambard operational commands;
  historical launch procedures are kept only under `archive_unused/`.
- `SIMPSON_MEETING_PHASE2_RESULTS_2026-07-23_FINAL.md`: frozen meeting brief
  covering the results presented at the completed supervisor meeting. Later
  research decisions are owned by `PHASE2_PROBE_PLAN.md`.

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
