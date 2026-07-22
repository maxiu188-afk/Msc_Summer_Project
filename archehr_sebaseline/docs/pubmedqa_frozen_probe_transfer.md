# PubMedQA Frozen-Probe Transfer Protocol

Last updated: 2026-07-22

## Decision and status

The official 500-question PubMedQA PQA-L test subset is the first external
dataset for Phase-2 transfer. The local files passed structural checks:

- all 500 test PMIDs occur in `ori_pqal.json` and their labels agree;
- test labels are 276 `yes`, 169 `no`, and 55 `maybe`;
- every selected record has a question, contexts, `LONG_ANSWER`, and decision;
- there are no duplicate PMIDs or normalized questions; and
- there is no exact normalized-question overlap with local BioASQ training13b.

Isambard smoke job `5734703` completed `0:0` in 39 seconds with all three answer
contracts, P(True) rows, hidden-state rows, and transfer outputs present. The
no-context full job `5739322` then completed 500/500. Its strict
decision accuracy was only 25% because the model emitted `maybe` 373 times,
while the official labels contain only 55 `maybe` questions. This distribution
shift motivated a second, separately named **context-conditioned** run using
the provided PubMed abstract `CONTEXTS`. Context smoke `5748203` completed
`0:0` in 1 minute; dependent full job `5748205` completed `0:0` in 37:43.

The context condition changes only the information/prompt condition. It keeps
the same 500 IDs, model snapshot, answer contract, generation settings, frozen
Probe bundle, feature position, and BioASQ-derived threshold. The corresponding
self-report scores are context-conditioned P(True)/confidence and must not be
silently labelled blind/no-context P(True).

## Completed transfer results

Both full conditions contain 500/500 examples, prompts, generations,
self-report rows, hidden-state index rows, and evaluation rows. Context hidden
states have shape `[500,4,3,3840]`; all 500 prompts use
`pubmedqa_context_explanation_v1`, contain every supplied context paragraph,
and have zero full-`LONG_ANSWER` leakage. Both conditions use the identical
frozen Probe bundle SHA-256
`05c4dee461cdf789af5fc1ca45ef223dbf8fc4eab54d4724e57c59993d129f5a`.

### Answer behaviour

| Condition | Accuracy | Predicted yes/no/maybe | Yes recall | No recall | Maybe recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| No context | 125/500 (25.0%) | 123 / 4 / 373 | 30.1% | 0.6% | 74.5% |
| PubMed context | 295/500 (59.0%) | 237 / 100 / 163 | 68.8% | 47.9% | 43.6% |

On the paired 500 questions, context changed 197 wrong answers to correct and
27 correct answers to wrong; 98 were correct in both conditions and 178 wrong
in both. The paired correctness improvement is large (exact McNemar/binomial
test, `p=4.41e-33`). Context therefore addresses much of the no-context
distribution failure, especially for `yes` and `no`, but it still predicts
`maybe` 163 times against only 55 official `maybe` labels and reduces recall on
that minority label.

### UQ against official decision error

`incorrect=1` is the positive risk class. The error prevalence changes from
75% without context to 41% with context, so AP values are interpretable within
each condition but should not be compared as prevalence-free improvements.

| Score | No-context AUROC | No-context AP | Context AUROC | Context AP |
| --- | ---: | ---: | ---: | ---: |
| Verbalized-confidence uncertainty | **0.7757** | **0.8766** | **0.7933** | **0.6939** |
| Direct/context-conditioned P(True) uncertainty | 0.7030 | 0.8431 | 0.6800 | 0.5625 |
| Frozen P(True)-Probe | 0.4188 | 0.7001 | 0.6630 | 0.5667 |
| Frozen Accuracy-Probe | 0.5396 | 0.7718 | 0.5960 | 0.5080 |
| Sequence NLL | 0.5287 | 0.7715 | 0.5716 | 0.4617 |
| Normalized NLL | 0.5097 | 0.7634 | 0.5668 | 0.4658 |
| Mean token entropy | 0.5012 | 0.7561 | 0.5668 | 0.4646 |
| Max token entropy | 0.5053 | 0.7534 | 0.5603 | 0.4431 |

Verbalized confidence is the strongest error-ranking score in both prompt
conditions. Context makes the frozen P(True)-Probe useful for correctness
ranking (`0.6630` rather than the no-context inverse-direction `0.4188`) and it
outperforms the frozen Accuracy-Probe (`0.5960`). The Accuracy-Probe therefore
shows only weak cross-dataset transfer despite its strong in-domain BioASQ
result. Token-only scores improve above chance with context but remain weak.

None of the probability-like scores is calibrated to PubMedQA error without
target calibration, which is intentionally forbidden here. In the context
condition their Brier scores are 0.3206 (verbal), 0.4019 (P(True)), 0.3253
(P(True)-Probe), and 0.3463 (Accuracy-Probe), all worse than the prevalence-only
constant baseline `0.41 × 0.59 = 0.2419`. Treat them as fixed ranking scores,
not calibrated PubMedQA probabilities.

### Frozen Probe fidelity to the P(True)-threshold target

The unchanged BioASQ threshold marks 145/500 no-context and 201/500 context
P(True) uncertainties as high. Context changes the self-report information
condition, so this is a frozen-threshold teacher target under context shift,
not the original blind/no-context target.

| Score | No-context AUROC/AP | Context AUROC/AP |
| --- | ---: | ---: |
| Verbalized-confidence uncertainty | 0.6829 / 0.3977 | 0.6663 / 0.5357 |
| Frozen P(True)-Probe | **0.5796 / 0.3382** | **0.6121 / 0.5354** |
| Frozen Accuracy-Probe | 0.5520 / 0.3313 | 0.5832 / 0.5155 |
| Sequence NLL | 0.5463 / 0.3342 | 0.5293 / 0.4298 |
| Normalized NLL | 0.5271 / 0.3246 | 0.5291 / 0.4458 |
| Mean token entropy | 0.5164 / 0.3138 | 0.5307 / 0.4452 |
| Max token entropy | 0.4910 / 0.2961 | 0.5066 / 0.4054 |

The P(True)-Probe retains only modest fidelity to its teacher target across the
dataset shift (`0.6121` AUROC with context versus `0.9026` on held-out BioASQ).
Its stronger `0.6630` association with PubMedQA decision error should therefore
be described as cross-target usefulness, not preservation of the original
BioASQ mapping.

## Interpretation and next prompt experiment

The context result is consistent with the expected transfer ordering:
P(True)-Probe captures a source-model uncertainty signal and ranks PubMedQA
errors better than the BioASQ/Claude-trained Accuracy-Probe, whose correctness
mapping is more dataset- and answer-contract-specific. This is supportive
evidence, not a claim that P(True)-Probe must dominate Accuracy-Probe on every
external dataset.

All fixed UQ methods remain weaker on PubMedQA than their principal BioASQ
comparators. The clearest task-level failure is excessive `maybe` prediction:
context reduces it from 373 to 163 but the official test contains only 55
`maybe` labels. This is the leading hypothesis for why PubMedQA correctness and
UQ transfer remain limited, but the two completed conditions do not by
themselves prove that output-label bias causes the AUROC reduction.

The next experiment will improve the **generation prompt**, not either Probe:

1. Freeze the two completed official-test conditions and do not use their
   per-example labels or UQ outcomes to choose another prompt.
2. Use the other 500 PQA-L records in `ori_pqal.json` (IDs absent from
   `test_ground_truth.json`) as the prompt-development pool. Create a
   deterministic label-stratified development/holdout split before generation.
3. Compare a small, predeclared prompt family that clarifies that `maybe` is
   reserved for genuinely inconclusive or mixed abstract evidence and asks for
   the study conclusion rather than generic biomedical plausibility.
4. Select the prompt using development-set strict decision accuracy and label
   distribution diagnostics, without reading Probe/UQ performance.
5. Freeze the chosen prompt, run the held-out portion once, and apply the same
   frozen block-24/LT Probes with no fitting, recalibration, threshold tuning,
   or feature selection.

Because the official 500 test outcomes have already informed the motivation,
any further run on those same IDs is adaptive/exploratory and must not be
presented as a fresh untouched confirmation. A clean confirmatory claim must
come from the pre-frozen non-test holdout or another independent dataset.

## What “transfer” means

This experiment compares two Probe models across datasets while keeping the
source language model fixed as the exact Gemma 3 12B snapshot used for BioASQ.
It is not a transfer across language-model backbones. Hidden dimensions and
representational spaces are not aligned across arbitrary base models, so
applying the same coefficient vector to a different backbone would be invalid.

Both frozen Probes use only block 24 / final answer token (`LT`):

| Frozen Probe | Original BioASQ supervision | PubMedQA original-target test |
| --- | --- | --- |
| P(True)-Probe | direct blind P(True) uncertainty at or above the BioASQ-train even threshold | same blind P(True) target with the unchanged BioASQ threshold |
| Accuracy-Probe | Claude `incorrect=1` for the BioASQ main answer | official PubMedQA decision-label mismatch `incorrect=1` |

Each Probe is also scored against the other target. This creates a fixed 2x2
comparison that separates original-target transfer from cross-target utility.
No PubMedQA answer, label, P(True), hidden state, or metric may fit a Probe,
change the threshold, select a layer/token, recalibrate a score, or choose a
model.

## Frozen-weight provenance repair

The completed first-pass analysis saved configuration, metrics, and test
predictions but omitted fitted scaler and coefficient arrays. The one-time
`materialize_frozen_phase2_probes.py` command reconstructs only the two already
selected block-24/LT logistic models from the original BioASQ **train**
artifacts. It does not read BioASQ validation/test features or any PubMedQA
data, and it does not repeat feature or threshold selection.

```bash
python scripts/materialize_frozen_phase2_probes.py \
  --bioasq-run-dir outputs/bioasq_phase2_gemma3_12b_single_answer_seed31 \
  --probe-config analysis_outputs/bioasq_phase2_linear_probes_seed31/probe_run_config.json \
  --output-dir analysis_outputs/bioasq_phase2_frozen_probe_bundle_seed31
```

The bundle consists of JSON metadata plus non-executable compressed NumPy
arrays and includes source SHA-256 values. Local replay reproduced both saved
BioASQ test score vectors with maximum absolute difference `5.551e-17`.

## Target answer and explanation contract

PubMedQA generation is no-evidence to match the BioASQ Phase-2 inference
condition. The model sees the question only and must:

1. start with exactly `yes`, `no`, or `maybe`; and
2. provide a concise one-to-three-sentence explanation.

The official decision and `LONG_ANSWER` are evaluation-only fields. They never
appear in the generation prompt, P(True) prompt, hidden-state input, or Probe
input. The leading decision is evaluated deterministically against the official
label. A missing/invalid leading label counts as incorrect and is reported as a
prompt-contract failure.

The generated explanation and stored `LONG_ANSWER` are preserved for a later,
separate Claude alignment judgment. The project does not currently assume that
`LONG_ANSWER` is a manually curated high-quality explanation. Before treating
Claude alignment as factual-quality supervision, inspect result behaviour and
manually review a fixed subset. Claude outputs must remain separate artifacts
and cannot alter this transfer experiment.

The context rerun uses prompt version `pubmedqa_context_explanation_v1`. It
places the official `CONTEXTS` paragraphs under `PubMed abstract context` and
requires the decision/explanation to use only that context. `LONG_ANSWER` and
the official decision remain evaluation-only. All 500 examples have context;
local/remote preflight found zero `LONG_ANSWER` exposure, and the longest
rendered context prompt is 3,147 characters.

## Other UQ comparisons

The same single answer supplies:

- direct blind P(True) uncertainty in the no-context condition, or the
  separately named context-conditioned P(True) uncertainty;
- verbalized-confidence uncertainty;
- sequence NLL and normalized NLL;
- mean token entropy and max token entropy; and
- the two frozen Probe scores.

Every score is evaluated against both fixed targets using the same eligible
questions. Report AUROC and average precision for every score; report Brier
only for scores natively in `[0,1]`. The evaluator also reports each Probe's
association with continuous direct P(True) uncertainty. There is no ten-sample
generation, NLI clustering, Semantic Entropy, P(True)-10, or sample
disagreement in this run.

## Isambard execution

Run a small deterministic smoke first:

```bash
MAX_EXAMPLES=3 \
PUBMEDQA_DATA_PATH=/path/to/ori_pqal.json \
PUBMEDQA_GROUND_TRUTH_PATH=/path/to/test_ground_truth.json \
FROZEN_PROBE_BUNDLE=/path/to/frozen_probe_bundle.json \
sbatch scripts/run_pubmedqa_frozen_probe_transfer_isambard.sbatch
```

After checking its answer contract, one-answer/P(True) counts, hidden tensor,
and evaluation outputs, submit the full 500-question job without
`MAX_EXAMPLES`. Required result files are:

```text
examples.jsonl
prompts.jsonl
best_generations.jsonl
uq_baselines/self_report_examples.csv
uq_baselines/single_answer_example_uq.csv
hidden_states/phase2_hidden_states_test.pt
hidden_states/phase2_hidden_states_test_index.jsonl
pubmedqa_transfer_run_metadata.json
transfer_metrics.csv
continuous_p_true_association.csv
transfer_predictions.csv
transfer_summary.json
```

The first eight are collection artifacts; the final four are written under the
configured analysis directory.
