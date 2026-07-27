# PubMedQA Frozen-Probe Transfer Protocol

Last updated: 2026-07-27

## Decision and status

The official 500-question PubMedQA PQA-L test subset is the first external
dataset for Phase-2 transfer. The local files passed structural checks:

- all 500 test PMIDs occur in `ori_pqal.json` and their labels agree;
- test labels are 276 `yes`, 169 `no`, and 55 `maybe`;
- every selected record has a question, contexts, `LONG_ANSWER`, and decision;
- there are no duplicate PMIDs or normalized questions; and
- there is no exact normalized-question overlap with local BioASQ training13b.

The active result is the completed abstract-context prompt version
`pubmedqa_context_explanation_v2`, which adds the official PubMedQA paper's
Appendix-C definitions of `yes`, `no`, and `maybe`. The previous abstract-context
v1 run remains the direct historical baseline. The question-only/no-context run
is no longer an active result because the official decision is defined from the
paper's experiments and results; its complete provenance and metrics are
retained in
`../../archive_unused/docs/historical_results/pubmedqa_no_context_transfer_20260721.md`.

The v1 and v2 context runs keep the same 500 IDs, PubMed abstracts, Gemma 3 12B
snapshot, answer contract, generation settings, frozen Probe bundle,
block-24/LT feature, and BioASQ-derived threshold. No PubMedQA answer, label,
hidden state, or metric fits or selects a Probe, calibrates a score, or changes
the threshold.

## Evidence and execution health

| Check | Context v1 | Context v2 |
| --- | --- | --- |
| Prompt version | `pubmedqa_context_explanation_v1` | `pubmedqa_context_explanation_v2` |
| Smoke job | `5748203`, `COMPLETED 0:0`, 1:00 | `5750742`, `COMPLETED 0:0`, 0:46 |
| Full job | `5748205`, `COMPLETED 0:0`, 37:43 | `5750745`, `COMPLETED 0:0`, 40:16 |
| Complete rows | 500/500 | 500/500 |
| Hidden states | `[500,4,3,3840]`, BF16 | `[500,4,3,3840]`, BF16 |
| Parsed leading labels | 500/500 | 500/500 |
| Full-`LONG_ANSWER` leakage | 0 | 0 |
| Frozen Probe bundle SHA-256 | `05c4dee4...d129f5a` | `05c4dee4...d129f5a` |

The complete local evidence snapshots are ignored generated artifacts rather
than source files:

```text
outputs/pubmedqa_context_frozen_probe_transfer_full500_seed31_20260722
analysis_outputs/pubmedqa_context_frozen_probe_transfer_full500_seed31_20260722
outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722
analysis_outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722
```

The v2 snapshot contains all prompts, answers, self-report/token UQ rows,
hidden-state indices, 500 per-question predictions, metrics, metadata, and
timing. The large hidden-state tensor remains on Isambard and was verified
there rather than duplicated locally.

## Complete answer-quality comparison

The official labels contain 276 `yes`, 169 `no`, and 55 `maybe` questions.

| Metric | Context v1 | Context v2 | Change |
| --- | ---: | ---: | ---: |
| Correct | 295/500 | **362/500** | **+67** |
| Strict accuracy | 59.0% | **72.4%** | **+13.4 pp** |
| Balanced accuracy | 0.5347 | **0.5652** | +0.0305 |
| Macro-F1 | 0.5211 | **0.5659** | +0.0448 |
| Weighted-F1 | 0.6367 | **0.7125** | +0.0758 |
| Predicted `yes/no/maybe` | 237 / 100 / 163 | **298 / 164 / 38** | closer to 276 / 169 / 55 |
| `yes` recall | 68.8% | **84.1%** | +15.2 pp |
| `no` recall | 47.9% | **72.8%** | +24.9 pp |
| `maybe` recall | **43.6%** | 12.7% | -30.9 pp |
| Error prevalence | 41.0% | **27.6%** | -13.4 pp |

The paired 500-question comparison is:

| Paired outcome | Count |
| --- | ---: |
| Correct in both | 274 |
| Correct only in v1 | 21 |
| Correct only in v2 | **88** |
| Wrong in both | 117 |
| Leading label changed | 134 |

The exact paired McNemar/binomial test gives `p=6.11e-11`. By official label,
v2 has a net gain of 42 correct `yes` questions and 42 correct `no` questions,
offset by a net loss of 17 correct `maybe` questions. The primary result is
therefore a large and statistically clear overall improvement, with reduced
minority-class `maybe` recall retained as a type-specific limitation rather
than treated as a reason to reject v2.

## UQ against official decision error

`incorrect=1` is the positive risk class. AP changes partly because error
prevalence falls from 0.410 in v1 to 0.276 in v2, so AUROC is the cleaner
cross-version ranking comparison.

| Score | v1 AUROC | v1 AP | v2 AUROC | v2 AP |
| --- | ---: | ---: | ---: | ---: |
| Frozen P(True)-Probe | 0.6630 | 0.5667 | **0.6839** | **0.4519** |
| Blind P(True) uncertainty | 0.6800 | 0.5625 | 0.6490 | 0.4210 |
| Verbalized-confidence uncertainty | **0.7933** | **0.6939** | 0.6402 | 0.4164 |
| Frozen Accuracy-Probe | 0.5960 | 0.5080 | 0.5901 | 0.3916 |
| Sequence NLL | 0.5716 | 0.4617 | 0.5400 | 0.3025 |
| Normalized NLL | 0.5668 | 0.4658 | 0.5424 | 0.3075 |
| Mean token entropy | 0.5668 | 0.4646 | 0.5423 | 0.3080 |
| Max token entropy | 0.5603 | 0.4431 | 0.5372 | 0.2914 |

The v2 P(True)-Probe is the best retained error-ranking score. Its AUROC rises
from 0.6630 to 0.6839 while the answer error count falls from 205 to 138.
Verbalized confidence falls from 0.7933 to 0.6402: the prompt improvement
removes many errors that were easy for self-reported confidence to identify,
leaving a smaller and harder error set. Token-only UQ remains weak.

### Calibration

| Probability-like score | v1 Brier | v2 Brier |
| --- | ---: | ---: |
| Frozen P(True)-Probe | 0.3253 | 0.3127 |
| Frozen Accuracy-Probe | 0.3463 | 0.3815 |
| Blind P(True) uncertainty | 0.4019 | 0.2659 |
| Verbalized-confidence uncertainty | 0.3206 | **0.2316** |
| Prevalence-only constant | **0.2419** | **0.1998** |

Every fixed score remains worse than the corresponding prevalence-only Brier
baseline. These outputs are ranking scores, not calibrated PubMedQA error
probabilities; target calibration remains intentionally forbidden.

## Frozen target and continuous P(True) fidelity

The unchanged BioASQ threshold marks 201/500 v1 and 202/500 v2 blind P(True)
uncertainties as high.

| Score | v1 frozen-target AUROC/AP | v2 frozen-target AUROC/AP |
| --- | ---: | ---: |
| Frozen P(True)-Probe | **0.6121 / 0.5354** | 0.5899 / 0.5140 |
| Frozen Accuracy-Probe | 0.5832 / 0.5155 | **0.6290 / 0.5499** |
| Verbalized-confidence uncertainty | 0.6663 / 0.5357 | 0.6609 / 0.5458 |
| Sequence NLL | 0.5293 / 0.4298 | 0.5340 / 0.4215 |
| Normalized NLL | 0.5291 / 0.4458 | 0.5324 / 0.4127 |
| Mean token entropy | 0.5307 / 0.4452 | 0.5416 / 0.4253 |
| Max token entropy | 0.5066 / 0.4054 | 0.5436 / 0.4293 |

Against continuous blind P(True) uncertainty, P(True)-Probe
Spearman correlation changes from 0.2395 to 0.1559 and Accuracy-Probe from
0.1787 to 0.2440. Neither frozen Probe preserves a strong numerical mapping to
the self-report score across this dataset/prompt shift. The v2 P(True)-Probe's
0.6839 correctness AUROC is therefore cross-target error-ranking usefulness,
not preserved source-target fidelity.

## Result interpretation

The Appendix-C prompt improvement is successful overall and v2 is the current
main PubMedQA context result. It increases accuracy, balanced accuracy,
Macro-F1, Weighted-F1, and the match between predicted and official label
counts without changing the model, data, generation settings, Probes,
threshold, or evaluator.

The remaining limitation is specific rather than global: v2 under-predicts the
minority `maybe` label. The 18 official `maybe` questions that v1 classified
correctly and v2 changed to `yes` or `no` remain documented as part of that
limitation, but no dedicated manual error review or further PubMedQA prompt
tuning is planned.

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

The active PubMedQA result supplies the official PubMed abstract context to
match the evidence from which the dataset decision was annotated. The model
must:

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

The historical context baseline uses prompt version
`pubmedqa_context_explanation_v1`. It
places the official `CONTEXTS` paragraphs under `PubMed abstract context` and
requires the decision/explanation to use only that context. `LONG_ANSWER` and
the official decision remain evaluation-only. All 500 examples have context;
local/remote preflight found zero `LONG_ANSWER` exposure, and the longest
rendered context prompt is 3,147 characters.

The completed current run uses `pubmedqa_context_explanation_v2`. It changes
only the prompt by adding the official Appendix-C `yes`/`no`/`maybe` criteria
summarized above; all other inputs, model settings, frozen-Probe settings, and
evaluation contracts remain unchanged. The v1 result remains the direct
historical baseline.

## Other UQ comparisons

The same single answer supplies:

- blind P(True) uncertainty;
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
