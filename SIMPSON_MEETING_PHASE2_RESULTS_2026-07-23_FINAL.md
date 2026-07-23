# Phase-2 Results Summary for Simpson Meeting

Date: 2026-07-23

## 1. Executive summary

Phase 2 has completed the main in-domain BioASQ hidden-state Probe study and
the first frozen-Probe transfer evaluation on PubMedQA.

- The frozen **P(True)-Probe** successfully recovers its BioASQ direct
  P(True)-derived target in-domain: held-out AUROC **0.9026**.
- The separately trained **Accuracy-Probe** predicts BioASQ answers judged
  incorrect by Claude: held-out AUROC **0.8058**.
- These are different tasks. P(True)-Probe measures fidelity to direct
  P(True), whereas Accuracy-Probe targets answer correctness.
- PubMedQA Appendix-C context v2 is complete. It reaches strict decision
  accuracy **72.4%**, Macro-F1 **0.5659**, and 138 errors on 500 questions.
- Frozen P(True)-Probe is the strongest v2 error-ranking score at AUROC
  **0.6839** / AP **0.4519**. The principal limitation is minority-class
  `maybe` recall at 12.7%; this is retained as error analysis rather than used
  to reject the clear overall improvement.

## 2. Why Phase 2 followed Phase 1

The final Phase-1 BioASQ experiment used 1,000 no-evidence questions. Blind
P(True) was the strongest overall UQ method across both generation seeds:

| Method | Seed 31 AUROC | Seed 47 AUROC |
| --- | ---: | ---: |
| Blind P(True) | **0.811** | **0.821** |
| Discrete Semantic Entropy | 0.763 | 0.779 |
| Cluster count | 0.765 | 0.781 |
| Predictive entropy / normalized NLL | 0.729 | 0.745 |

This motivated the Phase-2 question: can a lightweight Probe infer the useful
P(True) signal directly from the model's hidden states, avoiding the additional
self-evaluation generation?

## 3. Phase-2 experimental design

### 3.1 Dataset and leakage controls

The active corpus contains all 3,930 eligible factoid, list, and summary
questions in BioASQ training13b. Yes/no questions remain outside the active
free-form protocol.

| Split | Factoid | List | Summary | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,280 | 837 | 1,027 | 3,144 |
| Validation | 160 | 105 | 128 | 393 |
| Test | 160 | 105 | 128 | 393 |

Controls:

- All 1,000 Phase-1 reference questions are train-side only.
- Exact normalized duplicate questions cannot cross splits.
- Model/scaler fitting uses train only.
- Layer and token-position selection uses validation only.
- The frozen configuration is evaluated once on test.

### 3.2 Hidden-state contract

- Source model: Gemma 3 12B.
- Transformer blocks: `24`, `32`, `40`, and `48`.
- Token positions: `TBG` (last prompt token), `SLT` (penultimate answer token),
  and `LT` (last answer token).
- Hidden dimension: 3,840.
- Artifact layout: `[example, 4 blocks, 3 positions, 3840]`.
- Each candidate Probe receives one 3,840-dimensional vector; there is no
  layer concatenation, neural Probe, tree model, or feature search.

The full Isambard collection completed 3,930 questions in 3:44:58 and passed
the artifact health checks.

### 3.3 Basis for layer and token-position selection

Layer and token position were selected from validation results only. The
candidate comparison does **not** show a large, universal advantage for one
token position across every layer and target.

P(True)-Probe validation AUROC:

| Block | TBG | SLT | LT |
| --- | ---: | ---: | ---: |
| 24 | 0.8348 | 0.8578 | **0.8754** |
| 32 | 0.8125 | 0.8423 | **0.8630** |
| 40 | 0.8311 | 0.8129 | **0.8319** |
| 48 | 0.8074 | 0.7924 | **0.8400** |

Accuracy-Probe validation AUROC:

| Block | TBG | SLT | LT |
| --- | ---: | ---: | ---: |
| 24 | 0.8294 | 0.7337 | **0.8458** |
| 32 | 0.8107 | 0.7568 | **0.8394** |
| 40 | 0.8029 | 0.7764 | **0.8118** |
| 48 | **0.8367** | 0.7283 | 0.8270 |

Interpretation:

- For P(True)-fidelity, LT is best or effectively tied with TBG at every tested
  block, but its advantage is modest rather than dramatic. At block 24, LT is
  only 0.0177 AUROC above SLT and 0.0406 above TBG.
- For Accuracy-Probe, LT and TBG are generally close. At the selected block 24,
  LT exceeds TBG by only 0.0164 AUROC. SLT is consistently weaker in this
  track.
- The winning block-24/LT configuration is therefore a validation-based
  operational choice, not evidence that the last answer token is uniquely or
  causally superior.
- SLT has fewer valid rows because very short answers cannot supply a
  penultimate answer-token position: 389 rather than 393 validation rows for
  P(True)-Probe, and 380 rather than 384 for Accuracy-Probe. Its values are not
  based on exactly the same sample as TBG/LT.
- No paired bootstrap or other candidate-to-candidate significance test was
  performed. The defensible wording is **"no clear large difference between
  TBG and LT, with LT selected by the frozen validation rule"**, rather than a
  formal claim that token positions have no statistically significant
  difference.

## 4. In-domain BioASQ results

### 4.1 P(True)-Probe

Target: a train-derived even split of
`p_true_blind_uncertainty = 1 - P(True)` into low/high uncertainty.

Frozen model: L2 logistic regression at block 24 / LT.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| AUROC | 0.8754 | **0.9026** |
| Average precision | - | **0.8948** |
| Brier score | - | **0.1414** |

This is a **P(True)-fidelity result**, not an answer-correctness result.

#### P(True) target and linear-model variants

Four P(True)-fidelity formulations were evaluated. Each formulation selected
its own block/token position using validation data and was then evaluated on
the held-out BioASQ test split.

| Formulation | Target and model | Validation-selected feature | Held-out test result |
| --- | --- | --- | --- |
| Even-split hard threshold | Binary low/high uncertainty; L2 logistic regression | block 24 / LT | **AUROC 0.9026**; AP 0.8948; Brier 0.1414 |
| Minimum-within-variance threshold | Binary low/high uncertainty; L2 logistic regression | block 40 / LT | AUROC 0.7677; AP 0.3629; Brier 0.1436 |
| ElasticNet continuous regression | Continuous blind P(True) uncertainty | block 24 / SLT; 391 valid test rows | **Spearman 0.6057**; MAE 0.2189 |
| Ridge continuous regression | Continuous blind P(True) uncertainty | block 24 / TBG | Spearman 0.2146; MAE 0.4378 |

The two hard-threshold variants are classification Probes and are compared
with AUROC/AP/Brier. Ridge and ElasticNet predict the continuous uncertainty
score and are compared with Spearman correlation and MAE. A continuous
P(True) target has no native classification AUROC unless it is changed into a
separate binary target. The frozen P(True)-Probe used elsewhere in this report
remains the even-split hard-threshold model.

The underlying evidence remains available in
`analysis_outputs/bioasq_phase2_linear_probes_seed31/candidate_metrics.csv`.

#### Type-stratified fidelity

| Question type | Test AUROC | Test AP | High-target prevalence |
| --- | ---: | ---: | ---: |
| Factoid | 0.8746 | 0.8997 | 101/160 (63.1%) |
| List | 0.8110 | 0.9347 | 80/105 (76.2%) |
| Summary | 0.6831 | 0.3185 | 10/128 (7.8%) |
| Pooled | **0.9026** | **0.8948** | 191/393 (48.6%) |

The pooled AUROC is higher than every within-type AUROC, while target
prevalence differs sharply by question type. The overall score should therefore
be reported together with the stratified results: part of the pooled
separability may come from between-type differences, and summary-question
fidelity is substantially weaker.

### 4.2 Accuracy-Probe

Target: Claude binary judgment of the same low-temperature answer, with
`incorrect=1`.

Claude produced 3,858 valid labels from 3,930 questions. The remaining 72
blank/invalid labels were excluded only from Accuracy-Probe. Valid
train/validation/test sizes are 3,090/384/384.

Frozen model: L2 logistic regression at block 24 / LT.

| Metric | Held-out test |
| --- | ---: |
| AUROC | **0.8058** |
| Average precision | **0.8884** |
| Brier score | **0.2162** |

Type-stratified AUROC is 0.7836 for factoid, 0.7677 for list, and 0.7897 for
summary. List AP is high (0.9650), but 93/103 list answers are labelled
incorrect, so AUROC is the more informative ranking comparison for that subset.

### 4.3 Correctness comparison on the same 384 test questions

| Score | AUROC | AP | Interpretation |
| --- | ---: | ---: | --- |
| Accuracy-Probe | **0.8058** | **0.8884** | Dedicated correctness target |
| Direct blind P(True) uncertainty | 0.7900 | 0.8383 | Strongest non-Probe score |
| P(True)-Probe | 0.7452 | 0.8357 | Secondary cross-target use only |
| ElasticNet continuous P(True)-Probe | 0.6532 | 0.7945 | Continuous P(True) fit; 382 SLT-valid rows |
| Verbalized-confidence uncertainty | 0.6662 | 0.7426 | Discrete self-report score |
| Mean token entropy | 0.5583 | 0.7096 | Weak single-answer statistic |
| Normalized NLL | 0.5546 | 0.7077 | Weak single-answer statistic |
| Max token entropy | 0.5357 | 0.6737 | Weak single-answer statistic |
| Sequence NLL | 0.5304 | 0.6687 | Weak single-answer statistic |

Main interpretation:

- Accuracy-Probe contains correctness information beyond the P(True)-Probe.
- On the exact 382 rows available to ElasticNet, hard-even P(True)-Probe
  reaches 0.7456 AUROC versus ElasticNet 0.6532. Continuous regression is
  therefore weaker, not stronger, for correctness-UQ ranking.
- P(True)-Probe should not be judged primarily by Claude correctness because
  it was trained to reproduce a different target.
- The linear baseline is already strong enough that added model complexity is
  not justified without a new, pre-specified research question.

## 5. Frozen-Probe transfer to PubMedQA

### 5.1 Protocol

Both BioASQ Probes were applied to the official 500-question PubMedQA PQA-L
test subset using the same Gemma 3 12B representation space and frozen block
24 / LT configuration.

There was no PubMedQA fitting, recalibration, feature selection, layer/token
selection, or threshold tuning.

The completed v2 condition supplies the official PubMed abstract and applies
the paper's Appendix-C `yes`/`no`/`maybe` annotation criteria. The model must
start with one decision label and then give a concise explanation.

### 5.2 Answer behaviour

Official label counts are 276 `yes`, 169 `no`, and 55 `maybe`.

| Metric | Context v2 |
| --- | ---: |
| Correct | **362/500** |
| Strict accuracy | **72.4%** |
| Balanced accuracy | **0.5652** |
| Macro-F1 | **0.5659** |
| Weighted-F1 | **0.7125** |
| Predicted `yes/no/maybe` | 298 / 164 / 38 |
| Official `yes/no/maybe` | 276 / 169 / 55 |
| `yes` recall | **84.1%** |
| `no` recall | **72.8%** |
| `maybe` recall | 12.7% |
| Error prevalence | 27.6% |

The main answer-quality result is 72.4% overall accuracy. The type-specific
limitation is low recall on the minority `maybe` label.

### 5.3 UQ against official PubMedQA decision error

| Score | v2 AUROC | v2 AP |
| --- | ---: | ---: |
| Frozen P(True)-Probe | **0.6839** | **0.4519** |
| Blind P(True) | 0.6490 | 0.4210 |
| Verbalized confidence | 0.6402 | 0.4164 |
| Frozen Accuracy-Probe | 0.5901 | 0.3916 |
| Sequence NLL | 0.5400 | 0.3025 |
| Normalized NLL | 0.5424 | 0.3075 |
| Mean token entropy | 0.5423 | 0.3080 |
| Max token entropy | 0.5372 | 0.2914 |

Interpretation:

- P(True)-Probe is the strongest v2 error-ranking score at 0.6839 AUROC.
- Blind P(True) and verbalized confidence provide weaker
  but measurable ranking signals.
- AP should be interpreted against the v2 error prevalence of 0.276.

## 6. Source documents

- `Phase-1 UQ Results Summary.pdf`
- `PHASE2_PROBE_PLAN.md`
- `archehr_sebaseline/docs/phase2_bioasq_dataset_split.md`
- `archehr_sebaseline/docs/pubmedqa_frozen_probe_transfer.md`
- `archehr_sebaseline/docs/experiment_runtime_log.md`
- `archehr_sebaseline/analysis_outputs/bioasq_phase2_linear_probes_seed31/candidate_metrics.csv`
- `archehr_sebaseline/analysis_outputs/pubmedqa_context_appendix_c_v2_full500_seed31_20260722/transfer_metrics.csv`

## 7. Proposed final research direction

The project will now be framed more broadly around **hidden-state uncertainty
probes**, rather than Semantic Entropy Probes specifically. This change is
motivated by the Phase-1 BioASQ results, where direct P(True) was consistently
stronger than Semantic Entropy. Phase 2 therefore tests whether a useful
P(True)-derived signal can be recovered from a single generation's hidden
states, alongside a separately supervised Accuracy-Probe that directly targets
answer correctness.

### 7.1 Main thesis focus

BioASQ will remain the primary experimental setting and the main source of
thesis claims. The central questions are:

1. How well can a simple hidden-state Probe recover a useful P(True)-derived
   uncertainty target in-domain?
2. How does this model-internal uncertainty signal differ from a Probe trained
   directly on external correctness labels?
3. How much computation is saved by replacing direct P(True) self-evaluation
   with hidden-state Probe scoring?
4. How robust are these conclusions across BioASQ factoid, list, and summary
   questions?

The strongest current result is the held-out BioASQ P(True)-Probe fidelity
AUROC of 0.9026. This will be reported together with the within-type results,
since the pooled score is partly affected by substantial differences in target
prevalence across question types.

### 7.2 Remaining core work

The remaining high-priority work is:

- **Direct efficiency measurement.** Measure paired wall-clock latency,
  generated-token count, throughput, and GPU time for direct P(True) versus
  hidden-state Probe scoring under the same model, hardware, batching, and
  generation settings. This is required before making a quantitative low-cost
  claim.
- **Question-type-aware analysis.** Investigate whether within-type thresholds,
  type-specific Probes, or another simple type-controlled formulation improves
  performance and clarifies why summary-question fidelity is much weaker than
  the pooled result.
- **Statistical robustness.** Use paired bootstrap confidence intervals for key
  comparisons where numerical differences are small, including Accuracy-Probe
  versus direct P(True), transfer comparisons, and any fusion result.

### 7.3 Token selection and model complexity

The original Phase-3 token-selection track is no longer a priority. The current
layer/token-position ablation shows no large universal advantage for a single
position, while the fixed block-24/LT linear Probes already perform strongly.
The existing candidate comparison will therefore be retained as an ablation,
but adaptive token selection or a HaMI-style extension will not be pursued
unless later question-type analysis identifies a clear representation bottleneck.

A small validation-controlled fusion experiment may still be run to test
whether P(True)-Probe and Accuracy-Probe scores are complementary. This will be
reported as a limited diagnostic rather than a new project phase. A useful
fusion result would require a held-out improvement over the strongest single
Probe, supported by paired uncertainty estimates.

### 7.4 PubMedQA as a secondary transfer track

PubMedQA will remain a secondary cross-dataset stress test rather than the main
thesis setting. The Appendix-C v2 result is encouraging: answer accuracy rises
to 72.4%, and the frozen P(True)-Probe is the strongest evaluated error-ranking
score at 0.6839 AUROC. However, transfer remains weaker than the in-domain
BioASQ result and is sensitive to the output contract, especially the minority
`maybe` class.

Further transfer experiments will be pursued only if they add a clear and
methodologically clean contribution. Otherwise, PubMedQA will be used to show
the boundary of frozen Probe transfer under dataset, prompt, and output-space
shift. Any additional prompt iteration on the already inspected 500-question
test set will be described as exploratory rather than confirmatory.

### 7.5 Intended thesis contribution

The intended thesis contribution is therefore a controlled study of cheap
hidden-state uncertainty probing for clinical and biomedical QA. The thesis
will compare P(True)-derived and correctness-derived supervision, quantify the
in-domain fidelity and error-ranking value of simple linear Probes, measure
their actual computational advantage over direct self-evaluation, and analyse
how question type and distribution shift affect their reliability.

The exact scope of the remaining transfer and question-type experiments will be
finalised with Simpson. The completed results are fixed; the purpose of the
meeting is to agree the final thesis emphasis and the smallest set of additional
experiments needed to support it.
