# BioASQ medical-UQ results — Phase 1 complete (2026-07-19)

This is the canonical result record for the active BioASQ medical-QA UQ
direction. Earlier 100-question evidence/no-evidence and summary experiments
are pilot or archived diagnostics; they must not be presented as the current
benchmark.

## Final protocol

| Setting | Value |
|---|---|
| Dataset | `bioasq_medical_uq`, BioASQ training13b |
| Questions | 1,000 fixed questions: 480 factoid, 320 list, 200 summary |
| Model | `google/gemma-3-12b-it` |
| Samples | ten `T=1.0` samples for SE; one `T=0.1` main answer |
| Prompt | no evidence for all types |
| NLI | `pritamdeka/PubMedBERT-MNLI-MedNLI` |
| NLI rule | set-aware one-to-one bidirectional matching for factoid/list; ordinary bidirectional NLI for summary |
| UQ | SE variants, token/NLL baselines, exact disagreement, verbal confidence, blind P(True) |
| Judge | Claude Sonnet 5, binary `correct`/`incorrect`; factoid/list exact answers, summary ideal answers |
| Seeds | 31 / 47 |

Jobs 5706186/5706187 generated all 10,000 samples and 1,000 clusters per
seed in 8:00:08 / 7:14:24. They initially exited after computation because the
health check accepted only the old NLI name. Commit `7b90d54` corrected that
validation; both complete artifact sets then passed. Jobs 5715701/5715703
subsequently added blind P(True) and verbal confidence in 6:58 / 6:34 without
regenerating answers or clusters.

Claude batches returned 1,000 API successes per seed. A bounded retry of blank
responses raised the valid labels to 991 per seed; the remaining nine blanks in
each seed are excluded and are not retried. Seed 31 has 661 incorrect / 330
correct labels; seed 47 has 663 / 328. Of 988 questions with valid labels in
both seeds, 972 agree (98.4%).

## Final AUROC

Values are seed 31 / seed 47; `incorrect` is the positive class.

| Method | Overall | Factoid | List | Summary |
|---|---:|---:|---:|---:|
| Discrete SE | 0.763 / 0.779 | 0.720 / 0.725 | 0.845 / 0.881 | 0.565 / 0.595 |
| Likelihood-weighted SE | 0.763 / 0.779 | 0.718 / 0.724 | 0.841 / 0.881 | 0.565 / 0.596 |
| Number of clusters | 0.765 / 0.781 | 0.721 / 0.729 | **0.849 / 0.884** | 0.565 / 0.595 |
| Predictive entropy / normalized NLL / negative mean log-probability | 0.729 / 0.745 | 0.738 / 0.755 | 0.792 / 0.836 | 0.753 / 0.770 |
| Mean token entropy | 0.731 / 0.739 | 0.740 / 0.745 | 0.791 / 0.836 | 0.755 / 0.774 |
| Maximum token entropy | 0.720 / 0.722 | 0.724 / 0.731 | 0.827 / 0.873 | 0.745 / 0.743 |
| Sequence NLL | 0.650 / 0.654 | 0.723 / 0.737 | 0.815 / 0.869 | 0.783 / 0.785 |
| Exact-sample disagreement | 0.644 / 0.648 | 0.706 / 0.717 | 0.807 / 0.853 | 0.505 / 0.511 |
| Verbalized-confidence uncertainty | 0.696 / 0.695 | 0.662 / 0.663 | 0.806 / 0.820 | 0.681 / 0.670 |
| P(True)-blind | **0.811 / 0.821** | **0.795 / 0.800** | 0.845 / 0.852 | **0.808 / 0.826** |

The question-level artifacts and full overall/type-specific AUROC, rejection,
and AURAC CSVs are retained in the ignored local `server_results/` download and
on Isambard. They are generated data, not Git-tracked source.

## Natural answer-length contrast and UQ effect

Factoid, list, and summary provide an existing observational length gradient.
Because Semantic Entropy is computed from the ten `T=1.0` samples, the relevant
length is the mean generated-token count of those sampled answers, not the
length of the separately judged `T=0.1` main answer.

Values below are seed 31 / seed 47. AUROC is preferred for this cross-type
comparison because type-specific error prevalence differs.

| Type | Valid labels | Mean sampled-answer tokens | Discrete-SE AUROC | Blind-P(True) AUROC | P(True) minus SE, paired 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| Factoid | 477 / 479 | 8.08 / 8.01 | 0.720 / 0.725 | 0.795 / 0.800 | +0.075 `[+0.026,+0.123]` / +0.075 `[+0.024,+0.124]` |
| List | 317 / 317 | 35.42 / 35.41 | 0.845 / 0.881 | 0.845 / 0.852 | +0.000 `[-0.078,+0.068]` / -0.029 `[-0.088,+0.024]` |
| Summary | 197 / 195 | 111.13 / 110.92 | 0.565 / 0.595 | 0.808 / 0.826 | +0.243 `[+0.172,+0.311]` / +0.231 `[+0.155,+0.304]` |

This replicated pattern identifies different UQ operating regimes:

- on very short factoid answers, blind P(True) has a moderate advantage;
- on medium-length list answers, SE matches blind P(True), and cluster count is
  the strongest list point estimate in both seeds; and
- on long summary answers, whole-answer SE is weak while blind P(True) retains
  strong discrimination, producing by far the largest gap.

The contrast is not a causal estimate of answer length because answer
structure, question type, prompt, and NLI rule also change. It is nevertheless
the direct existing evidence that the SE-versus-P(True) relationship varies
with answer regime. The completed within-summary shortening intervention did
not recover SE, so length alone is not sufficient to explain the summary
failure and no additional length-specific generation experiment is required.

## Bounded long-summary clustering diagnostic

A fixed-generation, label-balanced mechanism diagnostic compared the accepted
PubMedBERT NLI clusters with a frozen Claude semantic-equivalence rubric on 48
current-condition summary questions. One repeatedly truncated correct-class
row was excluded without further retry, leaving 47 complete cases (23 correct,
24 incorrect).

| UQ score | AUROC | Average precision |
| --- | ---: | ---: |
| Blind P(True) | 0.825 | 0.812 |
| Original NLI Semantic Entropy | 0.579 | 0.579 |
| Claude-clustered Semantic Entropy | 0.727 | 0.749 |

Claude-clustered SE improved by `+0.149` AUROC over NLI-SE, but its
20,000-resample paired-bootstrap 95% interval `[-0.016,+0.313]` crosses zero.
The P(True)-minus-Claude-SE gap was `+0.098`, 95% interval
`[-0.061,+0.254]`.

The structural signal is strong: Claude produced 3.98 clusters per question
on average versus 1.23 for NLI, and 821 answer pairs were NLI-same but
Claude-different versus only 41 in the opposite direction. This makes NLI
over-merging a credible contributor to weak long-summary SE, but not a proven
dominant cause. The AUROC interval is unresolved, Claude's direct decisions
contain non-transitive triples, one row is missing, and the blinded 24-pair
human review remains pending. Full protocol and artifacts are recorded in
`summary_clustering_diagnostic.md`.

The terminal nested extension produced 93/96 complete cases (46 correct,
47 incorrect). Claude-clustered SE reached AUROC 0.781 versus 0.595 for
NLI-SE, a `+0.187` change with paired-bootstrap 95% CI
`[+0.080,+0.291]`. Blind P(True) was 0.815; its gap over SE fell from
`+0.220 [+0.110,+0.325]` under NLI clustering to
`+0.034 [-0.079,+0.147]` under Claude clustering. The alternative clustering
therefore closes about 85% of the original point-estimate gap.

With fixed generated answers, this is strong evidence that long-answer NLI
clustering difficulty is an important cause of the observed summary-SE
weakness. It is not a proposal to use Claude clustering in practice: the API
cost and latency are too high, shared-model bias remains possible because
Claude also supplied correctness labels, and the blinded pair review remains
pending.

## Gemma 3 4B versus 12B model-scale diagnostic

The Gemma 3 4B run changes only the generator checkpoint and uses the exact
Phase-1 seed-31 1,000-question cohort. The final comparison contains 990
questions with valid correctness labels for both models. `incorrect` is the
positive class.

| Model | Accuracy | Blind P(True) AUROC | Discrete-SE AUROC | P(True) minus SE |
| --- | ---: | ---: | ---: | ---: |
| Gemma 3 4B | 0.203 | 0.758 | 0.768 | -0.009 `[-0.051,+0.031]` |
| Gemma 3 12B | 0.332 | 0.811 | 0.764 | +0.047 `[+0.017,+0.076]` |

The paired 4B-minus-12B change in the P(True)-minus-SE gap is
`-0.056 [-0.103,-0.010]`. The clearest type-specific result is summary:
the gap changes from `+0.243 [+0.173,+0.313]` at 12B to
`+0.005 [-0.085,+0.094]` at 4B, a paired change of
`-0.238 [-0.347,-0.129]`.

Model capability therefore changes the UQ operating regime. Blind P(True)'s
self-evaluation advantage is present at 12B and disappears at 4B. This is not
evidence that SE becomes a strong small-model summary method: 4B summary
AUROC is only 0.648 for P(True) and 0.643 for SE, while normalized NLL reaches
0.794. The scale result is primarily P(True) weakening with lower capability.

## Statistical comparison and conclusion

The paired bootstrap AUROC difference, P(True)-blind minus discrete SE, is:

| Type | Seed 31 | Seed 47 |
|---|---:|---:|
| Overall | +0.048, 95% CI [+0.018, +0.075] | +0.042, 95% CI [+0.014, +0.071] |
| Factoid | +0.075, 95% CI [+0.026, +0.123] | +0.075, 95% CI [+0.024, +0.124] |
| List | +0.000, 95% CI [-0.078, +0.068] | -0.029, 95% CI [-0.088, +0.024] |
| Summary | +0.243, 95% CI [+0.172, +0.311] | +0.231, 95% CI [+0.155, +0.304] |

- **P(True)-blind is the strongest global baseline**, with a replicated
  bootstrap-supported advantage over SE overall, factoid, and summary.
- **SE remains a compelling list-specific method**: cluster count is the best
  list score in both seeds and P(True) is statistically indistinguishable there.
- **SE is not a summary method** under this protocol. Summary should instead
  retain P(True), sequence NLL, and token/NLL baselines.
- **P(True)-10 is retired.** It was only an early 100-question ablation and is
  neither generated nor evaluated in Phase 1.
- A future **P(True)-Probe** must train on
  `p_true_blind_uncertainty`, use grouped held-out questions, and be judged by
  fidelity/cost against direct blind P(True), not by an unsupported claim of
  beating it.

## Final post-hoc fusion diagnostic

The reproducible CPU-only `analysis/run_uq_feature_fusion.py` was rerun on the
final 991 labels per seed with repeated 5-fold OOF logistic regression and
2,000 paired bootstrap resamples. It neither calls a model nor changes the
Phase-1 artifacts.

| Comparison with P(True)-blind | Seed 31 AUROC change (95% CI) | Seed 47 AUROC change (95% CI) |
|---|---:|---:|
| P(True)-blind + discrete SE | +0.013 [-0.005, +0.035] | +0.014 [-0.005, +0.034] |
| P(True)-blind + normalized NLL | -0.040 [-0.074, -0.006] | -0.036 [-0.068, -0.004] |
| P(True)-blind + discrete SE + normalized NLL | +0.001 [-0.025, +0.031] | +0.005 [-0.022, +0.033] |

Discrete SE has a positive conditional coefficient in both seeds, but no
fusion clears the paired-bootstrap criterion. NLL fusion is reliably worse.
Therefore do not pursue a multi-target SE/NLL/P(True) Probe. A narrow
single-target P(True)-Probe remains a valid next experiment only as a
lower-cost fidelity study against direct blind P(True).

## Superseded 100-question pilot

The 100-question two-seed pilot established the protocol, PubMedBERT set-aware
NLI, binary Claude judge, and the P(True)-10 failure. Its raw local downloads
are archived under `archive_unused/results/server_results/`; its post-hoc fusion
diagnostic was inconclusive because its paired-bootstrap intervals crossed
zero. The reproducible fusion script is retained and now defaults to this
Phase-1 1,000-question baseline.
