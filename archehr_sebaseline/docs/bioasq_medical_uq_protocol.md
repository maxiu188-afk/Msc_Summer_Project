# BioASQ medical-UQ protocol

This is the Phase-1 closing protocol and the direct reference for Phase 2. It
tests whether Semantic Entropy remains useful at a larger scale while preserving
a direct blind P(True) target and a binary Claude correctness target for the
P(True)-Probe and Accuracy-Probe respectively. The completed project boundary
and reproducibility map are in `../../PROJECT_CLOSEOUT.md`.

- Use `bioasq_medical_uq`: factoid, list, and summary questions only. Yes/no
  questions are excluded.
- Prompts are type-specific. Factoid and list answers are constrained to
  semicolon-delimited entities/items; summary answers are concise paragraphs.
- All active runs use no retrieval evidence. Summary remains evidence-free even
  if a future mixed prompt run enables snippets for other types.
- Cluster factoid/list answers with PubMedBERT-MNLI-MedNLI and require a
  one-to-one, bidirectional-entailment matching of answer items. This rejects
  a list that omits or invents an item.
- Judge one low-temperature answer per question with Claude as `correct` or
  `incorrect`; there is no partial-credit category. Factoid/list references use
  `exact_answers`, summaries use `ideal_answers`.
- Report blind P(True) from the same low-temperature answer:
  `p_true_blind_uncertainty` sees only the question and answer. The concise
  `p_true_uncertainty` field is an alias of this blind condition. P(True)-10
  is retired and must not be generated or evaluated.
- Evaluate all remaining question-level UQ baselines: discrete and
  likelihood-weighted SE, cluster count, predictive entropy, mean/max token
  entropy, normalized and sequence NLL, negative mean token log-probability,
  exact-sample disagreement, and verbalized-confidence uncertainty.
- The Phase-1 closing sample has exactly 1,000 questions: 480 factoid, 320
  list, and 200 summary. A fixed selection seed makes both generation seeds
  use the same stratified question set.

## Final generation tracks and UQ naming

Each question has two deliberately separate generation tracks:

1. one `T=0.1` main answer used for binary correctness judging, blind P(True),
   verbalized confidence, and single-answer token/NLL diagnostics;
2. ten `T=1.0`, `top_p=0.9`, `top_k=50` samples used for semantic clustering,
   discrete/likelihood-weighted SE, cluster count, ten-sample NLL, token
   entropy, and sample disagreement.

`10-sample normalized NLL` is the mean per-token NLL across the ten sampled
answers. Historical fields `predictive_entropy`, `mean_normalized_nll`, and
negative mean token log-probability are rank-equivalent aliases in these
artifacts, not separate estimators. `10-sample sequence NLL` is unnormalized
and remains a distinct, length-confounded method. P(True)-10 is retired: the
active blind P(True) sees only the question and the low-temperature proposed
answer.

Run the GPU smoke before any full experiment:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
sbatch scripts/run_bioasq_medical_uq_smoke_isambard.sbatch
```

The smoke is three questions and three high-temperature samples. It validates
generation, PubMedBERT NLI, output health, and blind P(True); Claude is a
separate local post-processing stage and is intentionally not submitted from
the cluster.

## Phase 1 final result (2026-07-19)

The 1,000-question Phase-1 baseline is complete at seeds 31/47. After one
bounded retry, both seeds retain 991 valid Claude labels; the remaining nine
blank labels per seed are excluded. P(True)-blind is best overall
(0.811/0.821 AUROC), SE/cluster count is strongest on list questions
(0.849/0.884 cluster count), and SE remains weak on summary (0.565/0.595).
The full table, runtime provenance, and bootstrap comparison are in
`bioasq_medical_uq_results_20260718.md`.

The superseded 100-question two-seed pilot is retained only in
`../../archive_unused/docs/historical_results/bioasq_phase1_100q_pilot.md`.
