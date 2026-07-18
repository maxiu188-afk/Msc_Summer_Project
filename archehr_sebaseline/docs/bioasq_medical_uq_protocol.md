# BioASQ medical-UQ protocol

This is the active protocol for testing both whether Semantic Entropy remains
useful and whether P(True) benefits from high-temperature alternatives.

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
- Report two P(True) uncertainty scores from the same low-temperature answer:
  `p_true_blind_uncertainty` sees only the question and answer, while
  `p_true_with_samples_uncertainty` additionally sees the ten high-temperature
  answers. The historical `p_true_uncertainty` remains an alias of the latter.

Run the GPU smoke before any full experiment:

```bash
cd "$SCRATCHDIR/final_project/archehr_sebaseline"
DATA_PATH="$SCRATCHDIR/final_project/data/BioASQ-training13b/training13b.json" \
sbatch scripts/run_bioasq_medical_uq_smoke_isambard.sbatch
```

The smoke is three questions and three high-temperature samples. It validates
generation, PubMedBERT NLI, output health, and both P(True) paths; Claude is a
separate local post-processing stage and is intentionally not submitted from
the cluster.

## Completed two-seed result (2026-07-18)

The full no-evidence run completed with 100 questions, ten `T=1.0` samples,
one `T=0.1` target answer, and seeds 31/47. Both Isambard jobs passed health
checks (49:14 and 53:25). The sample is 44 factoid, 35 list, and 21 summary
questions. Claude batches returned 98 valid labels per seed (56 incorrect, 42
correct); two blank outputs per seed are excluded and were not retried further.

AUROC, reported as seed 31 / seed 47, is:

| Type | Discrete SE | P(True)-blind | P(True)-10 |
|---|---:|---:|---:|
| Overall | 0.729 / 0.759 | 0.781 / 0.824 | 0.717 / 0.758 |
| Factoid | 0.682 / 0.758 | 0.763 / 0.794 | 0.734 / 0.775 |
| List | 0.704 / 0.712 | 0.770 / 0.908 | 0.640 / 0.768 |
| Summary | 0.500 / 0.462 | 0.823 / 0.709 | 0.484 / 0.484 |

Weighted SE and number of clusters track discrete SE. On summary, sequence NLL
(0.781 / 0.824) and token entropy (0.792 / 0.681) are more informative than
SE. The high-temperature-answer condition does not improve P(True); it likely
adds answer-sampling disagreement rather than reliable correctness evidence.
This is a descriptive two-seed result, so the next research direction remains
open. Do not start a P(True)-probe or pool types before a careful review.
The complete recorded table and pause-point rationale are in
`bioasq_medical_uq_results_20260718.md`.
