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
