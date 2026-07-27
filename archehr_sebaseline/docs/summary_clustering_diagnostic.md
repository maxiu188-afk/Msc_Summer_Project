# Bounded summary clustering diagnostic

Last updated: 2026-07-26

## Question and boundary

This is the conditional Stage-3 mechanism diagnostic defined in
`../../PHASE2_PROBE_PLAN.md`. It asks only whether the accepted PubMedBERT NLI
semantic-equivalence decisions are a major contributor to weak Semantic
Entropy on the original, longer BioASQ summary answers.

No answer is regenerated, no correctness label is changed, and the shorter
prompt condition is not included. Claude clustering is not proposed as a
deployment method. Its API cost and latency make it unsuitable as the
practical clustering component being optimized here; it is used only as an
offline counterfactual judge to test whether NLI struggles with long answers.

## Frozen bounded cohort

The diagnostic uses 48 of the 123 current-condition questions:

- 24 with a valid `correct` main-answer label and 24 with a valid `incorrect`
  label;
- deterministic SHA-256 ranking within each label, seed `20260726`;
- ten already saved temperature-1 samples per question;
- 480 answer texts total, represented by 48 Claude batch requests; and
- 45 independently judged answer pairs per request.

This first stage is sized to detect a large mechanism signal, not a subtle
AUROC change. It must not expand beyond 48 questions automatically if the
result is inconclusive.

## Frozen equivalence rule

For each question, `claude-sonnet-5` with low effort judges all unordered pairs
of its ten answers. Two answers are equivalent only if they preserve the same
material answer to the question. Paraphrases, acronym expansions, reordered
claims, and harmless stylistic details are allowed. A material added or
omitted claim, contradiction, different named item, intervention, outcome,
polarity, scope, strength, or certainty makes the pair non-equivalent.
Correctness against a reference is explicitly excluded from this decision.

Each pair is judged independently. The first batch requested one frozen 45-bit
decision string in lexicographic pair order. When that serialization proved
unreliable, the retry retained the same rubric and pair order but used
schema-constrained JSON containing an array of 45 integer decisions. The
resulting decisions are converted to clusters using the same
representative-first, sample-order greedy rule as the accepted NLI pipeline.
Non-transitive triple decisions are counted and reported rather than silently
repaired.

## Frozen analysis

On the initial 48-question label-balanced cohort, and then identically on the
explicitly authorized nested 96-question extension, report:

- Claude-versus-NLI adjusted Rand index and pair co-clustering agreement;
- the two disagreement directions separately;
- cluster-count, cluster-size, and largest-cluster-fraction summaries;
- direct Claude equivalence transitivity violations;
- blind P(True), NLI-SE, Claude-SE, NLI cluster-count, and Claude
  cluster-count AUROC/AP; and
- paired, label-stratified 20,000-resample bootstrap intervals, seed
  `20260726`, for Claude-minus-NLI AUROC changes and the remaining
  P(True)-minus-SE gaps.

The analysis creates a deterministic blinded 24-pair human-review worksheet,
with a separate comparison key. A claim that Claude gives better semantic
equivalence remains conditional on that human review; shared-model bias is
otherwise a live alternative because Claude also supplied the correctness
labels.

## Current status

The protocol and 48-question cohort are frozen locally. The first Claude batch
`msgbatch_01BcAEtYs7FmcPVtNgaMaz2D` ended with all 48 API requests succeeded,
but only 2 responses contained all 45 decisions. Twenty ended normally after
42 decisions, while 26 hit `max_tokens`; therefore that batch is not a
scientific result.

A one-question schema-constrained retry smoke returned 45/45 valid decisions
with `end_turn`. Retry batch `msgbatch_01DvxtFBPJoeBtGequbL4Wpc` was then
submitted for only the 46 invalid questions with `max_tokens=512`. All 46 API
requests succeeded; strict validation accepted 37 and rejected 9 that still
hit `max_tokens`. Thus 39/48 questions are currently valid when combined with
the two valid first-pass rows.

Second repair batch `msgbatch_0147chhsAUMSZxS7zkTiBaE9` contains only those
nine truncated rows, retains the schema and all scientific inputs, and raises
`max_tokens` to 2,048. All nine API requests succeeded and eight passed strict
validation. The remaining correct-class question
`552446612c8b63434a00000c` again hit `max_tokens` and is excluded without
another retry, as requested. The final quantitative analysis is therefore a
47-question complete-case diagnostic: 23 correct and 24 incorrect.

## Quantitative result

| UQ score | AUROC | Average precision |
| --- | ---: | ---: |
| Blind P(True) | 0.825 | 0.812 |
| Original NLI Semantic Entropy | 0.579 | 0.579 |
| Claude-clustered Semantic Entropy | 0.727 | 0.749 |
| Original NLI cluster count | 0.583 | 0.578 |
| Claude cluster count | 0.720 | 0.737 |

Claude-clustered SE minus original NLI-SE AUROC is `+0.149`, with a
20,000-resample label-stratified paired-bootstrap 95% interval
`[-0.016,+0.313]`. Blind P(True) minus Claude-clustered SE is `+0.098`, with
95% interval `[-0.061,+0.254]`. The first interval crosses zero, so the AUROC
gain is suggestive rather than resolved on this bounded sample.

The cluster structures differ materially:

- mean cluster count is 1.23 for NLI and 3.98 for Claude;
- mean largest-cluster fraction is 0.943 for NLI and 0.657 for Claude;
- mean pair agreement is 0.592 and mean adjusted Rand index is 0.396;
- among 2,115 answer pairs, 821 are NLI-same/Claude-different, versus only 41
  NLI-different/Claude-same; and
- Claude's direct pair decisions contain 96 transitivity-violating triples.

This is evidence that the accepted NLI pipeline merges many distinctions that
Claude preserves, and replacing that clustering produces a sizeable SE AUROC
point-estimate recovery. It does not yet establish that Claude is the more
semantically correct judge: the frozen 24-pair blinded human review remains
pending, the Claude pair decisions are sometimes non-transitive, the
correctness labels also come from Claude, and one correct-class question is
missing. The bounded conclusion is therefore that NLI over-merging is a
credible contributor to weak long-summary SE, but it is not proven to be the
sole or dominant cause.

## Nested 96-question extension

The 47-question complete-case estimate is useful but small. A single bounded
extension doubles the frozen target cohort from 48 to 96 questions, still
restricted to the original long-summary condition:

- 48 correct and 48 incorrect questions under the same label source;
- the original 48-question cohort is an exact subset;
- only 48 newly selected questions require Claude calls;
- the same saved ten answers, rubric, pair order, model, and structured output
  are retained;
- the repeatedly truncated base row is not retried, and any invalid extension
  rows will be excluded without retry; and
- the extension must not grow automatically to the Phase-1 1,000-question
  corpus.

Extension batch `msgbatch_015tEX5qdiBHpb2sUUME4gDW` contains the 48 new
questions. All 48 API requests succeeded. Two new rows hit `max_tokens` and
were excluded without retry; together with the one invalid base row, the final
analysis contains 93/96 complete cases: 46 correct and 47 incorrect.

### Expanded UQ result

| UQ score | AUROC | Average precision |
| --- | ---: | ---: |
| Blind P(True) | 0.815 | 0.804 |
| Original NLI Semantic Entropy | 0.595 | 0.586 |
| Claude-clustered Semantic Entropy | 0.781 | 0.788 |
| Original NLI cluster count | 0.597 | 0.593 |
| Claude cluster count | 0.777 | 0.779 |

Claude-clustered SE minus NLI-SE AUROC is `+0.187`, with a
20,000-resample label-stratified paired-bootstrap 95% interval
`[+0.080,+0.291]`. Unlike the initial 47-question interval, this interval does
not cross zero. The blind-P(True)-minus-SE gap changes from `+0.220`
`[+0.110,+0.325]` with NLI clustering to `+0.034`
`[-0.079,+0.147]` with Claude clustering. Thus the alternative clustering
closes about 85% of the original P(True)-minus-SE point-estimate gap.

The expanded structural result replicates the original direction:

- mean cluster count is 1.19 for NLI and 3.88 for Claude;
- mean largest-cluster fraction is 0.960 for NLI and 0.668 for Claude;
- mean pair agreement is 0.588 and mean adjusted Rand index is 0.368; and
- 1,673 pairs are NLI-same/Claude-different, versus 50 in the opposite
  direction.

Because the ten generated answers and correctness labels are fixed, this shows
that the semantic-equivalence layer is a major contributor to the observed
long-summary SE weakness under this diagnostic. It does not make Claude
clustering the proposed method: its API cost and latency are impractical, its
direct decisions contain 210 transitivity-violating triples, and using Claude
for both correctness labels and alternative clustering creates shared-model
bias. The blinded 24-pair human review therefore remains necessary before
claiming that Claude is semantically more correct. The bounded project
conclusion is that long-answer NLI clustering difficulty is strongly
implicated; a practical replacement is outside this diagnostic's scope.
