# BioASQ Isambard Baseline and Replication Results

Last updated: 2026-07-16

## Scope

This document records the two matched formal BioASQ training-summary runs and
the fixed-subset Qwen judge. It supersedes the earlier Isambard result notes in
chat and complements the historical RunPod results in
`bioasq_runpod_results_20260713.md`.

Both formal baseline runs used:

```text
dataset: BioASQ training13b summary
examples: 100
samples per example: 10
model: google/gemma-3-12b-it
device/dtype: CUDA / bfloat16
max new tokens: 192
clustering: microsoft/deberta-v2-xlarge-mnli, bidirectional entailment
self-report UQ: verbalized confidence and P(True)
quality evaluation: lightweight BioASQ-aligned reference metrics
```

The only intended generation difference was the random seed: 31 for job
5654721 and 47 for job 5660346. Both Level 4 health checks passed.

## Evaluation revision and temperature follow-up

The results below retain the historical, ROUGE-based reference target exactly
as recorded on 2026-07-15. On 2026-07-16, the local evaluator was revised to
report reference-answer coverage and document-level overlap between cited
snippet documents and BioASQ standard documents. When standard documents are
available, the active quality score is their geometric combination; the fixed
threshold therefore requires calibration against manual review and is not
numerically interchangeable with the historical `0.15` ROUGE threshold.

A paired follow-up at `temperature=1.0`, `top_p=0.9`, and the same 192-token
cap, model, 100x10 workload, NLI clustering, and seeds 31/47 completed on
2026-07-16. Jobs 5679663 and 5679664 both passed their full health checks.
Their local reference-only re-evaluation is complete; jobs 5683932 and
5683933 completed the ideal-answer NLI axis in 01:38 and 01:34 without
regenerating answers. The evidence-conditioned temperature-1.0 pair now has a
final three-axis result, although a matched temperature-0.8 three-axis result
is still required for a final temperature claim.

### Temperature 1.0 preliminary comparison

All four runs were re-evaluated locally with the same citation-aware reference
target. Mean answer quality is effectively unchanged, so increasing temperature
did not measurably alter this reference-only quality target in the two-seed
pair.

| Temperature | Seed 31 mean quality | Seed 47 mean quality | Mean low-quality examples |
|---|---:|---:|---:|
| 0.8 | 0.3070 | 0.3111 | 18.5 / 100 |
| 1.0 | 0.3079 | 0.3105 | 19.0 / 100 |

Within-question answer variation increased slightly at temperature 1.0:
mean semantic-cluster count rose from 3.115 to 3.235, normalized discrete SE
from 0.3506 to 0.3596, and per-question quality standard deviation from 0.0368
to 0.0416. Cross-seed quality remained highly stable (Spearman 0.9871 at 0.8,
0.9908 at 1.0). Thus temperature broadened the sampled answer distribution
without materially changing aggregate reference quality; NLI results are still
needed before drawing a conclusion about answer correctness/coverage.

Reference-target AUROC changes were mixed. Token entropy, P(True), normalized
NLL, and verbalized-confidence uncertainty improved by +0.019, +0.015, +0.031,
and +0.014 respectively when averaged over seeds. Discrete and weighted SE
each fell by about 0.019 on average, with opposite directions in the two seeds.
This two-seed result is descriptive rather than evidence of a stable temperature
effect.

### Evidence-conditioned three-axis result and direct-answer follow-up

For the temperature-1.0 evidence-conditioned outputs, mean three-axis quality
was 0.1968 (seed 31) and 0.1864 (seed 47), with fixed-threshold low-quality
rates of 0.56 and 0.60. These scores include ideal-answer NLI coverage and are
not numerically interchangeable with the reference-only table above.

Jobs 5684358 (seed 31) and 5684360 (seed 47) were then submitted for a matched
direct-answer ablation: the model receives the BioASQ question but no snippets,
while temperature, top-p, seeds, model, data order, sample count, output cap,
NLI clustering, and self-report UQ remain unchanged. Since citations cannot be
produced without supplied snippets, that condition will be assessed by the
ROUGE + ideal-answer-NLI two-axis fallback rather than by document overlap.

## Runtime

| Job | Purpose | Status | Elapsed |
|---|---|---|---:|
| 5654721 | seed-31 formal baseline | completed, exit 0 | 03:43:51 |
| 5660345 | fixed 30-question Qwen judge | completed, exit 0 | 00:20:56 |
| 5660346 | seed-47 matched repeat | completed, exit 0 | 03:18:31 |
| 5679663 | seed-31 temperature-1.0 repeat | completed, exit 0 | 02:36:46 |
| 5679664 | seed-47 temperature-1.0 repeat | completed, exit 0 | 02:28:21 |

The two matched 100x10 runs average about 03:31:12. Plan approximately 04:20
for the same cached configuration, excluding queue time, while retaining the
12-hour Slurm limit until more repetitions establish stable variance. Detailed
resource records are in `experiment_runtime_log.md`.

## Answer-quality replication

The two stochastic runs generated materially different text: only 12 of 1,000
paired answers were exact string matches. Their example-level answer quality
was nevertheless highly stable:

| Measure | Seed 31 | Seed 47 |
|---|---:|---:|
| Mean quality | 0.2238 | 0.2245 |
| Median quality | 0.1988 | 0.2025 |
| Examples below the 0.15 operational threshold | 30 | 30 |

Paired example-quality Spearman correlation was 0.9926 and the mean absolute
difference was 0.0104. Twenty-nine of the thirty fixed-threshold low-quality
examples were shared between runs (Jaccard 0.935). The deterministic quality
target is therefore much more seed-stable than the uncertainty ranking.

The quality score is the mean of unstemmed ROUGE-2 F1 and ROUGE-SU4 F1 against
the best ideal reference. It follows BioASQ Task B metric families but is a
transparent local approximation, not the official BioASQ evaluator. The 0.15
threshold is an SE sensitivity target, not an official BioASQ pass mark.

## UQ replication

| UQ score | AUROC seed 31 | AUROC seed 47 | Change |
|---|---:|---:|---:|
| Discrete semantic entropy | 0.687 | 0.609 | -0.078 |
| Weighted semantic entropy | 0.687 | 0.609 | -0.078 |
| Mean token entropy | 0.629 | 0.670 | +0.041 |
| P(True) uncertainty | 0.649 | 0.686 | +0.037 |
| Normalized NLL | 0.624 | 0.638 | +0.014 |
| Verbalized confidence | 0.539 | 0.578 | +0.039 |

For discrete SE, the fixed-threshold bootstrap interval changed from
0.576-0.801 to 0.486-0.729. The repeat point estimate remains above chance,
but its interval includes 0.5. Continuous risk association remained positive:
Spearman rho was 0.485 for seed 31 and 0.328 for seed 47, with both bootstrap
intervals above zero.

Selective rejection remained directionally useful but weakened in the repeat:

| Retained coverage | Seed-31 retained accuracy | Seed-47 retained accuracy |
|---|---:|---:|
| 100% | 0.700 | 0.700 |
| 80% | 0.738 | 0.725 |
| 50% | 0.820 | 0.760 |

The defensible result is that uncertainty has a positive relationship with
lexical answer risk, but SE discrimination is modest and seed-sensitive. The
claim that SE is the strongest UQ baseline is not replicated: token entropy and
P(True) outperform it in seed 47.

## Fixed-subset Qwen judge

Job 5660345 used Qwen/Qwen2.5-7B-Instruct with greedy decoding on ten examples
from each lexical-quality third and judged all ten generations per example.

```text
judgments: 300
fully parsed: 296
mean judge score: 0.871
lexical quality vs judge quality Spearman rho: 0.253
SE vs judge continuous risk Spearman rho: 0.149
judge-low-quality examples at threshold 0.5: 0
```

Four responses emitted every numeric rubric field but reached the 160-token
limit before closing the JSON rationale. More importantly, the judge exhibited
a strong ceiling/leniency effect: no example crossed the preregistered
low-quality threshold and every valid judgment received maximum relevance.
Judge-label AUROC is therefore undefined. The result provides only weak
directional evidence and should not be treated as external validation.

A future closed-model API judge remains inexpensive and technically feasible,
but it is deferred while conventional SE discrimination is weak. If resumed,
use a stricter error-oriented rubric, structured outputs, a pinned model, and
manual calibration rather than reusing the lenient 0.5 aggregate threshold.

## Current conclusion and next work

The BioASQ SE/UQ baseline is complete as a reproducible engineering and
experimental baseline. Answer quality is strongly replicated across seeds; SE
ranking is not strong enough to support a robust superiority claim.

Immediate priorities are:

1. Verify the local summary scores once with a standard or official BioASQ
   evaluation implementation.
2. Report both seeds, bootstrap intervals, continuous association, and
   rejection curves; do not select only the stronger seed-31 AUROC.
3. Diagnose SE failure cases and clustering/sample sensitivity before spending
   effort on a larger LLM-judge study or SEP supervision.
4. Keep closed-model judge evaluation as a later validation step after the
   conventional target and baseline behaviour are better understood.

Downloaded raw outputs and Slurm logs are retained locally under the ignored
`outputs/` run directories. They remain excluded from Git because they contain
large, generated experiment artifacts; this tracked document records the
portable result summary and provenance.
