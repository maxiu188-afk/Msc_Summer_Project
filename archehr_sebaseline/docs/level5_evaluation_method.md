# Level 5 PubMedQA Evaluation Method

Last updated: 2026-07-04

## Purpose

Level 5 evaluates whether the uncertainty scores produced by the Level 4
Semantic Entropy baseline can identify answers that are likely to be wrong.

The current implementation is a lightweight PubMedQA proxy evaluation. It uses
PubMedQA's `final_decision` label (`yes`, `no`, or `maybe`) as the answer-quality
target. It does not yet evaluate full clinical factuality, evidence support, or
citation correctness.

## Input Artifacts

The evaluator reads a completed Level 4 output directory. Required files are:

```text
examples.jsonl
cleaned_generations.jsonl
se_scores.csv
```

The expected source of each file is:

- `examples.jsonl`: normalized examples, including the PubMedQA gold label.
- `cleaned_generations.jsonl`: generated answers after text cleaning.
- `se_scores.csv`: example-level Semantic Entropy and token-level uncertainty scores.

## Gold Labels

For each example, the evaluator reads the gold label from:

```text
examples.jsonl -> label
```

Valid labels are:

```text
yes
no
maybe
```

Any missing or unsupported gold label is normalized to:

```text
unknown
```

## Generated Label Extraction

Each generated answer is mapped to a predicted PubMedQA decision label using a
conservative rule-based extractor.

The extractor looks for explicit label patterns such as:

```text
The answer is: yes
[yes]
Therefore, the answer is maybe
No, ...
```

If no clear `yes`, `no`, or `maybe` label is found, the generated label is:

```text
unknown
```

This `unknown` value means the evaluator could not parse a decision label from
the generation. It is not a model uncertainty score and should not be interpreted
as Semantic Entropy.

## Per-Sample Correctness

Each generation is scored against the gold PubMedQA label:

```text
sample_correct = predicted_label == gold_label
```

The summary metric `per_sample_accuracy_known` only includes generations where
the predicted label was successfully parsed as `yes`, `no`, or `maybe`.
Generations with predicted label `unknown` are excluded from this particular
known-label accuracy.

## Majority-Vote Correctness

For each example, the evaluator aggregates the labels from all generated samples
using majority vote.

Rules:

1. Ignore `unknown` labels when counting votes.
2. If no known labels remain, the majority label is `unknown`.
3. If two or more known labels are tied, the majority label is `unknown`.
4. Otherwise, the majority label is the known label with the highest count.

The example-level correctness target is:

```text
majority_correct = majority_label == gold_label
```

The example is considered incorrect when:

```text
majority_label != gold_label
```

This majority-vote target is the current target used for AUROC and rejection
curve evaluation.

## Uncertainty Scores Evaluated

The evaluator tests whether the following Level 4 scores detect incorrect
majority-vote answers:

```text
normalized_discrete_semantic_entropy
normalized_likelihood_weighted_semantic_entropy
predictive_entropy
mean_normalized_nll
mean_token_entropy
```

Higher values are treated as more uncertain.

## AUROC

For each uncertainty score, the evaluator computes AUROC for detecting incorrect
majority-vote answers:

```text
y_true  = 1 if majority_label != gold_label else 0
y_score = uncertainty score
```

AUROC is interpreted as the probability that a randomly chosen incorrect example
receives a higher uncertainty score than a randomly chosen correct example.

The implementation computes AUROC directly, without adding a `scikit-learn`
dependency.

## Rejection Curve

The evaluator also computes a selective-prediction or rejection curve.

For each uncertainty score:

1. Sort examples from lowest uncertainty to highest uncertainty.
2. Retain the lowest-uncertainty examples.
3. Reject increasingly large fractions of the highest-uncertainty examples.
4. Compute retained accuracy at each rejection fraction.

If an uncertainty score is useful, retained accuracy should increase as the
highest-uncertainty examples are rejected.

## Output Artifacts

The evaluator writes:

```text
pubmedqa_label_predictions.csv
pubmedqa_eval_summary.json
rejection_curve.csv
auroc_bar.svg
rejection_curve.svg
```

### `pubmedqa_label_predictions.csv`

One row per generated answer. Important fields include:

```text
example_id
sample_id
gold_label
predicted_label
sample_correct
majority_label
majority_correct
known_vote_count
total_vote_count
is_tie
clean_answer
```

### `pubmedqa_eval_summary.json`

Run-level summary containing:

```text
num_examples
num_generations
gold_label_counts
majority_label_counts
majority_correct
majority_accuracy
majority_unknown
majority_ties
known_sample_predictions
unknown_sample_predictions
per_sample_accuracy_known
auroc_incorrect_by_score
```

### `rejection_curve.csv`

One row per uncertainty score and rejection fraction. Important fields include:

```text
score_name
rejection_fraction
coverage
num_retained
accuracy
correct_count
total_count
threshold
```

### SVG Plots

The evaluator writes two dependency-free SVG plots:

```text
auroc_bar.svg
rejection_curve.svg
```

These can be opened directly in a browser.

## Current 50x5 Pilot Result

For the local Qwen2.5-7B PubMedQA Level 4 pilot:

```text
examples: 50
generations: 250
num_samples: 5
```

The maintained Level 5 evaluator reports:

```text
majority accuracy: 28/50 = 0.560
known per-sample label accuracy: 0.649
unknown sample predictions: 99/250
```

AUROC for detecting incorrect majority-vote answers:

```text
normalized discrete SE: 0.547
normalized likelihood-weighted SE: 0.551
predictive entropy: 0.570
mean normalized NLL: 0.570
mean token entropy: 0.557
```

## Important Limitations

This evaluation is useful as a lightweight baseline, but it is not equivalent to
the original Semantic Entropy paper's main setup.

Main limitations:

1. The target is majority-vote PubMedQA label correctness, not correctness of a
   single low-temperature answer.
2. Label extraction is rule-based and conservative; many generated answers are
   currently parsed as `unknown`.
3. The current run uses `num_samples=5`, while the original Semantic Entropy
   paper used ten generations for sentence-length QA experiments.
4. PubMedQA here is grounded with evidence sentences, whereas the original paper
   intentionally removed context passages in QA experiments to induce
   confabulations.
5. The current correctness target does not evaluate whether the explanation is
   fully supported by evidence or whether citations are correct.
6. Stable but wrong answers may have low Semantic Entropy, because SE is better
   suited to detecting arbitrary or unstable errors than consistent mistakes.

## Recommended Next Improvements

The next improvements should make the target more comparable and less noisy:

1. Force future generations to output a structured first line:

```text
Decision: yes/no/maybe
Answer: ...
```

2. Add an evaluation target for `sample_id=0` correctness, closer to the original
   paper's single-answer correctness setup.
3. Run an `N=10` baseline to reduce discretization in Semantic Entropy estimates.
4. Add an automatic case-analysis report for low-SE wrong examples, high-SE
   wrong examples, and examples dominated by `unknown` label extraction.

