# BioASQ Semantic Entropy generation protocol

This protocol follows the original Semantic Entropy implementation rather than
treating the first high-temperature sample as the model answer. Each question
has two distinct generation tracks.

1. `best_generations.jsonl` contains one low-temperature (`T=0.1`) answer.
   This is the model answer evaluated by the independent Claude text-comparison
   judge. The direct-answer condition uses `poor` as low quality; the
   evidence-conditioned condition uses `partial` or `poor`. These are separate
   condition-specific targets and must not be pooled.
2. `cleaned_generations.jsonl` contains the high-temperature (`T=1.0`) samples
   used only to calculate semantic entropy and the other UQ scores. Sampling is
   explicit: `top_p=0.9`, `top_k=50`.

Verbalized confidence is calculated once for the low-temperature answer.
P(True) is also calculated once for that answer, with the high-temperature
samples included only as possible-answer context, matching the reference
implementation's separation of prediction and stochastic alternatives.
This differs from verbalized confidence, which sees only the question,
available evidence, and the low-temperature answer. In the reference P(True)
construction, the high-temperature answers are explicitly named
"Brainstormed Answers"; the low-temperature answer remains the separately
scored "Possible answer".

`predictive_entropy` is retained as a historical field name, but in this
implementation it is the Monte-Carlo mean negative per-token log-probability
of the high-temperature samples. It is therefore rank-equivalent to
`mean_normalized_nll` and `avg_token_logprob_uncertainty`, not a separately
computed full-vocabulary predictive-distribution entropy. `mean_token_entropy`
is the distinct token-distribution entropy baseline.

For a new full run, `scripts/run_level4.py` writes both tracks. For an existing
high-temperature run, use `scripts/run_bioasq_best_isambard.sbatch` to add the
missing first track without regenerating its UQ samples. Claude evaluation
remains a separate local post-processing step through
`scripts/run_bioasq_claude_judge.py`, and writes only under
`claude_main_answer_judge/`. This deliberately leaves the earlier
high-temperature exploratory labels in `claude_judge/` untouched.

## Current transition state

The four temperature-1.0 runs made before this correction initially contained
only the second track. On 2026-07-17, Isambard jobs 5692776/5692777 (with
evidence, seeds 31/47) and 5692779/5692780 (without evidence, seeds 31/47)
completed the 100-answer low-temperature backfill per run. Jobs
5696576--5696579 then recomputed self-report UQ using the corrected protocol.
All eight jobs completed with exit code 0. The downloaded `self_report` tables
contain 100 questions each, `answer_source=best_generation_low_temperature`,
and ten high-temperature answers only in P(True)'s possible-answer context.

An earlier local Claude Sonnet 5 batch labelled all 4,000 high-temperature
samples at low effort. It is preserved as an exploratory condition-quality
analysis only: those samples are not the paper-protocol model answer and must
not be used for the primary UQ AUROC/AURAC results.
