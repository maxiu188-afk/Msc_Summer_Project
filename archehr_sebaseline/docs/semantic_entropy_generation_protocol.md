# BioASQ Semantic Entropy generation protocol

This protocol follows the original Semantic Entropy implementation rather than
treating the first high-temperature sample as the model answer. Each question
has two distinct generation tracks.

1. `best_generations.jsonl` contains one low-temperature (`T=0.1`) answer.
   This is the model answer evaluated by the independent Claude text-comparison
   judge. `poor` is the only low-quality label used for AUROC and AURAC; `good`
   and `partial` are negative labels for that binary calculation.
2. `cleaned_generations.jsonl` contains the high-temperature (`T=1.0`) samples
   used only to calculate semantic entropy and the other UQ scores. Sampling is
   explicit: `top_p=0.9`, `top_k=50`.

For a new full run, `scripts/run_level4.py` writes both tracks. For an existing
high-temperature run, use `scripts/run_bioasq_best_isambard.sbatch` to add the
missing first track without regenerating its UQ samples. Claude evaluation
remains a separate local post-processing step through
`scripts/run_bioasq_claude_judge.py`, and writes only under `claude_judge/`.

## Current transition state

The four temperature-1.0 runs made before this correction contain only the
second track. On 2026-07-17, Isambard jobs 5692776/5692777 (with evidence,
seeds 31/47) and 5692779/5692780 (without evidence, seeds 31/47) were submitted
to add 100 low-temperature answers per run. They leave all existing SE, token,
NLI, self-report, and deterministic-evaluation files unchanged.

An earlier local Claude Sonnet 5 batch labelled all 4,000 high-temperature
samples at low effort. It is preserved as an exploratory condition-quality
analysis only: those samples are not the paper-protocol model answer and must
not be used for the primary UQ AUROC/AURAC results.
