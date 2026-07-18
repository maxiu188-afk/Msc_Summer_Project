# BioASQ medical-UQ two-seed result — 2026-07-18

## Protocol

| Setting | Value |
|---|---|
| Jobs | Isambard 5702970 / 5702980 |
| Seeds | 31 / 47 |
| Dataset | `bioasq_medical_uq`, training13b; 44 factoid, 35 list, 21 summary |
| Model | `google/gemma-3-12b-it` |
| Generation | 100 questions, ten `T=1.0` samples, one `T=0.1` target answer |
| Prompt | no evidence for every type |
| Clustering | `pritamdeka/PubMedBERT-MNLI-MedNLI`; one-to-one set matching for factoid/list |
| Target | low-temperature answer, Claude `correct` / `incorrect` |

Both jobs passed structural health checks. Runtime was 49:14 (seed 31) and
53:25 (seed 47). Claude produced 98 valid labels per seed: 56 incorrect and
42 correct. Two blank outputs per seed were excluded without further retry.
The 98 common valid labels agree across seeds for 94 questions (95.9%).

## AUROC

Values are seed 31 / seed 47. These are descriptive results only; bootstrap
intervals and qualitative error analysis remain outstanding.

| Method | Overall | Factoid | List | Summary |
|---|---:|---:|---:|---:|
| Discrete SE | 0.729 / 0.759 | 0.682 / 0.758 | 0.704 / 0.712 | 0.500 / 0.462 |
| Weighted SE | 0.732 / 0.758 | 0.696 / 0.766 | 0.694 / 0.684 | 0.500 / 0.462 |
| Number of clusters | 0.734 / 0.759 | 0.697 / 0.748 | 0.714 / 0.719 | 0.500 / 0.462 |
| Predictive entropy / normalized NLL | 0.675 / 0.674 | 0.668 / 0.698 | 0.702 / 0.702 | 0.708 / 0.659 |
| Token entropy | 0.684 / 0.682 | 0.673 / 0.705 | 0.712 / 0.691 | 0.792 / 0.681 |
| Sequence NLL | 0.601 / 0.597 | 0.624 / 0.687 | 0.666 / 0.671 | 0.781 / 0.824 |
| Verbalized-confidence uncertainty | 0.596 / 0.648 | 0.622 / 0.662 | 0.758 / 0.768 | 0.464 / 0.582 |
| P(True)-blind | **0.781 / 0.824** | **0.763 / 0.794** | **0.770 / 0.908** | 0.823 / 0.709 |
| P(True)-10 | 0.717 / 0.758 | 0.734 / 0.775 | 0.640 / 0.768 | 0.484 / 0.484 |

The local ignored run directories contain the complete labels, overall metrics,
and `claude_binary_main_answer_judge/by_type/{factoid,list,summary}/` metrics.

## Interpretation and pause point

- SE has a stable useful signal for factoid/list, but is at chance for summary.
- P(True)-blind is the strongest overall method in this run.
- Giving P(True) the ten high-temperature answers does not help and often hurts,
  especially on summary. It likely adds sampling disagreement and hallucinated
  answer items rather than correctness evidence.
- Do not start a P(True)-probe from the P(True)-10 target.
- Do not pool the three BioASQ types as a single homogeneous task in future
  reports or decisions.

No additional experiment is committed. Before the next direction is chosen,
review uncertainty intervals, representative failure cases, and the relation
between high-temperature disagreement and correctness.
