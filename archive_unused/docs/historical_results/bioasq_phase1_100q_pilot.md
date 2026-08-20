# Superseded BioASQ Phase-1 100-question Pilot

Archived: 2026-08-17
Original source: the former pilot section in
`archehr_sebaseline/docs/bioasq_medical_uq_protocol.md`

This pilot established the early no-evidence protocol but is not the current
Phase-1 result. The active result uses the fixed 1,000-question cohort and is
owned by `../../../archehr_sebaseline/docs/bioasq_medical_uq_results_20260718.md`.

The pilot completed 100 questions with ten `T=1.0` samples, one `T=0.1` target
answer, and seeds 31/47. Both Isambard jobs passed health checks in 49:14 and
53:25. The sample contained 44 factoid, 35 list, and 21 summary questions.
Claude batches returned 98 valid labels per seed: 56 incorrect and 42 correct;
two blank outputs per seed were excluded without further retry.

AUROC, reported as seed 31 / seed 47:

| Type | Discrete SE | P(True)-blind | Retired P(True)-10 |
| --- | ---: | ---: | ---: |
| Overall | 0.729 / 0.759 | 0.781 / 0.824 | 0.717 / 0.758 |
| Factoid | 0.682 / 0.758 | 0.763 / 0.794 | 0.734 / 0.775 |
| List | 0.704 / 0.712 | 0.770 / 0.908 | 0.640 / 0.768 |
| Summary | 0.500 / 0.462 | 0.823 / 0.709 | 0.484 / 0.484 |

Weighted SE and number of clusters tracked discrete SE. On summary, sequence
NLL (`0.781 / 0.824`) and token entropy (`0.792 / 0.681`) were more informative
than SE. The high-temperature-answer P(True) condition did not improve P(True)
and was retired. These values must not replace or be pooled with the formal
1,000-question result.
