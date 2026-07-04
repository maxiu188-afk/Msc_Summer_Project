# PubMedQA Label Heuristic Evaluation

Heuristic extracts yes/no/maybe from each generated clean answer, then uses majority vote per example.
This is a proxy check, not final factuality evaluation.

- examples: 50
- majority-vote accuracy: 0.660 (33/50)
- mean per-sample vote accuracy over known votes: 0.602
- unknown generated labels: 61/250
- AUROC incorrect by normalized discrete SE: 0.553
- AUROC incorrect by normalized weighted SE: 0.543
- AUROC incorrect by predictive entropy: 0.611

## Correctness by SE Band
- low SE (<0.1): n=12, majority accuracy=0.750
- mid SE: n=32, majority accuracy=0.625
- high SE (>=0.7): n=6, majority accuracy=0.667

## Likely Wrong Examples by Majority Vote
- 10811329: gold=yes, majority=unknown, preds=['unknown', 'unknown', 'unknown', 'unknown', 'unknown'], H_norm=0.828, clusters=[2, 1, 1, 1]
- 11035130: gold=yes, majority=maybe, preds=['maybe', 'no', 'unknown', 'maybe', 'unknown'], H_norm=0.828, clusters=[1, 1, 2, 1]
- 10473855: gold=no, majority=maybe, preds=['maybe', 'yes', 'maybe', 'no', 'yes'], H_norm=0.655, clusters=[2, 2, 1]
- 10173769: gold=yes, majority=no, preds=['no', 'no', 'maybe', 'no', 'no'], H_norm=0.590, clusters=[1, 1, 3]
- 10340286: gold=maybe, majority=no, preds=['no', 'no', 'no', 'no', 'no'], H_norm=0.590, clusters=[3, 1, 1]
- 10577397: gold=yes, majority=no, preds=['no', 'no', 'unknown', 'no', 'no'], H_norm=0.590, clusters=[3, 1, 1]
- 11034241: gold=maybe, majority=no, preds=['no', 'maybe', 'unknown', 'no', 'no'], H_norm=0.590, clusters=[3, 1, 1]
- 11149643: gold=yes, majority=no, preds=['unknown', 'no', 'no', 'no', 'unknown'], H_norm=0.590, clusters=[3, 1, 1]
- 10575390: gold=no, majority=yes, preds=['yes', 'yes', 'yes', 'no', 'yes'], H_norm=0.418, clusters=[2, 3]
- 11079675: gold=yes, majority=no, preds=['no', 'unknown', 'unknown', 'no', 'no'], H_norm=0.418, clusters=[3, 2]
- 10430303: gold=yes, majority=no, preds=['no', 'unknown', 'no', 'unknown', 'no'], H_norm=0.311, clusters=[4, 1]
- 10548670: gold=yes, majority=no, preds=['no', 'no', 'no', 'no', 'no'], H_norm=0.311, clusters=[4, 1]