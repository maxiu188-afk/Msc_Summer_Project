# Phase 2 BioASQ dataset split

## Decision

Phase 2 uses the full official **BioASQ Task B training13b** corpus, filtered
only to the three question types already established by the active protocol:
`factoid`, `list`, and `summary`. `yesno` remains out of scope: it was excluded
from the Phase-1 no-evidence medical-UQ protocol and would introduce a different
answer format and evaluation target.

The Phase-1 1,000-question set is **not** the Phase-2 dataset. It is an
already-observed baseline/reference cohort. Because its results informed the
choice of a P(True)-Probe, it cannot be used for validation or final test. Its
IDs are retained in the full corpus and assigned to the training side with a
`phase1_reference=true` manifest field. Thus every eligible BioASQ question is
used, while the final test set contains no Phase-1 question.

## Fixed partition

1. Load every eligible question from the pinned `training13b.json` source.
2. Read the actual Phase-1 `examples.jsonl` artifact and force those IDs, plus
   any exact duplicate-question group containing one, to `train`. If that
   generated artifact is unavailable, reconstruct the IDs only from the pinned
   raw `training13b.json` using the recorded seed `20260718` and quotas
   factoid/list/summary = 480/320/200; the output records this fallback.
3. Set the 80/10/10 quota independently from the *full* count of each BioASQ
   type, charge the Phase-1 reference IDs to that type's training quota, then
   allocate the remaining unobserved questions with a fixed seed:

   | Split | Target fraction of each full question type | Role |
   | --- | ---: | --- |
   | Train | 80% | Fit the probe to hidden-state features and blind P(True) target. |
   | Validation | 10% | Choose layer, token pooling, regularization, and stopping rule. |
   | Test | 10% | One final frozen evaluation only. |

The partitions preserve the factoid/list/summary proportions separately rather
than relying on a pooled random split. Integer quotas are calculated from each
type's complete count, so the final counts differ by at most rounding or an
exact-duplicate group.
The generated summary records the real counts; do not report planned counts as
observed counts.

## Generated manifest (2026-07-19)

The official `training13b.json` now available under the project-local `data/`
directory has SHA-256:

```text
38f1a4a54e90d6ee8b668f37f8d3d0be2df36dd4c89b26fa1ccceceb3861eaad
```

It contains 5,389 questions in total. The active three-type source contains
3,930 questions: 1,600 factoid, 1,047 list, and 1,283 summary. The Phase-1
artifact is not retained locally, so its 1,000 IDs were explicitly reconstructed
from this pinned source with seed `20260718` and quotas 480/320/200. The
generated manifest is ignored local data at:

```text
archehr_sebaseline/artifacts/phase2_bioasq_training13b_split/
```

| Split | Factoid | List | Summary | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,280 | 837 | 1,027 | 3,144 |
| Validation | 160 | 105 | 128 | 393 |
| Test | 160 | 105 | 128 | 393 |

All 1,000 reconstructed Phase-1 IDs are in train; no exact-question group
crosses a split. The generated JSON summary retains the reconstruction method,
selection seed, quotas, raw-data hash, and actual counts.

## Leakage controls

- Partition at the **question ID** level. All generations, low-temperature
  answers, hidden states, P(True) targets, SE scores, and Claude labels for one
  question must stay in its assigned split.
- Group identical normalized question text within a type before splitting. A
  duplicate can therefore never appear in both train and test.
- The split manifest records the SHA-256 of both the raw `training13b.json` and
  the Phase-1 `examples.jsonl`, which pins the exact inputs.
- Do not use Claude `correct`/`incorrect` labels, gold answers, exact answers,
  or ideal answers as probe input features. They remain evaluation metadata.
- The direct `p_true_blind_uncertainty` value is the regression target. It may
  be generated for test only after the probe design is frozen, then used solely
  to measure fidelity and cost against direct P(True).

## Reproducible manifest command

Prefer the recorded Phase-1 artifact when it is available:

```bash
cd archehr_sebaseline
python scripts/make_bioasq_phase2_splits.py \
  --data_path ../data/BioASQ-training13b/training13b.json \
  --phase1_examples ../outputs/bioasq_medical_uq_gemma3_12b_1000x10_phase1_seed31/examples.jsonl \
  --output_dir artifacts/phase2_bioasq_training13b_split
```

The command writes `phase2_bioasq_split_manifest.jsonl` and
`phase2_bioasq_split_summary.json`. Copy these small provenance files to the
run directory or commit them only after checking that the paths and SHA-256s
refer to the intended official dataset. The raw dataset and generated model
artifacts remain untracked.

The two Phase-1 seeds have the same question selection. Use one seed's
`examples.jsonl` only to obtain the IDs; the source-data SHA-256 check protects
against an accidental dataset-version mismatch.

When only the pinned official source is available locally, reconstruct the
known Phase-1 selection explicitly rather than silently assuming it:

```bash
python scripts/make_bioasq_phase2_splits.py \
  --data_path ../data/BioASQ-training13b/training13b.json \
  --reconstruct_phase1_reference \
  --output_dir artifacts/phase2_bioasq_training13b_split
```
