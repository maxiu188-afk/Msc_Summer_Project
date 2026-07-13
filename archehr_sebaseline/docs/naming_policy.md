# Active Naming Policy: BioASQ SE Track

Effective date: 2026-07-13.

## Decision

ArchEHR-QA has been retired as the main experimental dataset. New work belongs
to the BioASQ Semantic Entropy track, so **new descriptive names must not use
the `archehr` prefix** unless the artifact genuinely concerns the historical
ArchEHR-QA diagnostic path.

This is a forward-only policy. It improves the clarity of new work without
renaming established code paths or historical evidence.

## Use for new artifacts

Use `bioasq_se` for newly created BioASQ-specific names where a dataset-neutral
Level 4 name is not already established.

| Artifact | Preferred new pattern | Example |
| --- | --- | --- |
| Experiment directory | `bioasq_se_<split>_<model>_<examples>x<samples>` | `bioasq_se_golden_gemma3_12b_100x10` |
| Seed repeat | add `_seed<seed>` | `bioasq_se_train_gemma3_12b_50x10_seed47` |
| Server upload archive | `bioasq_se_runpod_<gpu>_<yyyymmdd>.tar.gz` | `bioasq_se_runpod_a40_20260713.tar.gz` |
| Evaluation directory | `bioasq_eval/` | `bioasq_eval/bioasq_eval_summary.json` |
| New report/document title | `BioASQ Semantic Entropy ...` | `BioASQ Semantic Entropy Batch Report` |

Existing `runpod_bioasq_*` output names remain acceptable because they already
identify BioASQ and are used by active scripts. Do not create new generic
names such as `archehr_*` for BioASQ runs, packages, reports, or result tables.

## Do not rename retroactively

The following names remain unchanged because they are real historical
ArchEHR-QA artifacts or active import/path contracts:

- `archehr_sebaseline/`, `src/archehr_sebaseline/`, and all Python imports.
- Existing `archehr_*` scripts, test fixtures, outputs, and diagnostic reports.
- Historical ArchEHR-QA plans, handoff documents, and literature extracts.
- Existing upload archives, including
  `archehr_sebaseline_runpod_a40_network_20260713.tar.gz`.

No path, module, script, archive, or completed result is renamed solely to
apply this policy. This avoids breaking imports, commands, cached Runpod
locations, and result provenance.

## Scope boundary

`ArchEHR-QA` is still the correct name whenever documenting its engineering
diagnostic implementation or its historical results. The policy changes only
the naming of **new BioASQ-main-track work**, not the factual description of
earlier work.
