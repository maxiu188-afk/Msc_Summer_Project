# Archived and Unused Material

This is the single repository-wide holding area for material that is no longer
part of the active research workflow but is retained for provenance or possible
future inspection. Moving a file here means **do not use it as current evidence
or as an active run instruction**.

## Layout

- `docs/project_history/`: superseded plans, handoff notes, and server guides.
- `docs/historical_results/`: result reports that are not comparable with the
  current BioASQ/PubMedQA protocol.
- `docs/literature_extracts/`: generated paper-text snapshots and old notes.
- `data/`: obsolete or superseded local data. Generated data remain ignored by
  Git; record their origin in `data/README.md` before placing them here.
- `results/`: obsolete generated outputs and server-result snapshots. Large or
  regenerated files remain ignored by Git.

## Rules for future use

1. Do not delete research artifacts merely because they are no longer active;
   move them into the matching subdirectory here.
2. Add a short reason, source path, and archive date to the nearest README.
3. Never place credentials, model weights, virtual environments, caches, or
   restricted datasets here.
4. Current documentation belongs in `archehr_sebaseline/docs/`; only one active
   document should own each protocol or result.
5. Archived commands may contain stale paths and must be revalidated before use.
