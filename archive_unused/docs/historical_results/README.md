# Archived Low-Usability Results

This directory holds historical result reports that are retained for
provenance but are not active evidence for the next research decision.

## Why these results are archived

- They use BioASQ summary prompts, including grounded/evidence-conditioned
  settings that are outside the current no-evidence medical-QA UQ direction.
- Their quality targets use historical lexical, citation-aware, or three-class
  Claude labels rather than the planned binary `correct`/`incorrect` target.
- The generic `microsoft/deberta-v2-xlarge-mnli` clustering model does not meet
  the planned requirement for free, biomedical, set-aware factoid/list NLI.
- The PubMedQA Level-4 pilot is an engineering smoke test, not a suitable
  medical free-form UQ benchmark.

These materials must not be cited as current SE, P(True), or probe results.
They remain useful for reproducing prior runs and checking artifact formats.

## Contents

- `bioasq_isambard_results_20260715.md`: historical Isambard summary results,
  including the evidence/direct ablation and three-class Claude evaluation.
- `bioasq_runpod_results_20260713.md`: historical grounded RunPod batch.
- `pubmedqa_no_context_transfer_20260721.md`: completed question-only PubMedQA
  transfer archived because the official decision depends on the omitted
  article evidence; raw artifacts are retained for provenance.
- `feature_fusion_diagnostics_20260728.md`: Phase-1 P(True)/SE/NLL fusion and
  Phase-2 two-Probe fusion, archived because neither justified a new main
  method.
- `bioasq_phase1_100q_pilot.md`: superseded protocol-establishing pilot; the
  active result is the fixed 1,000-question Phase-1 benchmark.
- `../../results/server_results/`: local raw summary outputs, related analyses,
  manual-review material, and the PubMedQA Level-4 pilot.
- `../../results/local_outputs/`: local downloaded BioASQ summary run
  directories. This path is ignored by Git because it contains generated data.

## Current-document pointer

This archived report deliberately does not describe the active direction. Use
`../../../archehr_sebaseline/docs/README.md` and
`../../../RESEARCH_RESULTS_SYNTHESIS_ZH.md` for the completed three-phase
result narrative and `../../../PROJECT_CLOSEOUT.md` for the frozen scope and
reproducibility map.
