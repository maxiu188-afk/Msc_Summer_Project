# Literature Notes for Implementation

These notes summarize implementation-relevant points from local Semantic Entropy and ArchEHR-QA references. They are design notes only and do not require reading restricted ArchEHR-QA data.

## Semantic Entropy

Implementation implications:

- Estimate uncertainty over meanings, not surface strings.
- Sample multiple answers for the same input, cluster semantically equivalent answers, and compute entropy over semantic clusters.
- Semantic equivalence is operationalized with bidirectional entailment: answer A and answer B are grouped when each entails the other under the chosen entailment procedure.
- When token log probabilities are available, estimate semantic-cluster probability mass with sequence likelihoods.
- Keep the count-based entropy estimator as a fallback and sanity check.
- Keep raw generations, cleaned generations, cluster records, and entropy scores as separate artifacts.

## ArchEHR-QA

Implementation implications:

- Each example is centered on grounded patient question answering from EHR note excerpts.
- Inputs include a patient question, a clinician-interpreted question, and sentence-level evidence items.
- Outputs should be patient-facing natural language answers with sentence-level citations where feasible.
- Citation and answer text should be handled separately: raw answers preserve citations, cleaned answers remove citation markers for semantic clustering.
- Evaluation should eventually compare uncertainty scores against answer quality, factuality, relevance, and citation quality.

## Current Baseline Implications

- Level 4 uses PubMedQA as an approved public pilot dataset before restricted ArchEHR-QA work.
- Qwen2.5-7B-Instruct plus NLI clustering now produces complete 50x5 pilot artifacts.
- The next implementation step is a maintained evaluation script for PubMedQA yes/no/maybe correctness proxies, followed by AUROC and selective-prediction analysis.
