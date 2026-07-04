"""NLI-based semantic clustering for Semantic Entropy."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol


ENTAILMENT = "entailment"
NEUTRAL = "neutral"
CONTRADICTION = "contradiction"


class EntailmentScorer(Protocol):
    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        *,
        question: str | None = None,
    ) -> str:
        """Return entailment, neutral, or contradiction."""


@dataclass(frozen=True)
class NLIConfig:
    model_name: str = "microsoft/deberta-v2-xlarge-mnli"
    device: str = "cpu"
    max_input_tokens: int = 512
    local_files_only: bool = False
    torch_dtype: str | None = None
    trust_remote_code: bool = False
    strict_entailment: bool = False
    condition_on_question: bool = True


class HuggingFaceNLIScorer:
    """Sequence-classification NLI scorer for bidirectional entailment clustering."""

    def __init__(self, config: NLIConfig):
        self.config = config
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "NLI clustering requires torch and transformers. Install the project "
                "environment from requirements.txt."
            ) from exc

        self._torch = torch
        model_kwargs: dict[str, Any] = {
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        }
        if config.torch_dtype and config.torch_dtype != "auto":
            dtype = getattr(torch, config.torch_dtype, None)
            if dtype is None:
                raise ValueError(f"Unsupported NLI torch dtype: {config.torch_dtype}")
            model_kwargs["torch_dtype"] = dtype
        elif config.torch_dtype == "auto":
            model_kwargs["torch_dtype"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            **model_kwargs,
        )
        self.model.to(config.device)
        self.model.eval()
        self.id2label = {
            int(idx): str(label).lower()
            for idx, label in getattr(self.model.config, "id2label", {}).items()
        }

    def _format_text(self, text: str, *, question: str | None) -> str:
        text = text.strip()
        if self.config.condition_on_question and question:
            return f"Question: {question.strip()}\nAnswer: {text}"
        return text

    def _label_from_index(self, index: int) -> str:
        label = self.id2label.get(index, "").lower()
        if "entail" in label:
            return ENTAILMENT
        if "contrad" in label:
            return CONTRADICTION
        if "neutral" in label:
            return NEUTRAL

        # Common MNLI fallback order used by DeBERTa MNLI checkpoints.
        if index == 0:
            return CONTRADICTION
        if index == 1:
            return NEUTRAL
        if index == 2:
            return ENTAILMENT
        return NEUTRAL

    def check_implication(
        self,
        premise: str,
        hypothesis: str,
        *,
        question: str | None = None,
    ) -> str:
        premise_text = self._format_text(premise, question=question)
        hypothesis_text = self._format_text(hypothesis, question=question)
        inputs = self.tokenizer(
            premise_text,
            hypothesis_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self.model(**inputs)
        label_index = int(self._torch.argmax(outputs.logits, dim=-1).detach().cpu().item())
        return self._label_from_index(label_index)


def are_bidirectionally_equivalent(
    text1: str,
    text2: str,
    scorer: EntailmentScorer,
    *,
    question: str | None = None,
    strict_entailment: bool = False,
) -> bool:
    implication_1 = scorer.check_implication(text1, text2, question=question)
    implication_2 = scorer.check_implication(text2, text1, question=question)

    if strict_entailment:
        return implication_1 == ENTAILMENT and implication_2 == ENTAILMENT

    implications = {implication_1, implication_2}
    return CONTRADICTION not in implications and not (
        implication_1 == NEUTRAL and implication_2 == NEUTRAL
    )


def cluster_by_bidirectional_entailment(
    generations: list[dict[str, Any]],
    *,
    scorer: EntailmentScorer,
    examples_by_id: dict[str, dict[str, Any]] | None = None,
    strict_entailment: bool = False,
) -> list[dict[str, Any]]:
    """Cluster answers by semantic equivalence using bidirectional NLI."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for generation in generations:
        grouped[str(generation["example_id"])].append(generation)

    cluster_records = []
    for example_id in sorted(grouped):
        example_generations = sorted(grouped[example_id], key=lambda item: item["sample_id"])
        example = (examples_by_id or {}).get(example_id, {})
        question = example.get("question") or example.get("patient_question")
        clusters: list[dict[str, Any]] = []
        semantic_ids: list[int] = []

        for generation in example_generations:
            clean_answer = generation.get("clean_answer")
            if clean_answer is None:
                raise ValueError("NLI clustering requires clean_answer on each generation.")

            assigned_cluster_id = None
            for cluster in clusters:
                if are_bidirectionally_equivalent(
                    str(cluster["representative_answer"]),
                    str(clean_answer),
                    scorer,
                    question=str(question) if question else None,
                    strict_entailment=strict_entailment,
                ):
                    assigned_cluster_id = int(cluster["cluster_id"])
                    break

            if assigned_cluster_id is None:
                assigned_cluster_id = len(clusters)
                clusters.append(
                    {
                        "cluster_id": assigned_cluster_id,
                        "cluster_key": str(clean_answer),
                        "representative_answer": str(clean_answer),
                        "sample_ids": [],
                    }
                )

            semantic_ids.append(assigned_cluster_id)
            clusters[assigned_cluster_id]["sample_ids"].append(generation["sample_id"])

        for cluster in clusters:
            cluster["cluster_size"] = len(cluster["sample_ids"])

        cluster_records.append(
            {
                "example_id": example_id,
                "num_samples": len(example_generations),
                "clustering_method": "nli_bidirectional_entailment",
                "strict_entailment": strict_entailment,
                "semantic_ids": semantic_ids,
                "cluster_sizes": [cluster["cluster_size"] for cluster in clusters],
                "clusters": clusters,
            }
        )

    return cluster_records
