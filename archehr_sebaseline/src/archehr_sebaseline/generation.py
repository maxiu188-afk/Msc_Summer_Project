"""HuggingFace text generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Protocol


class TextGenerator(Protocol):
    def generate(self, prompt: str, *, sample_index: int = 0) -> str:
        """Generate one answer for a prompt."""


class ScoredTextGenerator(TextGenerator, Protocol):
    def generate_with_scores(self, prompt: str, *, sample_index: int = 0) -> dict[str, Any]:
        """Generate one answer and token-level uncertainty details."""


@dataclass(frozen=True)
class GenerationConfig:
    model_name: str = "sshleifer/tiny-gpt2"
    num_samples: int = 3
    max_new_tokens: int = 48
    temperature: float = 0.8
    top_p: float = 0.9
    seed: int = 13
    device: str = "cpu"
    max_input_tokens: int = 512
    local_files_only: bool = False
    torch_dtype: str | None = None
    trust_remote_code: bool = False

    def validate(self) -> None:
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1].")
        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive.")


class MissingGenerationDependency(RuntimeError):
    """Raised when generation dependencies are unavailable."""


def _load_transformers() -> tuple[Any, Any, Any]:
    cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "torchinductor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise MissingGenerationDependency(
            "HuggingFace generation requires torch and transformers. "
            "Install the project environment from requirements.txt."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, set_seed


def _resolve_torch_dtype(torch: Any, dtype_name: str | None) -> Any | None:
    if dtype_name is None or dtype_name == "auto":
        return "auto" if dtype_name == "auto" else None
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


class HuggingFaceCausalLMGenerator:
    """HuggingFace causal-LM generator used by model-backed levels."""

    def __init__(self, config: GenerationConfig):
        config.validate()
        self.config = config
        torch, auto_model, auto_tokenizer, set_seed = _load_transformers()
        self._torch = torch
        self._set_seed = set_seed
        model_kwargs: dict[str, Any] = {
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        }
        torch_dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = auto_tokenizer.from_pretrained(
            config.model_name,
            local_files_only=config.local_files_only,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = auto_model.from_pretrained(config.model_name, **model_kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(config.device)
        self.model.eval()

    def generate(self, prompt: str, *, sample_index: int = 0) -> str:
        return self.generate_with_scores(prompt, sample_index=sample_index)["raw_answer"]

    def generate_with_scores(self, prompt: str, *, sample_index: int = 0) -> dict[str, Any]:
        self._set_seed(self.config.seed + sample_index)
        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        encoded = {key: value.to(self.config.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[-1]

        with self._torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        output_ids = outputs.sequences
        new_token_ids = output_ids[0][input_length:]
        answer = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        token_logprobs: list[float] = []
        token_entropies: list[float | None] = []

        if len(new_token_ids) > 0 and outputs.scores:
            transition_scores = self.model.compute_transition_scores(
                output_ids,
                outputs.scores,
                normalize_logits=True,
            )
            token_logprobs = [
                float(score)
                for score in transition_scores[0][: len(new_token_ids)].detach().cpu().tolist()
            ]
            for score in outputs.scores[: len(new_token_ids)]:
                log_probs = self._torch.nn.functional.log_softmax(score[0].float(), dim=-1)
                probs = self._torch.exp(log_probs)
                entropy_terms = self._torch.where(
                    self._torch.isfinite(log_probs),
                    probs * log_probs,
                    self._torch.zeros_like(log_probs),
                )
                entropy_value = float(-entropy_terms.sum().detach().cpu().item())
                token_entropies.append(entropy_value if math.isfinite(entropy_value) else None)

        generated_token_ids = [int(token_id) for token_id in new_token_ids.detach().cpu().tolist()]
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_token_ids)
        summary = summarize_token_scores(token_logprobs, token_entropies)
        return {
            "raw_answer": answer.strip() or "[EMPTY_GENERATION]",
            "generated_token_ids": generated_token_ids,
            "generated_tokens": generated_tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            **summary,
        }


class HuggingFaceTinyGenerator(HuggingFaceCausalLMGenerator):
    """Backward-compatible alias for the causal-LM generator."""


class StaticGenerator:
    """Test helper that mimics a text generator without model dependencies."""

    def __init__(self, answers: list[str]):
        if not answers:
            raise ValueError("StaticGenerator requires at least one answer.")
        self.answers = answers

    def generate(self, prompt: str, *, sample_index: int = 0) -> str:
        del prompt
        return self.answers[sample_index % len(self.answers)]

    def generate_with_scores(self, prompt: str, *, sample_index: int = 0) -> dict[str, Any]:
        answer = self.generate(prompt, sample_index=sample_index)
        tokens = answer.split() or ["[EMPTY_GENERATION]"]
        token_logprobs = [-0.5 for _ in tokens]
        token_entropies: list[float | None] = [0.25 for _ in tokens]
        return {
            "raw_answer": answer,
            "generated_token_ids": list(range(len(tokens))),
            "generated_tokens": tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            **summarize_token_scores(token_logprobs, token_entropies),
        }


def summarize_token_scores(
    token_logprobs: list[float],
    token_entropies: list[float | None] | None = None,
) -> dict[str, Any]:
    token_entropies = token_entropies or []
    finite_logprobs = [value for value in token_logprobs if math.isfinite(value)]
    finite_entropies = [
        value for value in token_entropies
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    num_generated_tokens = len(token_logprobs)
    sequence_logprob = sum(finite_logprobs) if finite_logprobs else None
    mean_token_logprob = (
        sequence_logprob / len(finite_logprobs)
        if sequence_logprob is not None and finite_logprobs else None
    )
    mean_token_entropy = (
        sum(finite_entropies) / len(finite_entropies) if finite_entropies else None
    )
    return {
        "num_generated_tokens": num_generated_tokens,
        "sequence_logprob": sequence_logprob,
        "mean_token_logprob": mean_token_logprob,
        "sequence_nll": -sequence_logprob if sequence_logprob is not None else None,
        "normalized_nll": -mean_token_logprob if mean_token_logprob is not None else None,
        "mean_token_entropy": mean_token_entropy,
        "max_token_entropy": max(finite_entropies) if finite_entropies else None,
    }


def generate_answer_records(
    prompt_records: list[dict[str, Any]],
    generator: TextGenerator,
    *,
    num_samples: int,
    model_name: str,
    generation_level: str = "generation",
    include_token_scores: bool = False,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    records: list[dict[str, Any]] = []
    for prompt_record in prompt_records:
        for sample_id in range(num_samples):
            generation_details: dict[str, Any] = {}
            if include_token_scores and hasattr(generator, "generate_with_scores"):
                generation_details = generator.generate_with_scores(  # type: ignore[attr-defined]
                    prompt_record["prompt"], sample_index=sample_id
                )
                raw_answer = generation_details.pop("raw_answer")
            else:
                raw_answer = generator.generate(prompt_record["prompt"], sample_index=sample_id)
            record = {
                "example_id": prompt_record["example_id"],
                "sample_id": sample_id,
                "raw_answer": raw_answer,
                "model_name": model_name,
                "generation_level": generation_level,
                **generation_details,
            }
            if prompt_record.get("dataset") is not None:
                record["dataset"] = prompt_record["dataset"]
            if prompt_record.get("split") is not None:
                record["split"] = prompt_record["split"]
            records.append(record)
    return records
