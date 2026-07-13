"""HuggingFace text generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Any, Callable, Protocol


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


def _load_transformers() -> tuple[Any, Any, Any, Any, Any, Any]:
    cache_dir = Path(__file__).resolve().parents[2] / ".cache" / "torchinductor"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(cache_dir))

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, set_seed

        try:
            from transformers import Gemma3ForConditionalGeneration
        except ImportError:
            Gemma3ForConditionalGeneration = None
    except ImportError as exc:
        raise MissingGenerationDependency(
            "HuggingFace generation requires torch and transformers. "
            "Install the project environment from requirements.txt."
        ) from exc
    return torch, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration, set_seed


def _resolve_torch_dtype(torch: Any, dtype_name: str | None) -> Any | None:
    if dtype_name is None or dtype_name == "auto":
        return "auto" if dtype_name == "auto" else None
    dtype = getattr(torch, dtype_name, None)
    if dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return dtype


def _is_gemma3_model(model_name: str) -> bool:
    normalized = model_name.lower().replace("_", "-")
    return "gemma-3" in normalized


class HuggingFaceCausalLMGenerator:
    """HuggingFace causal-LM generator used by model-backed levels."""

    def __init__(self, config: GenerationConfig):
        config.validate()
        self.config = config
        (
            torch,
            auto_model,
            auto_processor,
            auto_tokenizer,
            gemma3_model,
            set_seed,
        ) = _load_transformers()
        self._torch = torch
        self._set_seed = set_seed
        self.processor = None
        self.tokenizer = None
        model_kwargs: dict[str, Any] = {
            "local_files_only": config.local_files_only,
            "trust_remote_code": config.trust_remote_code,
        }
        torch_dtype = _resolve_torch_dtype(torch, config.torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        if _is_gemma3_model(config.model_name):
            if gemma3_model is None:
                raise MissingGenerationDependency(
                    "Gemma 3 requires transformers>=4.50.0. "
                    "Upgrade the server environment with: python -m pip install -U 'transformers>=4.50.0'"
                )
            self.processor = auto_processor.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is not None:
                self.tokenizer.truncation_side = "left"
            self.model = gemma3_model.from_pretrained(config.model_name, **model_kwargs)
        else:
            self.tokenizer = auto_tokenizer.from_pretrained(
                config.model_name,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
            self.tokenizer.truncation_side = "left"
            self.model = auto_model.from_pretrained(config.model_name, **model_kwargs)

        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(config.device)
        self.model.eval()

    def generate(self, prompt: str, *, sample_index: int = 0) -> str:
        return self.generate_with_scores(prompt, sample_index=sample_index)["raw_answer"]

    def generate_with_scores(self, prompt: str, *, sample_index: int = 0) -> dict[str, Any]:
        self._set_seed(self.config.seed + sample_index)
        encoded = self._encode_prompt(prompt)
        input_length = encoded["input_ids"].shape[-1]

        with self._torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self._pad_token_id(),
                return_dict_in_generate=True,
                output_scores=True,
            )

        output_ids = outputs.sequences
        new_token_ids = output_ids[0][input_length:]
        answer = self._decode(new_token_ids)
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
        generated_tokens = self._convert_ids_to_tokens(generated_token_ids)
        summary = summarize_token_scores(token_logprobs, token_entropies)
        return {
            "raw_answer": answer.strip() or "[EMPTY_GENERATION]",
            "generated_token_ids": generated_token_ids,
            "generated_tokens": generated_tokens,
            "token_logprobs": token_logprobs,
            "token_entropies": token_entropies,
            **summary,
        }

    def _encode_prompt(self, prompt: str) -> dict[str, Any]:
        if self.processor is not None:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ]
            encoded = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return self._truncate_and_move(encoded)

        if self.tokenizer is not None and getattr(self.tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": prompt}]
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            return self._truncate_and_move(encoded)

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_input_tokens,
        )
        return self._move_to_device(encoded)

    def _truncate_and_move(self, encoded: dict[str, Any]) -> dict[str, Any]:
        for key in ("input_ids", "attention_mask"):
            value = encoded.get(key)
            if value is not None and value.shape[-1] > self.config.max_input_tokens:
                encoded[key] = value[:, -self.config.max_input_tokens :]
        return self._move_to_device(encoded)

    def _move_to_device(self, encoded: dict[str, Any]) -> dict[str, Any]:
        moved = {}
        for key, value in encoded.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self.config.device)
            else:
                moved[key] = value
        return moved

    def _decode(self, token_ids: Any) -> str:
        if self.processor is not None and hasattr(self.processor, "decode"):
            return self.processor.decode(token_ids, skip_special_tokens=True)
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
        if self.tokenizer is None:
            return [str(token_id) for token_id in token_ids]
        return self.tokenizer.convert_ids_to_tokens(token_ids)

    def _pad_token_id(self) -> int | None:
        if self.tokenizer is not None:
            return self.tokenizer.pad_token_id
        return getattr(getattr(self.model, "generation_config", None), "pad_token_id", None)


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
    progress_callback: Callable[[int, int, str, int], None] | None = None,
) -> list[dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    records: list[dict[str, Any]] = []
    total_generations = len(prompt_records) * num_samples
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
            if progress_callback is not None:
                progress_callback(
                    len(records),
                    total_generations,
                    str(prompt_record["example_id"]),
                    sample_id,
                )
    return records
