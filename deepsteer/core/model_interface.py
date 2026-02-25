"""Model interface: abstract base, white-box (HuggingFace), and API wrappers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from deepsteer.core.types import AccessTier, GenerationResult, ModelInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> str:
    """Pick the best available device: CUDA > MPS > CPU."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_dtype(device: str) -> torch.dtype:
    """float16 on GPU/MPS, float32 on CPU."""
    if device in ("cuda", "mps") or device.startswith("cuda:"):
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ModelInterface(ABC):
    """Unified interface for all models DeepSteer evaluates."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Return metadata about this model."""

    @property
    def access_tier(self) -> AccessTier:
        """Convenience accessor for the model's access tier."""
        return self.info.access_tier

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        """Generate a completion for *prompt*."""

    @abstractmethod
    def score(self, prompt: str, completion: str) -> float:
        """Return the log-probability of *completion* given *prompt*."""

    def get_logprobs(self, prompt: str, completion: str) -> list[tuple[str, float]]:
        """Per-token log-probabilities for *completion* given *prompt*.

        Only available for models that expose token-level scoring.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support per-token log-probabilities"
        )


# ---------------------------------------------------------------------------
# White-box model (local HuggingFace weights)
# ---------------------------------------------------------------------------


class WhiteBoxModel(ModelInterface):
    """Local HuggingFace model with activation hooks for probing and patching."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        access_tier: AccessTier = AccessTier.WEIGHTS,
        checkpoint_step: int | None = None,
        revision: str | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = _resolve_device(device)
        self._dtype = torch_dtype or _default_dtype(self._device)

        logger.info(
            "Loading %s on %s (dtype=%s, revision=%s)",
            model_name_or_path, self._device, self._dtype, revision,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self._dtype,
            device_map=self._device if self._device != "cpu" else None,
            revision=revision,
        )
        if self._device == "cpu":
            self._model = self._model.to(self._device)
        self._model.eval()

        # Llama / Mistral tokenizers often lack a pad token, which causes
        # warnings during generate().  Fall back to eos_token.
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        n_layers = self._detect_n_layers()
        n_params = sum(p.numel() for p in self._model.parameters())
        self._info = ModelInfo(
            name=model_name_or_path,
            provider="huggingface",
            access_tier=access_tier,
            n_layers=n_layers,
            n_params=n_params,
            checkpoint_step=checkpoint_step,
        )
        logger.info(
            "Loaded %s: %d layers, %.1fM params",
            model_name_or_path,
            n_layers,
            n_params / 1e6,
        )

    # -- Properties ----------------------------------------------------------

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def model(self) -> torch.nn.Module:
        """The underlying ``nn.Module`` (for advanced use)."""
        return self._model

    @property
    def tokenizer(self) -> Any:
        """The HuggingFace tokenizer."""
        return self._tokenizer

    # -- Layer introspection -------------------------------------------------

    def _detect_n_layers(self) -> int:
        """Detect number of decoder layers by probing known attribute paths."""
        # OLMo / Llama / Mistral: model.model.layers
        inner = getattr(self._model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return len(layers)
        # GPT-2 / GPT-Neo: model.transformer.h
        transformer = getattr(self._model, "transformer", None)
        if transformer is not None:
            h = getattr(transformer, "h", None)
            if h is not None:
                return len(h)
        raise RuntimeError(
            f"Cannot detect layer count for {type(self._model).__name__}. "
            "Expected model.model.layers or model.transformer.h"
        )

    def _get_layer_module(self, layer_index: int) -> torch.nn.Module:
        """Return the ``nn.Module`` for decoder layer *layer_index*."""
        inner = getattr(self._model, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return layers[layer_index]
        transformer = getattr(self._model, "transformer", None)
        if transformer is not None:
            h = getattr(transformer, "h", None)
            if h is not None:
                return h[layer_index]
        raise RuntimeError(
            f"Cannot resolve layer {layer_index} for {type(self._model).__name__}"
        )

    # -- Activation capture --------------------------------------------------

    @torch.no_grad()
    def get_activations(
        self, text: str, layers: list[int] | None = None
    ) -> dict[int, Tensor]:
        """Capture hidden-state activations at specified layers.

        Args:
            text: Input text to process.
            layers: Layer indices to capture.  ``None`` means all layers.

        Returns:
            Mapping from layer index to a CPU tensor of shape
            ``(1, seq_len, hidden_dim)``.
        """
        if layers is None:
            layers = list(range(self._info.n_layers))  # type: ignore[arg-type]

        activations: dict[int, Tensor] = {}
        hooks: list[torch.utils.hooks.RemovableHook] = []

        def _make_hook(layer_idx: int):
            def hook(_module: torch.nn.Module, _input: Any, output: Any) -> None:
                # Most decoder layers return a tuple; hidden states are element 0.
                tensor = output[0] if isinstance(output, tuple) else output
                activations[layer_idx] = tensor.detach().cpu()
            return hook

        for idx in layers:
            module = self._get_layer_module(idx)
            hooks.append(module.register_forward_hook(_make_hook(idx)))

        try:
            inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
            self._model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        return activations

    def patch_activations(
        self,
        text: str,
        layer: int,
        patch_fn: Callable[[Tensor], Tensor],
        *,
        max_tokens: int = 64,
    ) -> GenerationResult:
        """Generate while applying *patch_fn* to layer *layer*'s output.

        Used for causal tracing / activation patching experiments.
        Based on: Meng et al. (2022), "Locating and Editing Factual Associations
        in GPT" (ROME); Vig et al. (2020), causal mediation analysis for LMs.
        """
        module = self._get_layer_module(layer)

        def _hook(_module: torch.nn.Module, _input: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            patched = patch_fn(tensor)
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        handle = module.register_forward_hook(_hook)
        try:
            result = self.generate(text, max_tokens=max_tokens)
        finally:
            handle.remove()
        return result

    # -- ModelInterface implementation ---------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        inputs = self._tokenizer(full_prompt, return_tensors="pt").to(self._device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        new_ids = output_ids[0, prompt_len:]
        text = self._tokenizer.decode(new_ids, skip_special_tokens=True)
        return GenerationResult(text=text, prompt=prompt)

    @torch.no_grad()
    def score(self, prompt: str, completion: str) -> float:
        full_text = prompt + completion
        inputs = self._tokenizer(full_text, return_tensors="pt").to(self._device)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        outputs = self._model(**inputs)
        # logits shape: (1, seq_len, vocab_size)
        logits = outputs.logits[0]  # (seq_len, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log-probs of completion tokens (each predicted by the preceding position)
        token_ids = inputs["input_ids"][0]
        total = 0.0
        for i in range(prompt_len, len(token_ids)):
            total += log_probs[i - 1, token_ids[i]].item()
        return total

    @torch.no_grad()
    def get_logprobs(self, prompt: str, completion: str) -> list[tuple[str, float]]:
        full_text = prompt + completion
        inputs = self._tokenizer(full_text, return_tensors="pt").to(self._device)
        prompt_ids = self._tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        outputs = self._model(**inputs)
        logits = outputs.logits[0]
        log_probs = F.log_softmax(logits, dim=-1)

        token_ids = inputs["input_ids"][0]
        result: list[tuple[str, float]] = []
        for i in range(prompt_len, len(token_ids)):
            token = self._tokenizer.decode([token_ids[i]])
            lp = log_probs[i - 1, token_ids[i]].item()
            result.append((token, lp))
        return result


# ---------------------------------------------------------------------------
# API model (Claude / GPT)
# ---------------------------------------------------------------------------


class APIModel(ModelInterface):
    """Wrapper for hosted LLM APIs (Anthropic Claude, OpenAI GPT)."""

    SUPPORTED_PROVIDERS = ("anthropic", "openai")

    def __init__(
        self,
        provider: str,
        model_id: str,
        *,
        api_key: str | None = None,
    ) -> None:
        if provider not in self.SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unsupported provider {provider!r}. "
                f"Choose from {self.SUPPORTED_PROVIDERS}"
            )
        self._provider = provider
        self._model_id = model_id
        self._api_key = api_key
        self._client: Any = None
        self._info = ModelInfo(
            name=model_id,
            provider=provider,
            access_tier=AccessTier.API,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    # -- Lazy client init ----------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if self._provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=self._api_key,  # falls back to ANTHROPIC_API_KEY env var
            )
        else:
            import openai

            self._client = openai.OpenAI(
                api_key=self._api_key,  # falls back to OPENAI_API_KEY env var
            )
        return self._client

    # -- ModelInterface implementation ---------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        system_prompt: str | None = None,
    ) -> GenerationResult:
        client = self._get_client()

        if self._provider == "anthropic":
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = client.messages.create(**kwargs)
            text = response.content[0].text
            return GenerationResult(
                text=text,
                prompt=prompt,
                finish_reason=response.stop_reason,
                metadata={"id": response.id, "usage": response.usage.model_dump()},
            )
        else:
            # OpenAI
            messages: list[dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            response = client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            choice = response.choices[0]
            return GenerationResult(
                text=choice.message.content or "",
                prompt=prompt,
                finish_reason=choice.finish_reason,
                metadata={"id": response.id},
            )

    def score(self, prompt: str, completion: str) -> float:
        if self._provider == "anthropic":
            raise NotImplementedError(
                "Anthropic API does not expose log-probabilities for scoring."
            )
        # OpenAI: use logprobs to compute score
        return sum(lp for _, lp in self.get_logprobs(prompt, completion))

    def get_logprobs(self, prompt: str, completion: str) -> list[tuple[str, float]]:
        if self._provider == "anthropic":
            raise NotImplementedError(
                "Anthropic API does not expose per-token log-probabilities."
            )
        client = self._get_client()
        full = prompt + completion
        response = client.chat.completions.create(
            model=self._model_id,
            messages=[{"role": "user", "content": full}],
            max_tokens=0,
            logprobs=True,
            echo=True,
        )
        # Parse completion token logprobs from response
        logprobs = response.choices[0].logprobs
        if logprobs is None:
            raise RuntimeError("OpenAI did not return logprobs — check model support.")
        result: list[tuple[str, float]] = []
        for token_lp in logprobs.content:
            result.append((token_lp.token, token_lp.logprob))
        return result
