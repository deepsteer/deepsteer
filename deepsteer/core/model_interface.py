"""Model interface: abstract base, white-box (HuggingFace), and API wrappers."""

from __future__ import annotations

import enum
import gc
import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from deepsteer.core.types import AccessTier, GenerationResult, ModelInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Architecture detection
# ---------------------------------------------------------------------------


class ModelFamily(str, enum.Enum):
    """Known model architecture families."""

    OLMO = "olmo"
    OLMOE = "olmoe"
    LLAMA = "llama"
    QWEN = "qwen"
    MISTRAL = "mistral"
    GPT2 = "gpt2"
    UNKNOWN = "unknown"


_CONFIG_TYPE_TO_FAMILY: dict[str, ModelFamily] = {
    "olmo3": ModelFamily.OLMO,
    "olmo2": ModelFamily.OLMO,
    "olmo": ModelFamily.OLMO,
    "olmoe": ModelFamily.OLMOE,
    "llama": ModelFamily.LLAMA,
    "qwen2": ModelFamily.QWEN,
    "qwen3": ModelFamily.QWEN,
    "mistral": ModelFamily.MISTRAL,
    "gpt2": ModelFamily.GPT2,
    "gpt_neo": ModelFamily.GPT2,
    "gpt_neox": ModelFamily.GPT2,
}


class UnsupportedArchitectureError(Exception):
    """Raised when a method is called on an unsupported architecture."""


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


def _auto_dequant_config(model_name_or_path: str, revision: str | None):
    """``Mxfp4Config(dequantize=True)`` for an mxfp4 repo (e.g. GPT-OSS), else None.

    mxfp4 inference needs triton kernels that are not always present, and quantized
    experts cannot be cleanly hooked/edited; dequantizing to the load dtype is the
    portable default for activation/probing work. Runs only when the caller passes
    no explicit ``quantization_config`` (which always takes precedence).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
        qc = getattr(cfg, "quantization_config", None)
        method = qc.get("quant_method") if isinstance(qc, dict) else getattr(qc, "quant_method", None)
        if method == "mxfp4":
            from transformers import Mxfp4Config
            logger.info("mxfp4 repo detected (%s); loading dequantized.", model_name_or_path)
            return Mxfp4Config(dequantize=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("mxfp4 auto-detect skipped for %s: %s", model_name_or_path, e)
    return None


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
        quantization_config: Any | None = None,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = _resolve_device(device)

        # Auto-dequantize mxfp4 repos (GPT-OSS) unless the caller pinned a config.
        # mxfp4 dequantizes to bf16, so force the whole model to bf16 when no dtype
        # was requested: the fp16 default would mismatch the dequantized bf16
        # experts in the MoE grouped_mm (Half != BFloat16).
        if quantization_config is None:
            quantization_config = _auto_dequant_config(model_name_or_path, revision)
            if quantization_config is not None and torch_dtype is None:
                torch_dtype = torch.bfloat16
        self._dtype = torch_dtype or _default_dtype(self._device)

        logger.info(
            "Loading %s on %s (dtype=%s, revision=%s)",
            model_name_or_path, self._device, self._dtype, revision,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision,
        )
        # ``quantization_config`` is optional and defaults to None (no change for
        # unquantized models). It is the clean path for loading a quantized repo
        # under an explicit policy, e.g. ``Mxfp4Config(dequantize=True)`` to force
        # GPT-OSS's mxfp4 experts to bf16 for a clean intervention (Paper 7 0d).
        if self._device == "mps":
            # MPS can fail with device_map="mps" on large models due to
            # single-buffer allocation limits.  Load to CPU first then move.
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
                revision=revision,
                quantization_config=quantization_config,
            )
            self._model = self._model.to("mps")
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=self._dtype,
                device_map=self._device if self._device != "cpu" else None,
                revision=revision,
                quantization_config=quantization_config,
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

    @property
    def model_family(self) -> ModelFamily:
        """Detected architecture family."""
        config_type = getattr(self._model.config, "model_type", "")
        return _CONFIG_TYPE_TO_FAMILY.get(config_type, ModelFamily.UNKNOWN)

    def _validate_family(
        self,
        supported: set[ModelFamily],
        method_name: str,
    ) -> None:
        """Check that this model's architecture is supported by *method_name*.

        If the family is UNKNOWN, log a warning. If it is known but not in
        *supported*, raise ``UnsupportedArchitectureError``.
        """
        family = self.model_family
        if family == ModelFamily.UNKNOWN:
            config_type = getattr(self._model.config, "model_type", "?")
            logger.warning(
                "Architecture %r has not been tested with %s; results may be unreliable.",
                config_type, method_name,
            )
        elif family not in supported:
            raise UnsupportedArchitectureError(
                f"{method_name} is not supported for {family.value}. "
                f"Supported: {', '.join(f.value for f in sorted(supported, key=lambda f: f.value))}."
            )

    # -- Layer introspection -------------------------------------------------

    def _unwrap_model(self) -> torch.nn.Module:
        """Unwrap PEFT/DeepSpeed wrappers to get the underlying HuggingFace model.

        PEFT wraps models as ``PeftModelForCausalLM`` with an extra
        ``base_model`` attribute.  This method traverses that wrapper so
        layer introspection and hook registration work regardless of
        whether LoRA adapters are active.
        """
        model = self._model
        # PEFT: PeftModelForCausalLM -> base_model (LoraModel) -> model (original)
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model") and model is not self._model:
            model = model.model
        return model

    def _detect_n_layers(self) -> int:
        """Detect number of decoder layers by probing known attribute paths."""
        base = self._unwrap_model()
        # OLMo / Llama / Mistral: model.model.layers
        inner = getattr(base, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return len(layers)
        # Direct: base.layers (when already unwrapped)
        layers = getattr(base, "layers", None)
        if layers is not None:
            return len(layers)
        # GPT-2 / GPT-Neo: model.transformer.h
        transformer = getattr(base, "transformer", None)
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
        base = self._unwrap_model()
        inner = getattr(base, "model", None)
        if inner is not None:
            layers = getattr(inner, "layers", None)
            if layers is not None:
                return layers[layer_index]
        layers = getattr(base, "layers", None)
        if layers is not None:
            return layers[layer_index]
        transformer = getattr(base, "transformer", None)
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

    # -- Batch activation collection ----------------------------------------

    @torch.no_grad()
    def collect_batch_activations(
        self,
        texts: list[str],
        layers: list[int] | None = None,
        pooling: str = "mean",
        batch_size: int = 32,
    ) -> dict[int, Tensor]:
        """Collect activations for many texts at specified layers, in true batches.

        Texts are processed ``batch_size`` at a time with a single forward per
        batch (hooks capture all requested layers at once). Sequences are
        right-padded and pooling is attention-mask-aware. For causal decoder
        LMs, right-padding leaves each real token's hidden state identical to
        the unpadded single-text forward, so mean/last/first pooling matches the
        per-text path numerically (cosine > 0.999; small fp differences only).

        Args:
            texts: Input texts.
            layers: Layer indices. ``None`` = all layers.
            pooling: Sequence reduction — ``"mean"`` (default), ``"last"``,
                ``"first"``, or ``"none"`` (keep full per-text sequence).
            batch_size: Texts per forward pass.

        Returns:
            Mapping from layer index to a float32 CPU tensor of shape
            ``(n_texts, hidden_dim)`` if pooling != ``"none"``, or a list of
            per-text ``(seq_len_i, hidden_dim)`` tensors if ``"none"``.
        """
        if pooling not in ("mean", "last", "first", "none"):
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        if layers is None:
            layers = list(range(self._info.n_layers))  # type: ignore[arg-type]

        n_texts = len(texts)
        per_layer: dict[int, list[Tensor]] = {idx: [] for idx in layers}
        if n_texts == 0:
            return {idx: torch.empty(0) for idx in layers}

        # Right-padding keeps real-token hidden states identical to the
        # single-text forward (causal attention ignores trailing pads; default
        # position ids stay aligned). Restore the prior side afterwards.
        prev_side = getattr(self._tokenizer, "padding_side", "right")
        self._tokenizer.padding_side = "right"
        try:
            for start in range(0, n_texts, batch_size):
                batch = texts[start:start + batch_size]
                captured: dict[int, Tensor] = {}

                def _make_hook(layer_idx: int):
                    def hook(_module: torch.nn.Module, _input: Any, output: Any) -> None:
                        captured[layer_idx] = output[0] if isinstance(output, tuple) else output
                    return hook

                hooks = [self._get_layer_module(idx).register_forward_hook(_make_hook(idx))
                         for idx in layers]
                try:
                    enc = self._tokenizer(
                        batch, return_tensors="pt", padding=True, truncation=True,
                    ).to(self._device)
                    self._model(**enc)
                finally:
                    for h in hooks:
                        h.remove()

                mask = enc["attention_mask"]            # (b, L)
                lengths = mask.sum(dim=1).clamp(min=1)  # (b,)
                for layer_idx in layers:
                    h = captured[layer_idx].float()     # (b, L, hidden)
                    if pooling == "mean":
                        m = mask.unsqueeze(-1).float()
                        pooled = (h * m).sum(dim=1) / lengths.unsqueeze(-1).float()
                        per_layer[layer_idx].append(pooled.cpu())
                    elif pooling == "last":
                        idx = (lengths - 1)
                        pooled = h[torch.arange(h.size(0), device=h.device), idx]
                        per_layer[layer_idx].append(pooled.cpu())
                    elif pooling == "first":
                        per_layer[layer_idx].append(h[:, 0].cpu())
                    else:  # "none": keep each text's real tokens
                        for bi in range(h.size(0)):
                            per_layer[layer_idx].append(h[bi, : int(lengths[bi])].cpu())
                logger.info("  Collected activations: %d/%d texts",
                            min(start + batch_size, n_texts), n_texts)
        finally:
            self._tokenizer.padding_side = prev_side

        if pooling == "none":
            return {idx: per_layer[idx] for idx in layers}  # type: ignore[return-value]
        return {idx: torch.cat(per_layer[idx], dim=0) for idx in layers}

    # -- Direction operation context managers --------------------------------

    @contextmanager
    def ablate_direction(
        self,
        layer: int,
        direction: np.ndarray | Tensor,
    ):
        """Context manager: project out *direction* from layer output.

        Usage::

            with model.ablate_direction(layer=8, direction=care_dir):
                result = model.score(prompt, completion)
        """
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction)
        d = direction.to(device=self._device, dtype=self._dtype)
        d = d / (d.norm() + 1e-12)

        def _hook(_module: torch.nn.Module, _input: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            proj = (tensor @ d).unsqueeze(-1) * d
            patched = tensor - proj
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        module = self._get_layer_module(layer)
        handle = module.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    @contextmanager
    def inject_direction(
        self,
        layer: int,
        direction: np.ndarray | Tensor,
        alpha: float = 1.0,
    ):
        """Context manager: add ``alpha * direction`` to layer output.

        Usage::

            with model.inject_direction(8, care_dir, alpha=2.0):
                result = model.generate(prompt)
        """
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction)
        d = direction.to(device=self._device, dtype=self._dtype)
        d = d / (d.norm() + 1e-12)

        def _hook(_module: torch.nn.Module, _input: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            patched = tensor + alpha * d
            if isinstance(output, tuple):
                return (patched,) + output[1:]
            return patched

        module = self._get_layer_module(layer)
        handle = module.register_forward_hook(_hook)
        try:
            yield
        finally:
            handle.remove()

    # -- Memory management ---------------------------------------------------

    def release(self) -> None:
        """Free model memory. Call when done with this model."""
        if hasattr(self, "_model"):
            del self._model
        if hasattr(self, "_tokenizer"):
            del self._tokenizer
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> WhiteBoxModel:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

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
