"""MoE-aware model interface for Mixture-of-Experts architectures."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from deepsteer.core.model_interface import (
    ModelFamily,
    UnsupportedArchitectureError,
    WhiteBoxModel,
)
from deepsteer.core.types import AccessTier

logger = logging.getLogger(__name__)


class MoEWhiteBoxModel(WhiteBoxModel):
    """Extension of WhiteBoxModel for Mixture-of-Experts architectures.

    Currently supports: OLMoE (allenai/OLMoE-1B-7B-0924).
    """

    SUPPORTED_MOE_FAMILIES = {ModelFamily.OLMOE}

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        family = self.model_family
        if family == ModelFamily.UNKNOWN:
            config_type = getattr(self._model.config, "model_type", "?")
            logger.warning(
                "Architecture %r has not been tested with MoEWhiteBoxModel; "
                "results may be unreliable.",
                config_type,
            )
        elif family not in self.SUPPORTED_MOE_FAMILIES:
            raise UnsupportedArchitectureError(
                f"MoEWhiteBoxModel does not yet support {family.value}. "
                f"Currently supported: {', '.join(f.value for f in self.SUPPORTED_MOE_FAMILIES)}. "
                f"To add support, implement _get_expert_modules() for this architecture."
            )
        self._n_experts = self._detect_n_experts()

    @property
    def n_experts(self) -> int:
        """Number of experts per layer."""
        return self._n_experts

    def _detect_n_experts(self) -> int:
        """Detect number of experts from the model architecture."""
        base = self._unwrap_model()
        layer0 = getattr(getattr(base, "model", base), "layers", None)
        if layer0 is None:
            raise RuntimeError("Cannot detect MoE layers")
        mlp = getattr(layer0[0], "mlp", None)
        if mlp is None:
            raise RuntimeError("Cannot detect MoE MLP block")
        experts = getattr(mlp, "experts", None)
        if experts is not None:
            gate_up = getattr(experts, "gate_up_proj", None)
            if gate_up is not None:
                return gate_up.shape[0]
        raise RuntimeError("Cannot detect number of experts")

    @torch.no_grad()
    def get_expert_activations(
        self,
        texts: list[str],
        layers: list[int],
    ) -> dict[int, Tensor]:
        """Per-expert mean-pooled activations, bypassing router.

        For each layer, manually applies all expert FFNs to the pre-MoE
        hidden state (bypassing the router). Returns activations mean-pooled
        across the sequence dimension.

        Returns:
            dict[layer, Tensor of shape (n_texts, n_experts, hidden_dim)]
        """
        all_expert_acts: dict[int, list[Tensor]] = {l: [] for l in layers}

        for i, text in enumerate(texts):
            if (i + 1) % 50 == 0 or i == 0:
                logger.info("  Collecting expert activations: %d/%d", i + 1, len(texts))

            pre_moe_states: dict[int, Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHook] = []

            for layer_idx in layers:
                layer_module = self._get_layer_module(layer_idx)

                def _pre_moe_hook(mod: Any, inp: Any, out: Any, idx: int = layer_idx) -> None:
                    pre_moe_states[idx] = out.detach().cpu()

                hooks.append(
                    layer_module.post_attention_layernorm.register_forward_hook(_pre_moe_hook)
                )

            try:
                inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
                self._model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            for layer_idx in layers:
                hidden = pre_moe_states[layer_idx].to(self._device)
                hidden = hidden.squeeze(0)  # (seq, hidden_dim)

                experts_module = self._get_layer_module(layer_idx).mlp.experts
                gate_up_proj = experts_module.gate_up_proj  # (n_experts, 2*inter, hidden)
                down_proj = experts_module.down_proj  # (n_experts, hidden, inter)
                act_fn = experts_module.act_fn

                gate_up = torch.einsum("sh,eoh->eso", hidden, gate_up_proj)
                gate, up = gate_up.chunk(2, dim=-1)
                intermediate = act_fn(gate) * up
                expert_out = torch.einsum("eso,eho->esh", intermediate, down_proj)
                expert_mean = expert_out.mean(dim=1)  # (n_experts, hidden_dim)
                all_expert_acts[layer_idx].append(expert_mean.cpu())

        return {l: torch.stack(acts) for l, acts in all_expert_acts.items()}

    @torch.no_grad()
    def get_router_logits(
        self,
        texts: list[str],
        layers: list[int],
    ) -> dict[int, list[Tensor]]:
        """Router selection logits for each text.

        Returns:
            dict[layer, list of Tensors of shape (seq_len, n_experts)]
        """
        all_router: dict[int, list[Tensor]] = {l: [] for l in layers}

        for text in texts:
            router_logits: dict[int, Tensor] = {}
            hooks: list[torch.utils.hooks.RemovableHook] = []

            for layer_idx in layers:
                layer_module = self._get_layer_module(layer_idx)

                def _router_hook(mod: Any, inp: Any, out: Any, idx: int = layer_idx) -> None:
                    router_logits[idx] = out[0].detach().cpu()

                hooks.append(layer_module.mlp.gate.register_forward_hook(_router_hook))

            try:
                inputs = self._tokenizer(text, return_tensors="pt").to(self._device)
                self._model(**inputs)
            finally:
                for h in hooks:
                    h.remove()

            for l in layers:
                all_router[l].append(router_logits[l])

        return all_router
