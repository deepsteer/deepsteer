"""Training-time intervention along a fixed probe direction.

Two methods (selected via :class:`SteeringMethod`):

- ``gradient_penalty``: the model's forward pass is unchanged; we capture
  the residual stream at ``target_layer`` and expose an auxiliary loss
  ``λ × probe_logit²`` (mean-pooled over assistant tokens) that the
  trainer adds to its main SFT loss.  Gradients flow back through the
  capture point and discourage representations aligned with the probe
  direction.

- ``activation_patch``: at every patched layer's output, a constant
  ``γ × (w / ||w||)`` is *subtracted* from every token's residual so
  the model has to learn the SFT task while seeing artificially
  anti-persona residuals.  No auxiliary loss; the patch alone biases
  what the model can use as features.

Step 1 (`papers/persona_monitoring/outputs/phase_d/c10_inference_patch/`) established that this
constant-offset variant is the right primitive at 1B scale: pure
``h - ((h·w)/||w||²) w`` projection has only ~4% norm magnitude (the
persona direction's natural amplitude) and is too weak to bite.  The
constant-offset analog forces a directional perturbation independent
of current alignment, and the model has to compensate.

Both methods leave probe weights frozen; only the LoRA / base-model
parameters receive gradient.

Usage::

    probe = PersonaProbeWeights.from_dict(json.load(open("probe.json"))["weights"])
    w_tensor, _ = probe.to_tensors()
    steering = TrainingTimeSteering(
        probe_weight=w_tensor,
        target_layer=probe.layer,
        method="gradient_penalty",
        coefficient=0.05,
    )
    trainer = ChatLoRATrainer(model, conversations, steering=steering, ...)
    trainer.train(...)

The steering object is attached to the model in
``ChatLoRATrainer.train()`` before the optimization loop and detached
before the final post-FT eval, so generation is measured *without* the
intervention active.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)


SteeringMethod = Literal["gradient_penalty", "activation_patch"]


# Recommended coefficient ranges, derived from Step 1 (inference-time)
# diagnostics + the rule of thumb that aux_loss should start at ~10% of
# sft_loss for gradient_penalty.  These are starting points; tune by
# inspecting the first-batch aux_loss / sft_loss ratio.
DEFAULT_GRADIENT_PENALTY_LAMBDA = 0.05
DEFAULT_ACTIVATION_PATCH_GAMMA = 1.5


class TrainingTimeSteering:
    """Hook-based training-time intervention along a frozen probe direction.

    Args:
        probe_weight: 1-D tensor of shape ``(hidden_dim,)`` defining the
            probe direction.  Frozen (detached) on construction; never
            receives gradient.
        target_layer: Decoder layer at which the gradient_penalty
            capture happens.  For activation_patch, this is the seed
            layer for ``patch_layers``.
        method: ``"gradient_penalty"`` or ``"activation_patch"``.
        coefficient: λ (gradient_penalty) or γ (activation_patch).
        patch_layers: Layer indices at which to apply activation_patch.
            Defaults to ``[target_layer, target_layer+1, target_layer+2]``
            (matches Step 1's bracketing of the C8 peak).  Ignored for
            gradient_penalty.
        pool_mode: How to aggregate per-token probe logits in
            gradient_penalty.  ``"mean"`` (default) mean-pools the
            residual stream over assistant tokens then applies the
            probe — matches the probe's training distribution.
        device: Device for the probe weight (defaults to model device
            on first attach).
    """

    def __init__(
        self,
        probe_weight: Tensor,
        target_layer: int,
        method: SteeringMethod,
        *,
        coefficient: float | None = None,
        patch_layers: list[int] | None = None,
        pool_mode: Literal["mean"] = "mean",
        device: str | None = None,
    ) -> None:
        if method not in ("gradient_penalty", "activation_patch"):
            raise ValueError(f"Unknown method: {method!r}")
        if pool_mode not in ("mean",):
            raise ValueError(f"Unknown pool_mode: {pool_mode!r}")

        self._w = probe_weight.detach().clone().to(torch.float32)
        self._target_layer = int(target_layer)
        self._method: SteeringMethod = method
        if coefficient is None:
            coefficient = (
                DEFAULT_GRADIENT_PENALTY_LAMBDA
                if method == "gradient_penalty"
                else DEFAULT_ACTIVATION_PATCH_GAMMA
            )
        self._coefficient = float(coefficient)
        self._patch_layers = list(
            patch_layers
            if patch_layers is not None
            else [target_layer, target_layer + 1, target_layer + 2]
        )
        self._pool_mode = pool_mode
        self._device = device

        # Captured tensor for gradient_penalty.  Held graph-attached so
        # aux_loss.backward() flows back through it.
        self._captured: Tensor | None = None
        self._handles: list[Any] = []

        # Step counters for logging / diagnostics.
        self._n_attached_steps = 0
        self._aux_loss_history: list[float] = []
        self._patch_norm_history: list[float] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def method(self) -> SteeringMethod:
        return self._method

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def set_coefficient(self, value: float) -> None:
        """Allow callers to retune (e.g. after first-batch calibration)."""
        self._coefficient = float(value)

    @property
    def target_layer(self) -> int:
        return self._target_layer

    @property
    def patch_layers(self) -> list[int]:
        return list(self._patch_layers)

    @property
    def is_attached(self) -> bool:
        return bool(self._handles)

    @property
    def captured(self) -> Tensor | None:
        return self._captured

    @property
    def aux_loss_history(self) -> list[float]:
        return list(self._aux_loss_history)

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, model: WhiteBoxModel) -> None:
        """Register hooks on the underlying HF model."""
        if self.is_attached:
            raise RuntimeError("Steering already attached")

        device = self._device or model._device
        self._w = self._w.to(device=device, dtype=torch.float32)

        if self._method == "gradient_penalty":
            self._attach_capture_hook(model)
        else:  # activation_patch
            self._attach_patch_hooks(model)

    def detach(self) -> None:
        """Remove all hooks; reset captured state."""
        for h in self._handles:
            h.remove()
        self._handles = []
        self._captured = None

    def _attach_capture_hook(self, model: WhiteBoxModel) -> None:
        module = model._get_layer_module(self._target_layer)

        def _hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            # IMPORTANT: do NOT detach.  We need this graph-attached so
            # aux_loss.backward() can flow gradients back through it.
            self._captured = tensor
            return output

        self._handles.append(module.register_forward_hook(_hook))
        logger.info(
            "TrainingTimeSteering attached: gradient_penalty at layer %d, "
            "λ=%.4f", self._target_layer, self._coefficient,
        )

    def _attach_patch_hooks(self, model: WhiteBoxModel) -> None:
        # Constant offset along negative unit_w applied at every patched
        # layer's output.  See Step 1's `make_amplify_patch` with α<0.
        w = self._w
        w_norm = float(torch.linalg.norm(w).item())
        # Subtract γ × unit_w → push residual *against* the probe
        # direction at every token, every patched layer.
        delta = -self._coefficient * (w / w_norm)  # (D,)

        def _make_patch_hook() -> Callable[..., Any]:
            def _hook(_m: nn.Module, _inp: Any, output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                orig_dtype = tensor.dtype
                patched = (tensor.to(torch.float32) + delta).to(orig_dtype)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return _hook

        for layer_idx in self._patch_layers:
            module = model._get_layer_module(layer_idx)
            self._handles.append(module.register_forward_hook(_make_patch_hook()))

        # Diagnostic: how big is the patch relative to typical residual?
        # We don't know yet (depends on first batch); record on first
        # forward via aux_loss-style logging.
        logger.info(
            "TrainingTimeSteering attached: activation_patch at layers %s, "
            "γ=%.4f, ||delta||=%.3f",
            self._patch_layers, self._coefficient,
            float(torch.linalg.norm(delta).item()),
        )

    # ------------------------------------------------------------------
    # gradient_penalty: aux_loss
    # ------------------------------------------------------------------

    def aux_loss(
        self,
        attention_mask: Tensor,
        *,
        label_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute λ × probe_logit² from the most recent forward.

        Args:
            attention_mask: ``(B, T)`` 0/1 mask of real tokens.
            label_mask: optional ``(B, T)`` 0/1 mask of *assistant*
                tokens (labels != IGNORE_INDEX).  When provided, the
                probe is only computed over assistant tokens.  This is
                preferred — the probe should "fire" on what the model
                produces, not on the user prompt.

        Returns:
            Scalar tensor (graph-attached to the captured residual).

        Raises:
            RuntimeError: if no forward has happened since attach.
        """
        if self._method != "gradient_penalty":
            raise RuntimeError(
                f"aux_loss is only valid for gradient_penalty (got {self._method!r})"
            )
        if self._captured is None:
            raise RuntimeError(
                "No captured activations.  Did you forget to run a forward "
                "pass between attach() and aux_loss()?"
            )

        h = self._captured  # (B, T, D), graph-attached, model dtype
        # Cast to fp32 for numerically-stable inner product against the
        # 2048-dim probe direction.  Gradient flows back through the
        # cast.
        h32 = h.to(torch.float32)

        # Build the pooling mask.  Prefer label_mask (assistant tokens)
        # if provided; otherwise fall back to attention_mask (all real
        # tokens — includes user prefix, less ideal but still works).
        if label_mask is not None:
            mask = label_mask.to(h32.device).to(torch.float32)
        else:
            mask = attention_mask.to(h32.device).to(torch.float32)
        mask3 = mask.unsqueeze(-1)  # (B, T, 1)
        denom = mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        pooled = (h32 * mask3).sum(dim=1) / denom  # (B, D)

        logits = pooled @ self._w  # (B,)
        penalty = self._coefficient * (logits ** 2).mean()
        self._aux_loss_history.append(float(penalty.detach().item()))
        return penalty

    def calibrate_lambda(
        self,
        first_batch_sft_loss: float,
        first_batch_aux_loss: float,
        target_ratio: float = 0.10,
    ) -> float:
        """Suggest a λ such that aux_loss ≈ ``target_ratio`` × sft_loss.

        Args:
            first_batch_sft_loss: SFT loss on the first batch (no penalty).
            first_batch_aux_loss: aux_loss on the same batch *with the
                current λ*.
            target_ratio: Desired ratio of aux_loss / sft_loss (default
                0.10).

        Returns:
            Suggested new λ.  Caller decides whether to apply it via
            :meth:`set_coefficient`.
        """
        if first_batch_aux_loss <= 0:
            return self._coefficient
        ratio = first_batch_aux_loss / max(first_batch_sft_loss, 1e-6)
        return self._coefficient * (target_ratio / max(ratio, 1e-6))

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"TrainingTimeSteering(method={self._method!r}, "
            f"target_layer={self._target_layer}, "
            f"coefficient={self._coefficient}, "
            f"patch_layers={self._patch_layers})"
        )


__all__ = [
    "DEFAULT_ACTIVATION_PATCH_GAMMA",
    "DEFAULT_GRADIENT_PENALTY_LAMBDA",
    "SteeringMethod",
    "TrainingTimeSteering",
]
