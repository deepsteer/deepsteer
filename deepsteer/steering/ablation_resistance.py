"""Training-time ablation-resistance steering (ART).

Sprint 6 of the Phase 3 ablation-resistance study. Where
:class:`TrainingTimeSteering` penalizes a *single* probe direction via a
single-forward auxiliary loss, ART operates on the full moral foundation
*subspace* via a PAIRED forward pass and the opposite objective: it *rewards* the
model when ablating the moral subspace HURTS, training the model to route
moral-content generation through that subspace so any future Heretic-style
ablation pays a quality cost.

Per training batch:

  1. (the trainer's normal forward gives) ``L_sft`` -- assistant-token CE.
  2. ablated forward: project the moral subspace out of the residual stream at
     every target layer (differentiable), recompute the same loss -> ``L_ablated``.
  3. ``ART loss = -coefficient * (L_ablated - L_sft)``.

So minimizing ``L_sft + ART`` minimizes ``L_sft`` while MAXIMIZING ``L_ablated``,
widening the dependency gap. The projection is differentiable (frozen
directions, live activations), so gradients flow back through the ablation and
teach the model to depend on the subspace.

The architecture is deliberately separate from ``TrainingTimeSteering``: that
class captures one activation in a single forward and reads it back in
``aux_loss``; ART needs a second, hook-modified forward and an opposite-sign
objective over a multi-direction subspace. Sharing one interface would force
both.

Sprint 5 found the BASE / pretraining moral subspace is the generation-functional
one (per-state self-dependency is *lower* than base-transfer dependency after
SFT, because the ~40 deg SFT rotation moves the probe-optimal direction but not
the generation-load-bearing one), so ART should target BASE directions.

Usage::

    directions = load_foundation_directions("…/olmo3_base/exp1_probe_directions.npz")
    art = AblationResistanceSteering(directions, coefficient=0.01)
    trainer = ChatLoRATrainer(model, conversations, art_steering=art, …)
    trainer.train(…)
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from torch import Tensor

from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.foundations import FOUNDATION_ORDER

logger = logging.getLogger(__name__)

# Start point: λ calibrated so |ART| ≈ 10% of L_sft on the first batch (same rule
# of thumb as TrainingTimeSteering's gradient_penalty default).
DEFAULT_ART_LAMBDA = 0.01

_KEY_RE = re.compile(r"^(.+)_layer(\d+)$")


# ---------------------------------------------------------------------------
# Direction loading + subspace basis (self-contained; mirrors the Sprint 5
# moral_dependency.py script so the package carries no paper-script dependency)
# ---------------------------------------------------------------------------


def load_foundation_directions(path) -> dict[str, dict[int, np.ndarray]]:
    """Load an ``exp1_probe_directions.npz``-format file -> ``{name: {layer: vec}}``."""
    npz = np.load(path)
    out: dict[str, dict[int, np.ndarray]] = {}
    for key in npz.files:
        m = _KEY_RE.match(key)
        if not m:
            continue
        name, layer = m.group(1), int(m.group(2))
        out.setdefault(name, {})[layer] = npz[key]
    return out


def build_subspace_basis(
    directions: dict[str, dict[int, np.ndarray]],
    *,
    kind: str = "probe",
    n_layers: int,
    foundations: list[str] | None = None,
    tol: float = 1e-6,
) -> tuple[dict[int, np.ndarray], dict[int, int], list[str]]:
    """Per-layer orthonormal basis for the moral foundation subspace.

    Returns ``(basis_by_layer, rank_by_layer, names)`` where ``basis_by_layer[L]``
    is an ``(r, hidden)`` float32 array of orthonormal rows spanning the moral
    subspace at layer ``L`` (only layers where every foundation is present).
    """
    foundations = foundations or FOUNDATION_ORDER
    if kind == "probe":
        names = list(foundations)
    elif kind == "meandiff":
        names = [f"{f}_meandiff" for f in foundations]
    else:
        raise ValueError(f"Unknown direction kind: {kind!r}")

    present = [n for n in names if n in directions]
    if len(present) < len(names):
        logger.warning("directions missing %d/%d %s keys: %s",
                       len(names) - len(present), len(names), kind,
                       [n for n in names if n not in directions])

    basis: dict[int, np.ndarray] = {}
    ranks: dict[int, int] = {}
    for layer in range(n_layers):
        vecs = [directions[n][layer] for n in present if layer in directions[n]]
        if not present or len(vecs) < len(present):
            continue
        mat = np.stack(vecs).astype(np.float64)  # (k, hidden)
        _, s, vt = np.linalg.svd(mat, full_matrices=False)
        r = int((s > tol * s[0]).sum()) if s.size and s[0] > 0 else 0
        basis[layer] = vt[:r].astype(np.float32)
        ranks[layer] = r
    return basis, ranks, present


def _make_projection_hook(basis: Tensor):
    """Forward hook projecting a layer's residual output off ``span(basis)``.

    ``basis`` is ``(r, hidden)`` orthonormal rows. Differentiable in the
    activations (basis frozen): ``x - (x @ basisᵀ) @ basis``, computed in fp32
    for stability and cast back to the activation dtype.
    """

    def hook(_module, _inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        orig_dtype = tensor.dtype
        t = tensor.to(torch.float32)
        proj = (t @ basis.t()) @ basis
        patched = (t - proj).to(orig_dtype)
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        return patched

    return hook


# ---------------------------------------------------------------------------
# ART steering
# ---------------------------------------------------------------------------


class AblationResistanceSteering:
    """Paired-forward steering that maximizes moral-subspace dependency.

    Args:
        moral_directions: ``{name: {layer: vec}}`` (e.g. from
            :func:`load_foundation_directions`) or a path to an
            ``exp1_probe_directions.npz``-format file.
        coefficient: λ_art. ``ART = -λ (L_ablated - L_sft)``. Default 0.01;
            calibrate so |ART| ≈ 10% of L_sft on the first batch.
        target_layers: Layers to ablate. ``None`` = all layers where the full
            moral subspace is present.
        direction_kind: ``"probe"`` (default) or ``"meandiff"``.
        foundations: Foundation order; defaults to ``FOUNDATION_ORDER``.
        device: Override device for the basis tensors (defaults to model device).
    """

    def __init__(
        self,
        moral_directions: dict[str, dict[int, np.ndarray]] | str,
        *,
        coefficient: float = DEFAULT_ART_LAMBDA,
        max_coefficient: float = 1.0,
        target_gap: float = 0.3,
        target_layers: list[int] | None = None,
        direction_kind: str = "probe",
        foundations: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        if isinstance(moral_directions, (str, bytes)) or hasattr(moral_directions, "__fspath__"):
            moral_directions = load_foundation_directions(moral_directions)
        self._directions = moral_directions
        self._coefficient = float(coefficient)
        self._max_coefficient = float(max_coefficient)
        self._target_gap = float(target_gap)
        self._requested_layers = target_layers
        self._direction_kind = direction_kind
        self._foundations = foundations
        self._device = device

        # Resolved at attach() (needs model.info.n_layers / device / dtype).
        self._basis_t: dict[int, Tensor] = {}
        self._ranks: dict[int, int] = {}
        self._names: list[str] = []
        self._layers: list[int] = []
        self._attached = False

        # Diagnostics.
        self._last_sft: float | None = None
        self._last_ablated: float | None = None
        self._last_gap: float | None = None
        self._art_history: list[float] = []
        self._gap_history: list[float] = []

    # -- properties ---------------------------------------------------------

    @property
    def coefficient(self) -> float:
        return self._coefficient

    def set_coefficient(self, value: float) -> None:
        self._coefficient = float(value)

    @property
    def is_attached(self) -> bool:
        return self._attached

    @property
    def target_layers(self) -> list[int]:
        return list(self._layers)

    @property
    def n_directions(self) -> int:
        return len(self._names)

    @property
    def last_gap(self) -> float | None:
        return self._last_gap

    @property
    def art_loss_history(self) -> list[float]:
        return list(self._art_history)

    @property
    def gap_history(self) -> list[float]:
        return list(self._gap_history)

    # -- attach / detach ----------------------------------------------------

    def attach(self, model: WhiteBoxModel) -> None:
        """Resolve the per-layer basis tensors on the model's device/dtype.

        Does NOT register persistent hooks: the ablation is applied only during
        the ablated forward inside :meth:`compute_art_loss`, so the model's
        normal forward (and the trainer's L_sft) stay untouched.
        """
        if self._attached:
            raise RuntimeError("ART steering already attached")
        n_layers = model.info.n_layers
        basis_np, self._ranks, self._names = build_subspace_basis(
            self._directions, kind=self._direction_kind, n_layers=n_layers,
            foundations=self._foundations,
        )
        if not basis_np:
            raise RuntimeError(
                "No layers with a complete moral subspace; check the directions "
                "npz matches the model hidden dim / layer count."
            )
        available = sorted(basis_np)
        if self._requested_layers is not None:
            wanted = set(self._requested_layers)
            self._layers = [L for L in available if L in wanted]
        else:
            self._layers = available

        param = next(model.model.parameters())
        device = self._device or param.device
        self._basis_t = {
            L: torch.from_numpy(basis_np[L]).to(device=device, dtype=torch.float32)
            for L in self._layers
        }
        self._attached = True
        logger.info(
            "ART attached: %d %s directions over %d layers, λ=%.4f",
            len(self._names), self._direction_kind, len(self._layers), self._coefficient,
        )

    def detach(self) -> None:
        """Drop the resolved basis tensors."""
        self._basis_t = {}
        self._layers = []
        self._attached = False

    @contextmanager
    def _ablation(self, model: WhiteBoxModel):
        """Register the projection hooks for the duration of one forward pass."""
        handles = []
        try:
            for L in self._layers:
                module = model._get_layer_module(L)
                handles.append(module.register_forward_hook(
                    _make_projection_hook(self._basis_t[L])))
            yield
        finally:
            for h in handles:
                h.remove()

    # -- ART loss -----------------------------------------------------------

    @torch.no_grad()
    def measure_gap(
        self,
        model: WhiteBoxModel,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        normal_loss: Tensor | float,
    ) -> float:
        """No-grad ``L_ablated - L_sft`` for cheap λ calibration (no graph kept)."""
        if not self._attached:
            raise RuntimeError("ART steering not attached; call attach(model) first.")
        with self._ablation(model):
            l_ablated = model.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            ).loss
        sft_val = float(normal_loss.detach().item()) if isinstance(normal_loss, Tensor) \
            else float(normal_loss)
        self._last_sft = sft_val
        self._last_ablated = float(l_ablated.item())
        self._last_gap = self._last_ablated - sft_val
        return self._last_gap

    def compute_art_loss(
        self,
        model: WhiteBoxModel,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        labels: Tensor,
        normal_loss: Tensor | float,
    ) -> Tensor:
        """Run the ablated forward and return ``λ·relu(target_gap - gap)``.

        ``gap = L_ablated - L_sft``. The hinge drives the gap up to
        ``target_gap`` then gives zero gradient (bounded, unlike the original
        ``-λ·gap`` which ran away). ``normal_loss`` is the trainer's SFT loss;
        pass it DETACHED so only the ablated term carries gradient. The returned
        tensor is graph-attached through ``L_ablated`` (and thus the
        differentiable projection), so ``.backward()`` teaches the model to
        depend on the subspace — up to the target.
        """
        if not self._attached:
            raise RuntimeError("ART steering not attached; call attach(model) first.")

        with self._ablation(model):
            ablated = model.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels,
            )
        l_ablated = ablated.loss

        sft_val = float(normal_loss.detach().item()) if isinstance(normal_loss, Tensor) \
            else float(normal_loss)
        gap = l_ablated - normal_loss          # graph-attached via l_ablated
        # Hinge: drive the gap UP toward target_gap, then ZERO gradient. The
        # unbounded -λ·gap objective runs away (gap exploded to 50+ nats and
        # destroyed the SFT loss); relu(target_gap - gap) bounds the dependency
        # the model is rewarded for building, then stops pushing.
        art = self._coefficient * torch.clamp(self._target_gap - gap, min=0.0)

        self._last_sft = sft_val
        self._last_ablated = float(l_ablated.detach().item())
        self._last_gap = self._last_ablated - sft_val
        self._gap_history.append(self._last_gap)
        self._art_history.append(float(art.detach().item()))
        return art

    def calibrate_coefficient(
        self,
        first_batch_sft_loss: float,
        first_batch_gap: float,
        target_ratio: float = 0.10,
    ) -> float:
        """Suggest λ so that |ART| ≈ ``target_ratio`` × L_sft on the first batch.

        With the hinge objective ``ART = λ·relu(target_gap - gap)``, solve for λ
        such that the initial ART loss ≈ ``ratio × sft``, then clamp to
        ``max_coefficient``. Uses the hinge magnitude ``relu(target_gap - gap)``
        (not the raw gap), so a tiny starting gap no longer blows λ up. Caller
        applies via :meth:`set_coefficient`.
        """
        shortfall = max(self._target_gap - first_batch_gap, 1e-6)
        lam = target_ratio * max(first_batch_sft_loss, 1e-6) / shortfall
        return min(lam, self._max_coefficient)

    def __repr__(self) -> str:
        return (f"AblationResistanceSteering(kind={self._direction_kind!r}, "
                f"coefficient={self._coefficient}, "
                f"n_layers={len(self._layers)}, n_directions={len(self._names)})")


__all__ = [
    "DEFAULT_ART_LAMBDA",
    "AblationResistanceSteering",
    "build_subspace_basis",
    "load_foundation_directions",
]
