"""Tests for ablation-resistance (ART) steering.

The critical property (plan Phase A step 8): the ablation projection must be
DIFFERENTIABLE, so the ART loss can teach the model to depend on the moral
subspace. The fast unit tests prove the projection itself carries gradient and
removes the subspace; the slow test confirms ART-loss gradient reaches LoRA
parameters through the ablated forward on a real OLMo model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepsteer.foundations import FOUNDATION_ORDER
from deepsteer.steering.ablation_resistance import (
    AblationResistanceSteering,
    _make_projection_hook,
    build_subspace_basis,
)


def _ortho_basis(r: int, hidden: int, seed: int = 0) -> np.ndarray:
    """Random ``(r, hidden)`` array with orthonormal rows."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden, r)))  # (hidden, r)
    return q.T.astype(np.float32)


# ---------------------------------------------------------------------------
# Projection hook (the differentiability that ART hinges on)
# ---------------------------------------------------------------------------


def test_projection_hook_is_differentiable():
    hidden, r = 32, 4
    V = torch.from_numpy(_ortho_basis(r, hidden))
    hook = _make_projection_hook(V)
    x = torch.randn(2, 5, hidden, requires_grad=True)

    out = hook(None, None, x)
    assert out.requires_grad and out.grad_fn is not None, "projection not in autograd graph"

    out.pow(2).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0, "no gradient flowed back through the projection"


def test_projection_removes_subspace():
    hidden, r = 48, 6
    V = torch.from_numpy(_ortho_basis(r, hidden))
    out = _make_projection_hook(V)(None, None, torch.randn(3, 7, hidden))
    # The residual must be orthogonal to every basis row.
    assert (out.to(torch.float32) @ V.t()).abs().max().item() < 1e-4


def test_projection_hook_preserves_tuple_output():
    hidden, r = 16, 2
    V = torch.from_numpy(_ortho_basis(r, hidden))
    out = _make_projection_hook(V)(None, None, (torch.randn(1, 4, hidden), "kv_cache"))
    assert isinstance(out, tuple) and out[1] == "kv_cache"
    assert out[0].shape == (1, 4, hidden)


# ---------------------------------------------------------------------------
# Subspace basis
# ---------------------------------------------------------------------------


def test_build_subspace_basis_orthonormal_and_rank():
    hidden, n_layers = 64, 3
    rng = np.random.default_rng(1)
    dirs = {f: {L: rng.standard_normal(hidden).astype(np.float32) for L in range(n_layers)}
            for f in FOUNDATION_ORDER}
    basis, ranks, names = build_subspace_basis(dirs, kind="probe", n_layers=n_layers)

    assert names == FOUNDATION_ORDER
    assert set(basis) == set(range(n_layers))
    for L, V in basis.items():
        assert V.shape == (6, hidden)          # 6 random vecs in 64-dim are full rank
        assert ranks[L] == 6
        assert np.abs(V @ V.T - np.eye(6)).max() < 1e-4


def test_build_subspace_basis_meandiff_keys():
    hidden, n_layers = 32, 2
    rng = np.random.default_rng(2)
    dirs = {f"{f}_meandiff": {L: rng.standard_normal(hidden).astype(np.float32)
                              for L in range(n_layers)} for f in FOUNDATION_ORDER}
    basis, _, names = build_subspace_basis(dirs, kind="meandiff", n_layers=n_layers)
    assert names == [f"{f}_meandiff" for f in FOUNDATION_ORDER]
    assert set(basis) == {0, 1}


def test_build_subspace_basis_skips_incomplete_layers():
    hidden = 16
    rng = np.random.default_rng(3)
    dirs = {f: {0: rng.standard_normal(hidden).astype(np.float32)} for f in FOUNDATION_ORDER}
    dirs[FOUNDATION_ORDER[0]].pop(0)  # drop one foundation at layer 0
    basis, _, _ = build_subspace_basis(dirs, kind="probe", n_layers=1)
    assert basis == {}  # layer 0 incomplete -> no basis


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def test_calibrate_coefficient():
    art = AblationResistanceSteering({}, coefficient=0.01)
    # want lambda*|gap| = ratio*sft -> lambda = 0.1*2.0/0.5 = 0.4
    assert abs(art.calibrate_coefficient(2.0, 0.5, target_ratio=0.1) - 0.4) < 1e-9
    # degenerate gap -> keep current
    assert art.calibrate_coefficient(2.0, 0.0) == 0.01


# ---------------------------------------------------------------------------
# End-to-end gradient flow on a real model (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_art_gradient_flows_to_lora_real_model():
    """ART loss alone must produce nonzero LoRA grads via the ablated forward."""
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    model = WhiteBoxModel("allenai/OLMo-2-0425-1B", device="cpu",
                          access_tier=AccessTier.WEIGHTS)
    hidden = model.model.config.hidden_size
    n_layers = model.info.n_layers

    model._model = get_peft_model(model._model, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    model._model.train()

    rng = np.random.default_rng(0)
    dirs = {f: {L: rng.standard_normal(hidden).astype(np.float32) for L in range(n_layers)}
            for f in FOUNDATION_ORDER}
    art = AblationResistanceSteering(dirs, coefficient=0.5)
    art.attach(model)

    ids = torch.randint(0, 100, (1, 8))
    attn = torch.ones_like(ids)
    labels = ids.clone()

    normal = model.model(input_ids=ids, attention_mask=attn, labels=labels).loss
    art_loss = art.compute_art_loss(
        model, input_ids=ids, attention_mask=attn, labels=labels,
        normal_loss=float(normal.detach()),
    )
    assert art_loss.requires_grad
    assert art.last_gap is not None and abs(art.last_gap) > 1e-6  # ablation moved the loss

    art_loss.backward()  # ART ALONE — no SFT term
    lora_grads = [p.grad for n, p in model.model.named_parameters()
                  if p.requires_grad and "lora" in n.lower()]
    assert lora_grads, "no trainable LoRA params found"
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in lora_grads), "ART grad did not reach LoRA params through the ablation"
    model.release()
