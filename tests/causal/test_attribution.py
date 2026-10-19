"""ST_PLAN T0.5: the AtP harness (deepsteer.causal.attribution)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from deepsteer.causal.attribution import atp, option_margin, patch_residual  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    pytest.importorskip("transformers")
    import st_tiny

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    path = st_tiny.build("olmo3_think", tmp_path_factory.mktemp("ck") / "olmo3")
    return WhiteBoxModel(str(path), access_tier=AccessTier.WEIGHTS, device="cpu",
                         torch_dtype=torch.float32)


def _pair(m):
    ic = m.tokenizer.encode("The answer to the question is clearly the first one")
    ir = list(ic)
    ir[3] = ic[3] + 1
    return ic, ir


def test_atp_reaches_every_layer_and_matches_patching_exactly_when_linear(tiny):
    # most probable failure: the gradient chain is cut at each hooked site (activations detached),
    # so every site but the last block reads zero downstream effect
    ic, ir = _pair(tiny)
    mf = option_margin([32], [33])
    r = atp(tiny, ic, ir, mf, neuron_layers=[1])
    n_layers = len(tiny._decoder_layers())  # noqa: SLF001
    assert r.residual.shape == (n_layers, len(ic))
    assert np.all(r.residual[:, :3] == 0)  # identical prefix: nothing to patch
    assert np.abs(r.residual[0, 3:]).max() > 0  # the first block's sites reach the metric
    assert r.neurons[1].shape[0] == len(ic) and r.gate_sign_flip[1].dtype == bool
    assert not r.gate_sign_flip[1][:3].any()
    # patching the last block's output at the last position is exactly linear in the residual
    real = patch_residual(tiny, ic, ir, mf, [(n_layers - 1, len(ic) - 1)])[0]
    assert real == pytest.approx(r.metric_corrupt - r.metric_clean, abs=1e-4)


def test_atp_leaves_weights_untouched_and_needs_aligned_inputs(tiny):
    # most probable failure: backward through the model accumulates weight gradients (memory on
    # a 7B, and leaked state across calls), or unaligned inputs are silently truncated
    ic, ir = _pair(tiny)
    flags = [p.requires_grad for p in tiny.model.parameters()]
    atp(tiny, ic, ir, option_margin([32], [33]))
    assert all(p.grad is None for p in tiny.model.parameters())
    assert [p.requires_grad for p in tiny.model.parameters()] == flags
    with pytest.raises(ValueError, match="token-aligned"):
        atp(tiny, ic, ir[:-1], option_margin([32], [33]))


@pytest.mark.slow
def test_atp_positive_control_on_olmo2_1b():
    # ST_PLAN T0.5's positive control: AtP vs real patching on 50 residual sites (seed 0, the
    # cells at and after the one differing token), Spearman >= 0.7. Measured 2026-10-19: 0.997.
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    try:
        m = WhiteBoxModel("allenai/OLMo-2-0425-1B", access_tier=AccessTier.WEIGHTS,
                          device="cpu", torch_dtype=torch.float32)
    except OSError:
        pytest.skip("OLMo-2-0425-1B not available offline")
    tok = m.tokenizer
    c = "Q: What is the capital of France?\nA) Paris\nB) Rome\nAnswer:"
    ic, ir = tok.encode(c), tok.encode(c.replace("France", "Italy"))
    mf = option_margin(tok.encode(" A", add_special_tokens=False),
                       tok.encode(" B", add_special_tokens=False))
    r = atp(m, ic, ir, mf)
    d = next(i for i, (a, b) in enumerate(zip(ic, ir)) if a != b)
    cells = [(la, p) for la in range(r.residual.shape[0]) for p in range(d, len(ic))]
    sites = [cells[i] for i in np.random.default_rng(0).choice(len(cells), 50, replace=False)]
    real = patch_residual(m, ic, ir, mf, sites)
    est = np.array([r.residual[la, p] for la, p in sites])
    rho = np.corrcoef(np.argsort(np.argsort(est)), np.argsort(np.argsort(real)))[0, 1]
    assert rho >= 0.7, rho
