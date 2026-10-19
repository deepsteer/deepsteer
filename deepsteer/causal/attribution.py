"""Attribution patching (AtP) for a decision-token metric (ST_PLAN §7 T0.5).

AtP estimates, for every site at once, the effect on a metric of patching that site's activation
from a corrupted run into the clean run: ``(a_corrupt - a_clean) . d metric / d a`` at the clean
run (a first-order Taylor estimate). Sites:

- **residual**: each decoder block's output (hidden-state index ``layer + 1``) at each position;
- **MLP neurons** (SwiGLU blocks): the input of ``down_proj`` (``act(gate) * up``) per position
  and neuron, with a **sign-regime flag** where the gate pre-activation changes sign between the
  two runs: SiLU is not monotone below zero, so the linear estimate is least trustworthy exactly
  where the gate crosses into or out of that regime.

The metric of record is the **option margin** at the decision token: log-probability mass of the
target option's token(s) minus the other option's, at the last position. Clean and corrupted
inputs must be token-aligned (equal length; the alignment rule is fixed, not post hoc).
``patch_residual`` runs the real patch for chosen sites, the positive control AtP is validated
against. Pure PyTorch on ``WhiteBoxModel``'s decoder layers; no Captum, no new hook stack.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)

MetricFn = Callable[[torch.Tensor], torch.Tensor]  # last-position logits [vocab] -> scalar


def option_margin(target_ids: Sequence[int], other_ids: Sequence[int]) -> MetricFn:
    """The decision-token option margin: logsumexp of the target option's token log-probs minus
    the other option's, from the last position's logits."""
    t, o = list(target_ids), list(other_ids)

    def metric(logits_last: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(logits_last.float(), dim=-1)
        return torch.logsumexp(lp[t], 0) - torch.logsumexp(lp[o], 0)

    return metric


@dataclass
class AtPResult:
    """AtP estimates (rows = layers, columns = positions) and the two runs' metrics."""

    residual: np.ndarray  # [n_layers, seq]: estimated metric change per residual site
    metric_clean: float
    metric_corrupt: float
    neurons: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> [seq, d_ff]
    gate_sign_flip: dict[int, np.ndarray] = field(default_factory=dict)  # layer -> bool [seq, d_ff]


def _mlp(layer: torch.nn.Module) -> torch.nn.Module:
    mlp = getattr(layer, "mlp", None)
    if mlp is None or not all(hasattr(mlp, a) for a in ("gate_proj", "up_proj", "down_proj")):
        raise ValueError(f"{type(layer).__name__} has no SwiGLU MLP (gate_proj/up_proj/down_proj)")
    return mlp


def _ids(model: WhiteBoxModel, ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor([list(ids)], device=model.model.device)


def _run(model: WhiteBoxModel, ids: Sequence[int], metric_fn: MetricFn, *, grad: bool,
         neuron_layers: Sequence[int] = ()) -> tuple[float, dict, dict, dict]:
    """One forward pass capturing every block output and the chosen layers' down_proj inputs and
    gate pre-activations; with ``grad`` the metric is back-propagated to all of them."""
    layers = model._decoder_layers()  # noqa: SLF001
    resid: dict[int, torch.Tensor] = {}
    neur: dict[int, torch.Tensor] = {}
    gate: dict[int, torch.Tensor] = {}
    handles = []

    def keep(store, key):
        def fn(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            if grad:
                t.retain_grad()
            store[key] = t
        return fn

    def keep_in(store, key):
        def fn(_m, args):
            if grad:
                args[0].retain_grad()
            store[key] = args[0]
        return fn

    for i, lay in enumerate(layers):
        handles.append(lay.register_forward_hook(keep(resid, i)))
    for i in neuron_layers:
        mlp = _mlp(layers[i])
        handles.append(mlp.down_proj.register_forward_pre_hook(keep_in(neur, i)))
        handles.append(mlp.gate_proj.register_forward_hook(keep(gate, i)))
    # gradients reach the activations through the input leaf only; parameters stay frozen, so
    # no weight gradient is accumulated (a 1B model's would be ~4 GB in fp32)
    flags = [(p_, p_.requires_grad) for p_ in model.model.parameters()]
    for p_, _ in flags:
        p_.requires_grad_(False)
    try:
        x = _ids(model, ids)
        emb = model.model.get_input_embeddings()(x).detach()
        with torch.set_grad_enabled(grad):
            if grad:  # a leaf at the input, so every downstream activation carries a gradient
                emb.requires_grad_(True)
            logits = model.model(inputs_embeds=emb).logits[0, -1]
            m = metric_fn(logits)
            if grad:
                m.backward()
    finally:
        for h in handles:
            h.remove()
        for p_, f in flags:
            p_.requires_grad_(f)
    return float(m.detach()), resid, neur, gate


def atp(model: WhiteBoxModel, clean_ids: Sequence[int], corrupt_ids: Sequence[int],
        metric_fn: MetricFn, neuron_layers: Sequence[int] = ()) -> AtPResult:
    """AtP estimates of patching each residual site (and each neuron of ``neuron_layers``) from
    the corrupted run into the clean run."""
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("AtP needs token-aligned inputs (equal length)")
    m_c, rc, nc, gc = _run(model, clean_ids, metric_fn, grad=True, neuron_layers=neuron_layers)
    with torch.no_grad():
        m_r, rr, nr, gr = _run(model, corrupt_ids, metric_fn, grad=False,
                               neuron_layers=neuron_layers)
        res = torch.stack([((rr[i] - rc[i]) * rc[i].grad).sum(-1)[0]
                           for i in range(len(rc))]).float().cpu().numpy()
        neurons = {i: ((nr[i] - nc[i]) * nc[i].grad)[0].float().cpu().numpy()
                   for i in neuron_layers}
        flips = {i: (torch.sign(gr[i]) != torch.sign(gc[i]))[0].cpu().numpy()
                 for i in neuron_layers}
    return AtPResult(residual=res, metric_clean=m_c, metric_corrupt=m_r, neurons=neurons,
                     gate_sign_flip=flips)


@torch.no_grad()
def patch_residual(model: WhiteBoxModel, clean_ids: Sequence[int], corrupt_ids: Sequence[int],
                   metric_fn: MetricFn, sites: Sequence[tuple[int, int]]) -> np.ndarray:
    """The real patch, site by site: the clean run with block ``layer``'s output at ``position``
    replaced by the corrupted run's; returns the metric change per site."""
    if len(clean_ids) != len(corrupt_ids):
        raise ValueError("patching needs token-aligned inputs (equal length)")
    layers = model._decoder_layers()  # noqa: SLF001
    m_c, _, _, _ = _run(model, clean_ids, metric_fn, grad=False)
    _, rr, _, _ = _run(model, corrupt_ids, metric_fn, grad=False)
    out = []
    for layer, pos in sites:
        src = rr[layer][0, pos].clone()

        def swap(_m, _i, o, pos=pos, src=src):
            t = o[0] if isinstance(o, tuple) else o
            t = t.clone()
            t[0, pos] = src
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        h = layers[layer].register_forward_hook(swap)
        try:
            logits = model.model(input_ids=_ids(model, clean_ids)).logits[0, -1]
        finally:
            h.remove()
        out.append(float(metric_fn(logits)) - m_c)
    return np.asarray(out)
