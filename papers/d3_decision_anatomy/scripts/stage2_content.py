#!/usr/bin/env python3
"""C1 Stage 2 — what the top-k writing heads READ. Pre-registered in ../PREREGISTRATION.md (§2).

For each top-k head (from Stage 1): (i) attention source distribution over position classes
(t_inst / content / template-sink) at the decision token; (ii) value-side content = the head's
attention-weighted source residual ("read vector") projected onto the mean_content V_moral subspace
and onto the t_inst harm direction (Zhao et al. 2025, arXiv:2507.11878, harmfulness != refusal).

Hypotheses (both pre-declared): copy-head-for-harm (t_inst attention plurality + harm loading >
V_moral loading by M) vs moral-content-reading (V_moral loading clears its band-min - M). Every
value-side projection ships on a ladder (null q95 + M; V_moral band) with an MDE
(instrument-calibration). Pure math is model-free + unit-tested; the hook pass uses
output_attentions (eager attention required) + the L_ref input residual.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from stage1_attribution import _unit  # noqa: E402

MARGIN_M = 0.05
K = 1000


def _ortho(dirs):
    Q, _ = np.linalg.qr(np.stack(dirs, axis=1))
    return Q


def _frac(Q, v):
    return float(np.linalg.norm(Q.T @ _unit(v)))


def source_distribution(attn_row: np.ndarray, spans: dict) -> dict:
    """Aggregate a head's decision-token attention over position classes. spans maps class ->
    list of source indices; classes: t_inst, content, template (+ any leftover = other)."""
    a = np.asarray(attn_row, np.float64)
    a = a / (a.sum() + 1e-12)
    out = {cls: round(float(a[idx].sum()), 4) for cls, idx in spans.items()}
    out["plurality"] = max(out, key=out.get) if out else None
    return out


def value_side(read_vec: np.ndarray, vmoral_basis: np.ndarray, harm_dir: np.ndarray) -> dict:
    """Project a head's read vector onto V_moral (fraction) and the harm direction (|cos|)."""
    return {"vmoral_frac": round(_frac(vmoral_basis, read_vec), 4),
            "harm_abs_cos": round(abs(float(_unit(read_vec) @ _unit(harm_dir))), 4)}


def value_null_q95(Xc: np.ndarray, vmoral_basis: np.ndarray, rng, k: int = K) -> float:
    """Covariance-matched null for the V_moral fraction, from the value-input (channel) sample."""
    n = Xc.shape[0]
    fr = [_frac(vmoral_basis, Xc.T @ rng.standard_normal(n)) for _ in range(k)]
    return float(np.percentile(fr, 95))


def classify_head(src: dict, vs: dict, band_min: float, null_q95: float,
                  m: float = MARGIN_M) -> dict:
    """Both pre-registered hypotheses, per head."""
    reads_vmoral = vs["vmoral_frac"] > null_q95 + m
    copy_for_harm = (src.get("plurality") == "t_inst"
                     and vs["harm_abs_cos"] > vs["vmoral_frac"] + m)
    moral_reading = vs["vmoral_frac"] >= band_min - m
    return {"reads_vmoral_above_null": bool(reads_vmoral),
            "copy_head_for_harm": bool(copy_for_harm),
            "moral_content_reading": bool(moral_reading),
            "label": ("copy-head-for-harm" if copy_for_harm else
                      "moral-content-reading" if moral_reading else "neither")}


# --------------------------- hook pass (model-dependent; pod) ---------------------------

def stage2_pass(model, prompt: str, top_heads: list, l_ref: int, spans_fn) -> dict:
    """Per top head (at ANY of its layers, not only l_ref -- Amendment 3 per-layer coverage):
    decision-token attention row + read vector (attention-weighted input residual to that head's
    layer). Requires eager attention (output_attentions). spans_fn(model, prompt) -> position spans
    dict (token-index classes, layer-independent). Returns {head: {attn_row, read_vec, source_dist}}."""
    import torch
    tok = model.tokenizer
    text = (tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True)
            if getattr(tok, "chat_template", None) else prompt)
    layers = sorted({int(l) for (l, _) in top_heads})
    resid_in = {}

    def make_pre(l):
        def pre(module, inp):
            resid_in[l] = inp[0][0].detach().float().cpu().numpy()  # (seq, hidden) input to layer l
        return pre
    hooks = [model._get_layer_module(l).register_forward_pre_hook(make_pre(l)) for l in layers]
    # SDPA/flash don't materialize attention weights -> force eager for this pass (restore after).
    cfg = model.model.config
    prev = getattr(cfg, "_attn_implementation", None)
    try:
        try:
            model.model.set_attn_implementation("eager")
        except Exception:
            cfg._attn_implementation = "eager"
        with torch.no_grad():
            out = model.model(**tok(text, return_tensors="pt").to(model._device),
                              output_attentions=True)
    finally:
        for h in hooks:
            h.remove()
        try:
            model.model.set_attn_implementation(prev or "sdpa")
        except Exception:
            cfg._attn_implementation = prev
    if not out.attentions or len(out.attentions) <= max(layers) or out.attentions[max(layers)] is None:
        raise RuntimeError(f"attention weights not materialized up to layer {max(layers)} "
                           f"(got {type(out.attentions).__name__}, len "
                           f"{len(out.attentions) if out.attentions else 0}); eager attention failed")
    spans = spans_fn(model, text)
    res = {}
    for (l, hh) in top_heads:
        a = out.attentions[l][0, hh, -1, :].float().cpu().numpy()  # decision-token attention row
        read_vec = (a[:, None] * resid_in[l]).sum(0)               # attention-weighted source residual
        res[(l, hh)] = {"attn_row": a, "read_vec": read_vec,
                        "source_dist": source_distribution(a, spans)}
    return res
