#!/usr/bin/env python3
"""C1 Stage 1 — per-head direct write attribution onto the refusal direction at the decision-site
token. Pre-registered in ../PREREGISTRATION.md (§1) + Amendment 1.

Method credit: per-head OV attribution of safety heads is Sahara-style (Zhou et al. 2024,
arXiv:2410.13708) — applied here, not a C1 novelty. The novelty is the calibrated-channel specificity
score + what Stage 2 does with the writers (content transport).

The residual stream is LINEAR: resid_{L}[t] = embed[t] + sum_{l<=L} (attn_l[t] + mlp_l[t]), and
attn_l[t] = sum_h W_O^{l,h} z_h[t]. So <resid_{L_ref}[decision], r_hat> decomposes EXACTLY into
per-(layer,head) OV writes + per-layer MLP writes + embed. The reconstruction control (a TWO-SIDED
band 0.90 <= recon <= 1.10; instrument-calibration positive control) catches (a) missing components
(undershoot), or (b) an un-folded norm on the block output (overshoot).

REORDERED NORM (OLMo-2/3, ANOMALY A3): these families apply RMSNorm to the ATTENTION/MLP *output*
before the residual add (post_attention_layernorm, post_feedforward_layernorm; no input norm), so the
real residual write is norm(sum_h W_O^h z_h), not the raw sum -- the naive decomposition overshoots
(~3x on Olmo-3-7B). RMSNorm is diagonal at a fixed token: norm(x) = (gamma / rms(x)) (.) x, so folding
the per-dim gain g = gamma/rms onto each pre-norm component write is EXACT (sum -> <norm(total), r>).
This is the pre-registered LN-fold escalation; it fires automatically for reordered-norm models and is
a no-op for pre-norm models (Llama/Qwen), whose block output IS the residual write (recon ~1 natively).

Amendment 1: head specificity = <contrib_h, r_hat> - mean_j <contrib_h, c_hat_j> over a norm-matched
CHANNEL control basis {c_hat_j} (the decision-token channel minus r_hat) -- at PR~15 raw write
onto r_hat conflates "refusal-writer" with "token-writer". Causal ablation is mean/resample, not
zeroing (built in Stage 2 / the causal cells; the harness exposes the head list + resample hook).

The pure-math (channel basis, specificity, reconstruction, k-selection, MLP fraction) is model-free
and unit-tested in local_test.py. attribute() is the hook pass (VALIDATE=1 tiny-model smoke on pod).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "6_cross_model" / "scripts"))

K_THRESH = 0.80   # cumulative |specificity| for k-selection
K_CAP = 10
MLP_BRANCH = 0.50  # MLP write fraction above this -> Jacobian stage
RECON_FLOOR = 0.90  # two-sided reconstruction band ...
RECON_CEIL = 1.10   # ... an un-folded norm shows up as OVERSHOOT (recon > ceil), not undershoot


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def rms_gain(x: np.ndarray, weight: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-dim fold factor of an RMSNorm applied to x: g = weight / sqrt(mean(x^2) + eps). For a
    reordered-norm block (OLMo-2/3) the residual write is norm(block_out); multiplying the pre-norm
    per-component write vectors by g recovers the EXACT residual contribution, because RMSNorm is a
    diagonal (per-dim gain * shared scalar) map at a fixed token -- sum_h (contrib_h (.) g) = norm(sum_h
    contrib_h). No-op region: pre-norm models never call this (their block output is the write)."""
    x = np.asarray(x, np.float64)
    rms = float(np.sqrt(np.mean(x ** 2) + eps))
    return np.asarray(weight, np.float64) / rms


def reconstruction_ok(recon: float) -> bool:
    """Two-sided: undershoot (missing components) OR overshoot (un-folded block norm) both fail."""
    return bool(RECON_FLOOR <= recon <= RECON_CEIL)


# --------------------------- pure math (model-free, unit-tested) ---------------------------

def channel_control_basis(X: np.ndarray, r_hat: np.ndarray, k_ch: int | None = None) -> np.ndarray:
    """Orthonormal basis of the decision-token CHANNEL with the refusal direction projected out --
    the 'same channel, not r_hat' control directions (Amendment 1). X = decision-token act sample
    (n, d). k_ch defaults to the participation ratio (rounded), capped to rank."""
    Xc = np.asarray(X, np.float64) - np.asarray(X, np.float64).mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    if k_ch is None:
        ev = (S ** 2)
        k_ch = int(round((ev.sum() ** 2) / (ev ** 2).sum()))  # participation ratio
    k_ch = max(1, min(k_ch, Vt.shape[0]))
    C = Vt[:k_ch].T                                   # (d, k_ch) top principal directions
    r = _unit(r_hat)
    C = C - np.outer(r, r @ C)                        # project r_hat out of the channel basis
    Q, _ = np.linalg.qr(C)
    # keep columns with meaningful norm (drop the one collapsed by removing r_hat)
    keep = [i for i in range(Q.shape[1]) if np.linalg.norm(C[:, i]) > 1e-6]
    return Q[:, keep] if keep else Q[:, : max(1, k_ch - 1)]


def head_specificity(contribs: dict, r_hat: np.ndarray, Qc: np.ndarray) -> dict:
    """{head_key: <contrib, r_hat> - mean_j <contrib, c_hat_j>}. contribs values are residual
    contribution VECTORS (hidden,). r_hat, Qc columns are unit / orthonormal (norm-matched)."""
    r = _unit(r_hat)
    out = {}
    for h, c in contribs.items():
        c = np.asarray(c, np.float64)
        write = float(c @ r)
        chan = float(np.mean(Qc.T @ c)) if Qc.shape[1] else 0.0
        out[h] = {"write": round(write, 5), "specificity": round(write - chan, 5),
                  "channel_mean": round(chan, 5)}
    return out


def reconstruction_ratio(component_writes: list[float], target: float) -> float:
    """Sum of component writes onto r_hat / the true <resid, r_hat>. ~1.0 if the decomposition is
    complete (residual is linear); < RECON_FLOOR -> escalate LN handling."""
    return float(sum(component_writes) / (target + 1e-12))


def kselect(spec_scores: dict, thresh: float = K_THRESH, cap: int = K_CAP) -> dict:
    """Smallest k of heads whose cumulative |specificity| >= thresh of the total, capped at cap.
    Returns k, the ranked head list, and the cumulative curve."""
    ranked = sorted(spec_scores.items(), key=lambda kv: -abs(kv[1]["specificity"]))
    tot = sum(abs(v["specificity"]) for _, v in ranked) + 1e-12
    cum, curve, k = 0.0, [], None
    for i, (h, v) in enumerate(ranked, 1):
        cum += abs(v["specificity"])
        curve.append(round(cum / tot, 4))
        if k is None and cum / tot >= thresh:
            k = min(i, cap)
    return {"k": min(k or len(ranked), cap), "ranked_heads": [h for h, _ in ranked],
            "sparsity_curve": curve, "top": ranked[: (k or cap)]}


def mlp_fraction(head_writes: list[float], mlp_writes: list[float]) -> dict:
    hw = sum(abs(x) for x in head_writes)
    mw = sum(abs(x) for x in mlp_writes)
    frac = float(mw / (hw + mw + 1e-12))
    return {"mlp_write_fraction": round(frac, 4),
            "jacobian_branch": bool(frac > MLP_BRANCH),
            "note": "MLP > 0.50 -> head story incomplete, add the Jacobian stage (§1 branch)"}


# --------------------------- hook pass (model-dependent; pod / MPS) ---------------------------

def attribute(model, prompt: str, r_hat: np.ndarray, l_ref: int) -> dict:
    """Decompose <resid_{l_ref}[decision-token], r_hat> into per-(layer,head) OV writes + per-layer
    MLP writes + embed, at the last (decision-site) token. Returns writes + contribution vectors +
    the reconstruction. Standard HF decoder layout (self_attn.o_proj, mlp)."""
    import torch
    tok = model.tokenizer
    text = (tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                    add_generation_prompt=True)
            if getattr(tok, "chat_template", None) else prompt)
    cfg = model.model.config
    n_heads = cfg.num_attention_heads
    hidden = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", hidden // n_heads)
    r = torch.tensor(_unit(r_hat), dtype=model._dtype, device=model._device)

    cap = {"per_head": {}, "mlp": {}, "resid": None, "embed": None}
    hooks = []

    def oproj_pre(l):
        def hook(module, inp):
            z = inp[0][0, -1, :]                       # (n_heads*head_dim,) at last token
            W = module.weight                          # (hidden, n_heads*head_dim)
            for h in range(n_heads):
                sl = slice(h * head_dim, (h + 1) * head_dim)
                cap["per_head"][(l, h)] = (W[:, sl] @ z[sl]).detach().float().cpu().numpy()
        return hook

    def mlp_fwd(l):
        def hook(module, inp, out):
            cap["mlp"][l] = out[0, -1, :].detach().float().cpu().numpy()
        return hook

    def layer_fwd(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap["resid"] = t[0, -1, :].detach().float().cpu().numpy()

    for l in range(l_ref + 1):
        lm = model._get_layer_module(l)
        hooks.append(lm.self_attn.o_proj.register_forward_pre_hook(oproj_pre(l)))
        hooks.append(lm.mlp.register_forward_hook(mlp_fwd(l)))
    hooks.append(model._get_layer_module(l_ref).register_forward_hook(layer_fwd))
    emb = model.model.get_input_embeddings()
    hooks.append(emb.register_forward_hook(
        lambda m, i, o: cap.__setitem__("embed", o[0, -1, :].detach().float().cpu().numpy())))

    try:
        with torch.no_grad():
            model.model(**tok(text, return_tensors="pt").to(model._device))
    finally:
        for h in hooks:
            h.remove()

    # Reordered-norm fold (A3): OLMo-2/3 write norm(attn_out)/norm(mlp_out) to the residual, so fold
    # the per-layer RMSNorm gain onto each pre-norm component write vector -> exact residual contribs.
    reordered = hasattr(model._get_layer_module(0), "post_feedforward_layernorm")
    if reordered:
        for l in range(l_ref + 1):
            lm = model._get_layer_module(l)
            a_out = np.sum([cap["per_head"][(l, h)] for h in range(n_heads)], axis=0)  # o_proj out (norm in)
            na = lm.post_attention_layernorm
            g_a = rms_gain(a_out, na.weight.detach().float().cpu().numpy(),
                           getattr(na, "variance_epsilon", getattr(na, "eps", 1e-6)))
            for h in range(n_heads):
                cap["per_head"][(l, h)] = cap["per_head"][(l, h)] * g_a
            nm = lm.post_feedforward_layernorm
            g_m = rms_gain(cap["mlp"][l], nm.weight.detach().float().cpu().numpy(),
                           getattr(nm, "variance_epsilon", getattr(nm, "eps", 1e-6)))
            cap["mlp"][l] = cap["mlp"][l] * g_m

    rn = _unit(r_hat)
    target = float(cap["resid"] @ rn)
    head_writes = {k: float(v @ rn) for k, v in cap["per_head"].items()}
    mlp_writes = {l: float(v @ rn) for l, v in cap["mlp"].items()}
    embed_write = float(cap["embed"] @ rn)
    recon = reconstruction_ratio(list(head_writes.values()) + list(mlp_writes.values()) + [embed_write],
                                 target)
    return {"target": target, "reconstruction": round(recon, 4),
            "reconstruction_ok": reconstruction_ok(recon), "reordered_norm": bool(reordered),
            "per_head_contrib": cap["per_head"], "head_writes": head_writes,
            "mlp_writes": mlp_writes, "embed_write": embed_write}


def main() -> None:
    ap = argparse.ArgumentParser(description="C1 Stage 1 per-head write attribution.")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--key", default="olmo3")
    ap.add_argument("--refusal-npz", required=True)
    ap.add_argument("--act-sample-npz", required=True, help="decision-token chat act sample (channel)")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--prompt", default="How can I kill a person?")
    ap.add_argument("--out", default=str(HERE.parent / "outputs"))
    args = ap.parse_args()

    from deepsteer.directions import extraction as du
    if os.environ.get("VALIDATE") == "1":
        args.model = "allenai/OLMo-2-0425-1B"; args.layer = 8
    model = du.load_whitebox(args.model)
    r_hat = np.load(args.refusal_npz)["refusal"].astype(np.float64)
    X = np.load(args.act_sample_npz)["X"].astype(np.float64)
    Qc = channel_control_basis(X, r_hat)
    res = attribute(model, args.prompt, r_hat, args.layer)
    spec = head_specificity(res["per_head_contrib"], r_hat, Qc)
    ks = kselect(spec)
    mf = mlp_fraction(list(res["head_writes"].values()), list(res["mlp_writes"].values()))
    out = {"model": args.model, "layer": args.layer, "channel_dim": int(Qc.shape[1]),
           "reconstruction": res["reconstruction"], "reconstruction_ok": res["reconstruction_ok"],
           "reordered_norm": res["reordered_norm"],
           "mlp": mf, "k": ks["k"], "top_heads": [{"head": list(h), **spec[h]} for h in ks["ranked_heads"][:ks["k"]]],
           "sparsity_curve": ks["sparsity_curve"]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"stage1_attribution_{args.key}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if not res["reconstruction_ok"]:
        print(f"WARNING: reconstruction {res['reconstruction']} outside [{RECON_FLOOR}, {RECON_CEIL}] "
              f"(reordered_norm={res['reordered_norm']}) -> a component is still un-folded")


if __name__ == "__main__":
    main()
