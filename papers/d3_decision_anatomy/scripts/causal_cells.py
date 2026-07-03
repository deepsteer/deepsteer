"""C1 causal cells — interchange patching at content positions, decision read at the bottleneck +
behaviorally. Pre-registered in ../PREREGISTRATION.md (§3) + Amendment 1.

Cells: (a) twin patch (outcome = judgment; refusal via request-twins), (b) V_moral-restricted vs
full patch + the TRANSPORT POSITIVE CONTROL, (c) XSTest harm-flip vs surface-held, (d) top-head
mean/resample ablation (NOT zeroing). The decisive-cell verdict logic (b) is pure + unit-tested;
the patch/generation passes are model-dependent (pod).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "d2_decision_coupling" / "scripts"))
from stage1_attribution import _unit  # noqa: E402


# --------------------------- pure decision logic (model-free, unit-tested) ---------------------------

def mde_bootstrap(deltas: list[float], rng, b: int = 2000, alpha: float = 0.05) -> float:
    """Minimum detectable effect: the 95th percentile of |bootstrap-resampled mean| under the null
    (deltas re-centered to zero) -- the smallest true |mean delta| we could distinguish at this n."""
    d = np.asarray(deltas, np.float64)
    d0 = d - d.mean()
    boots = [abs(rng.choice(d0, len(d0), replace=True).mean()) for _ in range(b)]
    return float(np.percentile(boots, 100 * (1 - alpha)))


def cell_b_verdict(full_refusal_delta: float, restricted_refusal_delta: float,
                   restricted_judgment_delta: float, mde_refusal: float,
                   mde_judgment: float) -> dict:
    """Amendment 1 decisive-cell rule, transport-control-gated. All four branches pre-declared."""
    transport_ok = abs(restricted_judgment_delta) > mde_judgment      # restricted CAN move judgment
    full_moves = abs(full_refusal_delta) > mde_refusal
    restricted_moves = abs(restricted_refusal_delta) > mde_refusal
    if not full_moves:
        return {"verdict": "no_content_effect", "transport_control_passed": bool(transport_ok),
                "reason": "full patch does not move refusal beyond MDE -> surface-only branch; the "
                          "forced-coupling sign-flip gets a dedicated re-examination"}
    if not transport_ok:
        return {"verdict": "uninformative", "transport_control_passed": False,
                "reason": "V_moral-restricted patch cannot move even judgment -> rank-3 under-transfer "
                          "not excluded; only the positive branch of this cell carries weight"}
    if restricted_moves:
        return {"verdict": "vmoral_is_read_substrate", "transport_control_passed": True,
                "reason": "restricted patch moves refusal -> geometric orthogonality at the readout + "
                          "causal transport of V_moral content into the writers (complete "
                          "through-weights story with anatomy)"}
    return {"verdict": "reads_non_vmoral_features", "transport_control_passed": True,
            "reason": "full moves refusal, V_moral-restricted (which CAN move judgment) does not -> "
                      "refusal reads moral content through NON-V_moral features -> explains every "
                      "geometric null in the program"}


def ablation_outlier(refusal_delta_top: float, random_head_deltas: list[float]) -> dict:
    """Cell (d): top-head ablation refusal effect is head-specific iff it is an outlier BELOW the
    matched-random-head null (mean/resample ablation, not zeroing -- enforced in the pass)."""
    arr = np.asarray(random_head_deltas, np.float64)
    q10 = float(np.percentile(arr, 10))
    return {"top_head_refusal_delta": round(refusal_delta_top, 4),
            "random_head_q10": round(q10, 4),
            "head_specific": bool(refusal_delta_top < q10)}


def ratio_of_ratios(full_ref, restr_ref, full_jud, restr_jud, rng, b: int = 2000,
                    m_ratio: float = 0.15) -> dict:
    """Amendment 3 verdict test (pure, unit-tested). Compare the fraction of each full-patch effect
    that survives V_moral restriction: R_refusal = mean(restr_ref)/mean(full_ref) on request-twins,
    R_judgment = mean(restr_jud)/mean(full_jud) on compositional twins. Bootstrap the difference,
    resampling each twin set independently. reads_non_vmoral_features STANDS iff (R_j - R_r) >=
    m_ratio AND the 95% CI is entirely positive; CI includes 0 -> under_transfer (rank sweep); CI
    entirely negative -> reads_vmoral_more (surprising, refusal keeps MORE V_moral than judgment)."""
    fr, rr = np.asarray(full_ref, float), np.asarray(restr_ref, float)
    fj, rj = np.asarray(full_jud, float), np.asarray(restr_jud, float)
    R_ref, R_jud = float(rr.mean() / fr.mean()), float(rj.mean() / fj.mean())
    diffs = []
    for _ in range(b):
        i = rng.integers(0, len(fr), len(fr))
        j = rng.integers(0, len(fj), len(fj))
        diffs.append(rj[j].mean() / fj[j].mean() - rr[i].mean() / fr[i].mean())
    lo, hi = (float(x) for x in np.percentile(diffs, [2.5, 97.5]))
    diff = R_jud - R_ref
    if lo > 0:
        verdict = "reads_non_vmoral_features" if diff >= m_ratio else \
                  "reads_non_vmoral_features_small_margin"
    elif hi < 0:
        verdict = "reads_vmoral_more"
    else:
        verdict = "under_transfer"       # ratios comparable -> redesign with the rank sweep
    return {"R_refusal": round(R_ref, 4), "R_judgment": round(R_jud, 4), "diff": round(diff, 4),
            "ci": [round(lo, 4), round(hi, 4)], "ci_excludes_0": bool(lo > 0 or hi < 0),
            "stands": bool(verdict == "reads_non_vmoral_features"), "verdict": verdict,
            "m_ratio": m_ratio}


def random_ortho_basis(hidden: int, k: int, rng, exclude_Q=None) -> np.ndarray:
    """Random rank-k orthonormal basis (optionally orthogonalized against exclude_Q's columns) for
    the random-rank-k restricted CONTROL -- shows the V_moral restriction is not 'any rank-k fails'."""
    M = rng.standard_normal((hidden, k))
    if exclude_Q is not None:
        M = M - exclude_Q @ (exclude_Q.T @ M)
    Q, _ = np.linalg.qr(M)
    return Q[:, :k]


def rankk_moral_basis(source_dirs: list, k: int) -> np.ndarray:
    """Over-complete moral basis truncated to rank k for the rank sweep k in {1,3,8,16}: QR of the
    stacked source directions (per-source mean_content dirs), then the first k columns. Rank is
    capped at the number of available source directions."""
    Q, _ = np.linalg.qr(np.stack(source_dirs, axis=1))
    return Q[:, :min(k, Q.shape[1])]


# --------------------------- patch mechanics (model-dependent; pod) ---------------------------

def _templated(model, text: str) -> str:
    """Chat-template so the last token is the decision site (final_pre_assistant), matching where
    refusal/judgment live (D2 convention)."""
    tok = model.tokenizer
    return (tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True)
            if getattr(tok, "chat_template", None) else text)


def patch_positions(model, a: str, b: str) -> tuple[list[int], list[int]]:
    """Token indices of the flipped content span in each (chat-templated) member (shared-prefix +
    flipped-span rule, Amendment 1). Raises if the pair cannot be aligned within budget."""
    tok = model.tokenizer
    ta = tok(_templated(model, a), add_special_tokens=False)["input_ids"]
    tb = tok(_templated(model, b), add_special_tokens=False)["input_ids"]
    p = 0
    for x, y in zip(ta, tb):
        if x != y:
            break
        p += 1
    sa, sb = 0, 0
    for x, y in zip(ta[::-1], tb[::-1]):
        if x != y or sa >= len(ta) - p - 1 or sb >= len(tb) - p - 1:
            break
        sa += 1; sb += 1
    idx_a = list(range(p, len(ta) - sa))
    idx_b = list(range(p, len(tb) - sb))
    if not idx_a or not idx_b:
        raise ValueError("twin pair not alignable (empty flipped span)")
    return idx_a, idx_b


def _resid_at(model, text, layer, positions):
    import torch
    tok = model.tokenizer
    cap = {}
    hh = model._get_layer_module(layer).register_forward_pre_hook(
        lambda m, inp: cap.__setitem__("x", inp[0][0].detach().clone()))
    try:
        with torch.no_grad():
            model.model(**tok(_templated(model, text), return_tensors="pt").to(model._device))
    finally:
        hh.remove()
    return cap["x"][positions]  # (len(pos), hidden)


def _patch_hook_factory(tgt_pos, src_comp, Q, restrict_mode, sig=None):
    """Forward-pre-hook that swaps content at tgt_pos. FULL (Q None): replace with src_comp.
    SUBSPACE: swap only the in-Q component (tgt keeps off-Q). COMPLEMENT: swap only the off-Q
    component. STANDARDIZE (sig given, ANOMALIES A1): the restriction is done in per-dim-standardized
    coordinates (÷ sig), then mapped back to raw (× sig) for injection, so the outlier dims do not
    dominate the V_moral restriction on Llama/Qwen/GPT-OSS. src_comp + Q are in the standardized space
    when sig is given."""
    def patch_hook(module, inp):
        x = inp[0]
        pos = [p for p in tgt_pos if p < x.shape[1]]
        if pos:
            cur = x[0, pos, :]
            cur_w = cur / sig if sig is not None else cur       # -> work (standardized) space
            if Q is None:
                new_w = src_comp.expand(len(pos), -1)
            elif restrict_mode == "complement":
                new_w = (cur_w @ Q) @ Q.T + src_comp            # keep tgt in-Q, swap src off-Q
            else:                                               # subspace
                new_w = cur_w - (cur_w @ Q) @ Q.T + src_comp    # keep tgt off-Q, swap src in-Q
            x[0, pos, :] = new_w * sig if sig is not None else new_w   # -> raw space for injection
        return (x,) + inp[1:]
    return patch_hook


def interchange(model, src, tgt, src_pos, tgt_pos, layer, r_hat, read_layer=None,
                restrict_Q=None, restrict_mode="subspace", sigma=None):
    """Patch src's content into tgt at `layer`; return the decision-token projection onto r_hat read
    at `read_layer` (default = layer). Length-agnostic: the src flipped span is MEAN-POOLED to one
    content vector and broadcast across tgt's flipped-span positions (an aggregate-content swap, so
    differently-tokenized moral-intent spans are comparable). Modes: FULL (restrict_Q None) replaces
    the tgt content vector; SUBSPACE (restrict_Q hidden,k) swaps only the in-subspace component;
    COMPLEMENT swaps only the off-subspace component (Amendment 3 confirmatory cell). tgt positions
    guarded against the current seq length (safe under KV-cached decoding).

    STANDARDIZE (sigma given, ANOMALIES A1): the restriction runs in per-dim-standardized coordinates
    and the readout onto r_hat is standardized (r_hat + restrict_Q must be supplied in the same
    standardized space). Patch injection is still raw; only the restriction basis and the readout
    metric are standardized, so the A1 outlier dims cannot dominate the V_moral restriction or the
    projection on Llama/Qwen/GPT-OSS."""
    import torch
    read_layer = layer if read_layer is None else read_layer
    src_resid = _resid_at(model, src, layer, src_pos).mean(0, keepdim=True)   # (1, hidden) content mean
    sig = None if sigma is None else torch.tensor(sigma, dtype=src_resid.dtype, device=src_resid.device)
    src_w = src_resid / sig if sig is not None else src_resid                 # -> work space
    Q = None if restrict_Q is None else torch.tensor(restrict_Q, dtype=src_resid.dtype,
                                                     device=src_resid.device)
    if Q is None:
        src_comp = src_w
    elif restrict_mode == "complement":
        src_comp = src_w - (src_w @ Q) @ Q.T                    # src's off-subspace component (work space)
    else:
        src_comp = (src_w @ Q) @ Q.T                            # src's in-subspace component (work space)
    r = torch.tensor(_unit(r_hat), dtype=torch.float32, device=model._device)
    sig_r = None if sigma is None else torch.tensor(sigma, dtype=torch.float32, device=model._device)
    proj = {}

    def read_hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        tw = (t[0, -1, :].float() / sig_r) if sig_r is not None else t[0, -1, :].float()
        proj["p"] = float(tw @ r)

    tok = model.tokenizer
    h1 = model._get_layer_module(layer).register_forward_pre_hook(
        _patch_hook_factory(tgt_pos, src_comp, Q, restrict_mode, sig))
    h2 = model._get_layer_module(read_layer).register_forward_hook(read_hook)
    try:
        with torch.no_grad():
            model.model(**tok(_templated(model, tgt), return_tensors="pt").to(model._device))
    finally:
        h1.remove(); h2.remove()
    return proj["p"]


def baseline_proj(model, text: str, read_layer: int, r_hat, sigma=None) -> float:
    """Unpatched decision-token projection onto r_hat (the patch baseline). STANDARDIZE (sigma given):
    standardized readout, r_hat supplied in the standardized space."""
    import torch
    tok = model.tokenizer
    r = torch.tensor(_unit(r_hat), dtype=torch.float32, device=model._device)
    sig_r = None if sigma is None else torch.tensor(sigma, dtype=torch.float32, device=model._device)
    cap = {}

    def read_hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        tw = (t[0, -1, :].float() / sig_r) if sig_r is not None else t[0, -1, :].float()
        cap["p"] = float(tw @ r)
    h = model._get_layer_module(read_layer).register_forward_hook(read_hook)
    try:
        with torch.no_grad():
            model.model(**tok(_templated(model, text), return_tensors="pt").to(model._device))
    finally:
        h.remove()
    return cap["p"]


def generate_plain(model, tgt: str, max_tokens: int = 48) -> str:
    """Chat-templated greedy completion with no patch (the behavioral baseline)."""
    import torch
    tok = model.tokenizer
    ids = tok(_templated(model, tgt), return_tensors="pt").to(model._device)
    with torch.no_grad():
        out = model.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)


def generate_under_patch(model, src, tgt, src_pos, tgt_pos, layer, restrict_Q=None,
                         restrict_mode="subspace", max_tokens: int = 48) -> str:
    """Install the interchange patch (same modes as `interchange`) and GENERATE, returning the
    completion text for behavioral refusal classification -- the behavioral upgrade from the
    projection readout. The patch fires only on the prefill forward (the tgt_pos guard filters it out
    during KV-cached decode, where seq len is 1)."""
    import torch
    src_resid = _resid_at(model, src, layer, src_pos).mean(0, keepdim=True)
    Q = None if restrict_Q is None else torch.tensor(restrict_Q, dtype=src_resid.dtype,
                                                     device=src_resid.device)
    if Q is None:
        src_comp = src_resid
    elif restrict_mode == "complement":
        src_comp = src_resid - (src_resid @ Q) @ Q.T
    else:
        src_comp = (src_resid @ Q) @ Q.T
    tok = model.tokenizer
    ids = tok(_templated(model, tgt), return_tensors="pt").to(model._device)
    tgt_len = ids["input_ids"].shape[1]
    pos = [p for p in tgt_pos if p < tgt_len]
    h = model._get_layer_module(layer).register_forward_pre_hook(
        _patch_hook_factory(pos, src_comp, Q, restrict_mode))
    try:
        with torch.no_grad():
            out = model.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
    finally:
        h.remove()
    return tok.decode(out[0, tgt_len:], skip_special_tokens=True)


def severity_psychometric(model, twins, is_refusal, viol_thresh: float = 0.5,
                          foll_comply_thresh: float = 0.5, boundary=(0.4, 0.7),
                          max_tokens: int = 48) -> dict:
    """Refusal-vs-severity psychometric curve (Amendment 4/8). Per level, the baseline refusal rate of
    the following (benign) vs violating members. OPERATING BAND = levels where the violating member
    refuses (>= viol_thresh) AND the following complies. BOUNDARY BAND (Amendment 8, dynamic range) =
    levels where the violating refusal rate is in `boundary` [0.4,0.7] AND the following complies -- the
    unsaturated levels where the patch has room to move refusal. Selection-regression: the per-level
    rate is split-half re-estimated and the shrinkage (max half-to-half gap over selected levels)
    reported. twins = (foundation, level, following, violating); level may be an int or a sublevel str."""
    import numpy as np
    from collections import defaultdict
    by = defaultdict(lambda: {"f": [], "v": []})
    keys_seen = []
    for _f, lvl, foll, viol in twins:
        if lvl not in keys_seen:
            keys_seen.append(lvl)
        by[lvl]["f"].append(int(is_refusal(generate_plain(model, foll, max_tokens))))
        by[lvl]["v"].append(int(is_refusal(generate_plain(model, viol, max_tokens))))
    lo, hi = boundary
    curve, oband, bband, half_gaps = {}, [], [], []
    for lvl in keys_seen:
        v = by[lvl]["v"]; f = by[lvl]["f"]
        fr, vr = float(np.mean(f)), float(np.mean(v))
        curve[str(lvl)] = {"following_refusal_rate": round(fr, 3), "violating_refusal_rate": round(vr, 3),
                           "separation": round(vr - fr, 3), "n": len(v)}
        follows_comply = (1.0 - fr) >= foll_comply_thresh
        if vr >= viol_thresh and follows_comply:
            oband.append(lvl)
        if lo <= vr <= hi and follows_comply:
            bband.append(lvl)
            h = len(v) // 2
            if h >= 1:
                half_gaps.append(abs(np.mean(v[:h]) - np.mean(v[h:])))
    sep = max((c["separation"] for c in curve.values()), default=0.0)
    return {"curve": curve, "operating_band": oband, "boundary_band": bband,
            "separation_max": round(sep, 3),
            "selection_regression_halfgap": round(float(max(half_gaps)), 3) if half_gaps else None}


def head_mean_ablation_refusal_rate(model, prompts, layer, head, is_refusal, ref_prompts=None,
                                    max_tokens: int = 48) -> dict:
    """Mean-ablate a single attention head (replace its o_proj input slice z_h with the reference-set
    mean -> the head writes its mean contribution, on-distribution, with no variation; NOT zeroing)
    and measure the refusal rate of the completions on `prompts`. For the L15 H15 anti-refusal
    discriminator: on benign (norm-following) prompts, ablating it should INCREASE over-refusal."""
    import torch
    cfg = model.model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    sl = slice(head * head_dim, (head + 1) * head_dim)
    tok = model.tokenizer
    oproj = model._get_layer_module(layer).self_attn.o_proj
    ref = ref_prompts if ref_prompts is not None else prompts

    zs = []                                       # decision-token z_h over the reference set
    def cap(m, inp):
        zs.append(inp[0][0, -1, sl].detach().float())
    hc = oproj.register_forward_pre_hook(cap)
    try:
        with torch.no_grad():
            for p in ref:
                model.model(**tok(_templated(model, p), return_tensors="pt").to(model._device))
    finally:
        hc.remove()
    mean_z = torch.stack(zs).mean(0).to(model._dtype)

    def ablate(m, inp):
        z = inp[0]
        z[0, :, sl] = mean_z
        return (z,) + inp[1:]

    def rate(hooks):
        n = 0
        for p in prompts:
            ids = tok(_templated(model, p), return_tensors="pt").to(model._device)
            hs = [oproj.register_forward_pre_hook(ablate)] if hooks else []
            try:
                with torch.no_grad():
                    out = model.model.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
            finally:
                for h in hs:
                    h.remove()
            n += int(is_refusal(tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)))
        return n / max(1, len(prompts))

    base, abl = rate(False), rate(True)
    return {"layer": layer, "head": head, "n_prompts": len(prompts),
            "baseline_refusal_rate": round(base, 4), "ablated_refusal_rate": round(abl, 4),
            "over_refusal_increase": round(abl - base, 4),
            "anti_over_refusal_head": bool(abl - base > 0)}
