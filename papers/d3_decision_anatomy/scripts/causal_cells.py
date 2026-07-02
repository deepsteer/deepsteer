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


def interchange(model, src, tgt, src_pos, tgt_pos, layer, r_hat, read_layer=None,
                restrict_Q=None):
    """Patch src's content into tgt at `layer`; return the decision-token projection onto r_hat read
    at `read_layer` (default = layer). Length-agnostic: the src flipped span is MEAN-POOLED to one
    content vector and broadcast across tgt's flipped-span positions (an aggregate-content swap, so
    differently-tokenized moral-intent spans are comparable). FULL replaces the tgt content vector;
    RESTRICTED (restrict_Q hidden,k) swaps only the in-subspace component (tgt keeps its off-subspace
    part). tgt positions guarded against the current seq length (safe under KV-cached decoding)."""
    import torch
    read_layer = layer if read_layer is None else read_layer
    src_resid = _resid_at(model, src, layer, src_pos).mean(0, keepdim=True)   # (1, hidden) content mean
    Q = None if restrict_Q is None else torch.tensor(restrict_Q, dtype=src_resid.dtype,
                                                     device=src_resid.device)
    if Q is not None:
        src_resid = (src_resid @ Q) @ Q.T                       # src's in-subspace component only
    r = torch.tensor(_unit(r_hat), dtype=torch.float32, device=model._device)
    proj = {}

    def patch_hook(module, inp):
        x = inp[0]
        pos = [p for p in tgt_pos if p < x.shape[1]]
        if pos:
            cur = x[0, pos, :]
            x[0, pos, :] = (cur - (cur @ Q) @ Q.T + src_resid) if Q is not None \
                else src_resid.expand(len(pos), -1)
        return (x,) + inp[1:]

    def read_hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        proj["p"] = float(t[0, -1, :].float() @ r)

    tok = model.tokenizer
    h1 = model._get_layer_module(layer).register_forward_pre_hook(patch_hook)
    h2 = model._get_layer_module(read_layer).register_forward_hook(read_hook)
    try:
        with torch.no_grad():
            model.model(**tok(_templated(model, tgt), return_tensors="pt").to(model._device))
    finally:
        h1.remove(); h2.remove()
    return proj["p"]


def baseline_proj(model, text: str, read_layer: int, r_hat) -> float:
    """Unpatched decision-token projection onto r_hat (the patch baseline)."""
    import torch
    tok = model.tokenizer
    r = torch.tensor(_unit(r_hat), dtype=torch.float32, device=model._device)
    cap = {}

    def read_hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        cap["p"] = float(t[0, -1, :].float() @ r)
    h = model._get_layer_module(read_layer).register_forward_hook(read_hook)
    try:
        with torch.no_grad():
            model.model(**tok(_templated(model, text), return_tensors="pt").to(model._device))
    finally:
        h.remove()
    return cap["p"]
