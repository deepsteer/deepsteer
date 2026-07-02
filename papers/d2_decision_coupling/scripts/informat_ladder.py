#!/usr/bin/env python3
"""In-format ladder (D2 Amendment 1 §7 + addendum): re-derive the full moral-geometry ladder with
V_moral, controls, persona, AND the null's act_samples all extracted in CHAT format at a matched
position class, so R2/G3/R5 are format-consistent instead of chat-direction-vs-raw-V_moral.

Factor-decomposed: ONE forward pass per text yields three position classes (band-layer slices):
  * final_pre_assistant  (PRIMARY, the decision site — apples-to-apples with refusal/judgment)
  * last_content         (secondary)
  * mean_content         (secondary; sink-free: content span excludes BOS/template tokens)

Every rung is in-format at the same position class. Standardized nulls (per-dim z-score from the
CHAT act_sample at that position class) + a top-k projection-out variant for massive-activation
families; the covariance null MUST consume the format/position-matched act_sample (hard-guarded).

Pre-registered branch (Amendment 1 §7): D1's orthogonality is FORMAT-ROBUST iff the refusal gate
sits below the in-format moral-family band by M at the PRIMARY position class; else it is scoped to
the raw narrative register (both publishable). GPU extraction; VALIDATE=1 -> tiny-model smoke, and
the model-free reproduction + invariant checks live in b_session1_local_test.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_P6 = HERE.parents[1] / "6_cross_model" / "scripts"
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(_P6))
sys.path.insert(0, str(HERE))

from deepsteer.directions import extraction as du  # noqa: E402
# Reuse the EXACT ladder math from the standardized recompute so raw reproduction is meaningful.
from standardized_recompute import ortho, frac, std_unit, topk_mask, band  # noqa: E402
from b3_batched_extractions import FABLE_SCHEMA_PAIRS  # noqa: E402

MARGIN_M = 0.05
K = 2000
SEED = 0
SRC = ["moral_stories", "fables", "ethics"]
CONTROLS = ["syntax", "register", "sentiment"]
POS_CLASSES = ["final_pre_assistant", "last_content", "mean_content"]  # PRIMARY first
PRIMARY = "final_pre_assistant"
_unit = du.unit_vector


# --------------------------- position-class extraction ---------------------------

def find_content_span(full_ids: list[int], content_ids: list[int]) -> tuple[int, int]:
    """Locate the contiguous content-token block inside the chat-templated ids. Falls back to the
    final token if the content does not appear verbatim (template re-tokenization at the seam)."""
    n, m = len(full_ids), len(content_ids)
    for start in range(n - m, -1, -1):  # search from the end (content precedes the assistant tail)
        if full_ids[start:start + m] == content_ids:
            return start, start + m
    # degraded fallback: trailing region (secondary classes only; PRIMARY uses the true last token)
    return max(n - 1, 0), n


def chat_position_acts(model, text: str, layers: list[int]) -> dict[str, dict[int, np.ndarray]]:
    """One forward over the chat-templated text -> {position_class: {layer: vector}}."""
    tok = model.tokenizer
    templated = (tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                         add_generation_prompt=True)
                 if getattr(tok, "chat_template", None) else text)
    full_ids = tok(templated, add_special_tokens=False)["input_ids"]
    content_ids = tok(text, add_special_tokens=False)["input_ids"]
    s, e = find_content_span(full_ids, content_ids)
    acts = model.get_activations(templated, layers=layers)
    out: dict[str, dict[int, np.ndarray]] = {p: {} for p in POS_CLASSES}
    for L in layers:
        H = acts[L][0].float().numpy()  # (seq, hidden)
        out["final_pre_assistant"][L] = H[-1]
        out["last_content"][L] = H[e - 1]
        out["mean_content"][L] = H[s:e].mean(0)
    return out


def extract_positions(model, pairs, layers):
    """{position_class: {layer: (unit mean-diff dir, per-pair diffs (n,d))}} for a contrast set."""
    pos_diffs = {p: {L: [] for L in layers} for p in POS_CLASSES}
    for a, b in pairs:
        pa, pb = chat_position_acts(model, a, layers), chat_position_acts(model, b, layers)
        for p in POS_CLASSES:
            for L in layers:
                pos_diffs[p][L].append(pa[p][L] - pb[p][L])
    out = {}
    for p in POS_CLASSES:
        out[p] = {L: (_unit(np.stack(pos_diffs[p][L]).mean(0)), np.stack(pos_diffs[p][L]))
                  for L in layers}
    return out


def pooled_chat_actsample(model, texts, layers) -> dict[str, dict[int, np.ndarray]]:
    """Chat act_samples for the covariance null, per position class: {pos: {layer: (n, d)}}."""
    acc = {p: {L: [] for L in layers} for p in POS_CLASSES}
    for t in texts:
        pa = chat_position_acts(model, t, layers)
        for p in POS_CLASSES:
            for L in layers:
                acc[p][L].append(pa[p][L])
    return {p: {L: np.stack(acc[p][L]) for L in layers} for p in POS_CLASSES}


# --------------------------- ladder (typed, format-guarded) ---------------------------

def _require_matched(tag_actsample: tuple, tag_dirs: tuple) -> None:
    """HARD GUARD (the most likely build bug): the null's act_sample must share (format,
    position_class) with the directions it nulls. Raise loudly on a mismatch."""
    if tag_actsample != tag_dirs:
        raise ValueError(f"format/position mismatch: null act_sample {tag_actsample} != "
                         f"directions {tag_dirs} — the chat null must consume chat act_samples")


def _null_q95_on(Q, Ac, rng, k=K):
    n = Ac.shape[0]
    return float(np.percentile([frac(Q, Ac.T @ rng.standard_normal(n)) for _ in range(k)], 95))


def ladder(sources, controls, persona, act_sample, tag, *, refusal=None, judgment=None,
           standardize=True, rng=None):
    """Compute band + R5 (+ G3/R2 if refusal/judgment given) in the (standardized) space defined by
    act_sample. `tag`=(format, position_class) is asserted to match between dirs and act_sample."""
    rng = rng or np.random.default_rng(SEED)
    _require_matched(act_sample["tag"], tag)   # <-- the invariant
    X = np.asarray(act_sample["X"], np.float64)
    sig = (X.std(0) + 1e-12) if standardize else np.ones(X.shape[1])
    S = {s: std_unit(sources[s], sig) for s in SRC}
    Q = ortho([S[s] for s in SRC])
    Ac = ((X - X.mean(0)) / sig)
    Ac = Ac - Ac.mean(0)
    q95 = _null_q95_on(Q, Ac, rng)
    bnd, hv = band(S)
    c = {ck: round(frac(Q, std_unit(cv, sig)), 4) for ck, cv in controls.items()}
    c["persona"] = round(frac(Q, std_unit(persona, sig)), 4)
    out = {"format": tag[0], "position_class": tag[1], "standardized": standardize,
           "sigma_provenance": act_sample.get("sigma_provenance", "act_sample @ (format,pos)"),
           "null_q95": round(q95, 4), "moral_family_band": bnd, "heldout": hv, "c_controls": c}
    if refusal is not None:
        p_ref = round(frac(Q, std_unit(refusal, sig)), 4)
        floor = min(c["syntax"], c["register"])
        band_min = bnd[0]
        out["G3_refusal_p"] = p_ref
        out["R5_min_c_syntax_register"] = round(floor, 4)
        out["R5_strong_form_holds"] = bool(p_ref <= floor + MARGIN_M)
        # Format-robustness branch (PRIMARY position only): refusal below the IN-FORMAT band by M.
        out["format_robust_refusal_below_band"] = bool(p_ref <= band_min - MARGIN_M)
    if judgment is not None:
        out["R2_judgment_p"] = round(frac(Q, std_unit(judgment, sig)), 4)
    return out


# --------------------------- driver ---------------------------

def control_pairs():
    from deepsteer.datasets import get_register_pairs, get_sentiment_pairs, get_syntax_pairs
    return {"syntax": get_syntax_pairs(), "register": get_register_pairs(),
            "sentiment": get_sentiment_pairs()}


def persona_pairs(n_cap):
    from deepsteer.datasets import get_persona_pairs
    ps = get_persona_pairs()
    return ps[:n_cap] if n_cap else ps


def load_moral_pairs(dataset, fables_json, n_cap):
    ds = json.loads(Path(dataset).read_text())
    ms = [(p["moral"], p["neutral"]) for p in ds["train"] if p.get("source") == "moral_stories"] \
        or [(p["moral"], p["neutral"]) for p in ds["train"]]
    eth = [(p["moral"], p["neutral"]) for p in ds["eval_generalization_probe"]]
    fab = json.loads(Path(fables_json).read_text())
    fab = [(p["moral"], p["neutral"]) for p in fab["pairs"] if p.get("clean")]
    if n_cap:
        ms, eth, fab = ms[:n_cap], eth[:n_cap], fab[:n_cap]
    return {"moral_stories": ms, "fables": fab, "ethics": eth}


def main():
    ap = argparse.ArgumentParser(description="In-format (chat) ladder — Amendment 1 §7.")
    ap.add_argument("--model", default="allenai/Olmo-3-7B-Instruct")
    ap.add_argument("--key", default="olmo3")
    ap.add_argument("--dataset", default=str(HERE.parents[2] / "deepsteer" / "datasets" / "d1_vmoral_v1.json"))
    ap.add_argument("--fables", default=str(HERE.parents[1] / "d1_moral_subspace" / "outputs" / "full" / "fables_train_full.json"))
    ap.add_argument("--refusal-npz")   # chat decision-site refusal (from B1 chunk)
    ap.add_argument("--judgment-npz")  # b1_judgment_dir_<key>.npz
    ap.add_argument("--out", default=str(HERE.parent / "outputs"))
    args = ap.parse_args()

    validate = os.environ.get("VALIDATE") == "1"
    if validate:
        args.model = "allenai/OLMo-2-0425-1B"
    import model_registry as reg  # noqa: E402
    spec = reg.get(args.key)
    n_cap = 12 if validate else None

    model = du.load_whitebox(args.model)
    same = model.info.n_layers == spec.n_layers
    layer = spec.primary_layer if same else model.info.n_layers // 2
    band_layers = (list(range(spec.band[0], spec.band[1] + 1)) if same else [layer])
    layers = sorted(set([layer, *band_layers]))
    standardize = args.key != "olmo3"  # OLMo well-conditioned; standardize the outlier families

    moral = load_moral_pairs(args.dataset, args.fables, n_cap)
    ctrl = {k: (v[:12] if validate else v) for k, v in control_pairs().items()}

    src_pos = {s: extract_positions(model, moral[s], layers) for s in SRC}
    ctl_pos = {c: extract_positions(model, ctrl[c], layers) for c in CONTROLS}
    per_pos = extract_positions(model, persona_pairs(12 if validate else None), layers)
    # null act_samples: pooled chat activations over a mix of moral + control texts, per pos class.
    sample_texts = [t for s in SRC for pair in moral[s][:40] for t in pair]
    act_pos = pooled_chat_actsample(model, sample_texts, layers)

    refusal = _unit(np.load(args.refusal_npz)["refusal"].astype(np.float64)) if args.refusal_npz and Path(args.refusal_npz).exists() else None
    judgment = _unit(np.load(args.judgment_npz)["judgment_dir"].astype(np.float64)) if args.judgment_npz and Path(args.judgment_npz).exists() else None

    result = {"model": args.model, "key": args.key, "layer": layer, "primary": PRIMARY,
              "standardized": standardize, "positions": {}}
    for p in POS_CLASSES:
        tag = ("chat", p)
        acts = {"X": act_pos[p][layer], "tag": tag, "sigma_provenance": f"chat act_sample @ {p}"}
        sources = {s: src_pos[s][p][layer][0] for s in SRC}
        controls = {c: ctl_pos[c][p][layer][0] for c in CONTROLS}
        persona = per_pos[p][layer][0]
        is_primary = p == PRIMARY
        result["positions"][p] = ladder(
            sources, controls, persona, acts, tag,
            refusal=refusal if is_primary else None,
            judgment=judgment if is_primary else None,
            standardize=standardize, rng=np.random.default_rng(SEED))

    # Save chat V_moral basis + chat act_sample at PRIMARY for B5 injection (Amendment 1 §7 rider 3).
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    prim_sources = [_unit(src_pos[s][PRIMARY][layer][0]) for s in SRC]
    np.savez(out / f"chat_vmoral_{args.key}.npz",
             basis=ortho(prim_sources).T, act_sample=act_pos[PRIMARY][layer],
             layer=layer, position_class=PRIMARY, standardized=standardize)
    (out / f"informat_ladder_{args.key}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
