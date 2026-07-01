#!/usr/bin/env python3
"""Phase 2c: diff-of-means at t_inst (harmfulness) AND t_post-inst (refusal).

Zhao et al. (arXiv:2507.11878): harmfulness is encoded at t_inst (last instruction
token), refusal at t_post-inst (last templated token). Our prior extractor read
only t_post-inst (the weak causal handle). This extracts the diff-of-means
direction at BOTH positions per layer, and produces the clustering data for the
Zhao Fig 2 validity check:
  * at t_inst,    harmful vs harmless should separate (harmfulness clustering);
  * at t_post-inst, refused vs not should separate (refusal clustering);
the clustering FLIPS between the sites (a prompt accepted-but-harmful sits with
the harmful cluster at t_inst, with the accepted cluster at t_post-inst).

Refusal labels come from ``refusal_baseline.json`` if present (so the t_post-inst
refusal clustering uses real behavior); otherwise only the harmful/harmless
clustering at t_inst is reported. Token positions are located by
``token_positions`` (model-agnostic common-suffix method; validated on the real
tokenizers). The model is run on the EXACT templated ids so positions align.

Usage (driven by run_phase2c / remote_phase2c.sh):
    python papers/7_reasoning/scripts/extract_positions.py --key ds_r1_llama8b \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --layer 16 --band 15 31 --n 64 --output-dir .../ds_r1_llama8b
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as reg  # noqa: E402
from deepsteer.directions import extraction as du  # noqa: E402
from deepsteer.reasoning import token_positions as tp  # noqa: E402
from extract_two_site import _acts_from_ids  # noqa: E402


_unit = du.unit_vector  # shared: deepsteer.directions.extraction.unit_vector


def collect_positions(model, prompts, post_count, layers):
    """Per-prompt residuals at t_inst and t_post-inst, ``{site: {layer: (n,d)}}``."""
    tok = model.tokenizer
    sites = {"t_inst": {L: [] for L in layers}, "t_post_inst": {L: [] for L in layers}}
    for p in prompts:
        pos = tp.instruction_positions(tok, p, post_count)
        ids = torch.tensor(pos["ids"]).unsqueeze(0)
        acts = _acts_from_ids(model, ids, layers)  # {L: (seq, hidden)}
        for L in layers:
            sites["t_inst"][L].append(acts[L][pos["t_inst"]])
            sites["t_post_inst"][L].append(acts[L][pos["t_post_inst"]])
    return {s: {L: np.stack(v) for L, v in d.items()} for s, d in sites.items()}


def _separation(proj_pos, proj_neg):
    """Cohen's d' of two 1-D projection sets (how cleanly the label separates)."""
    vp, vn = float(np.var(proj_pos)), float(np.var(proj_neg))
    pooled = np.sqrt(0.5 * (vp + vn)) + 1e-12
    return round(float((np.mean(proj_pos) - np.mean(proj_neg)) / pooled), 4)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2c: t_inst + t_post-inst diff-of-means")
    ap.add_argument("--key", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--layer", type=int, required=True, help="headline layer")
    ap.add_argument("--band", type=int, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=64, help="prompts/class.")
    ap.add_argument("--baseline", default=None, help="refusal_baseline.json for refusal labels.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = args.band
    headline = args.layer
    layers = sorted(set(range(lo, hi + 1)) | {headline})
    ps = json.load(open(args.prompts))
    harmful, harmless = ps["harmful"][: args.n], ps["harmless"][: args.n]

    t0 = time.time()
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    spec.assert_matches_model(model.info.n_layers,
                              getattr(model.model.config, "hidden_size", None),
                              model_type_live=getattr(model.model.config, "model_type", None))
    post_count = tp.post_instruction_token_count(model.tokenizer)
    desc = tp.describe(model.tokenizer, harmful[0], post_count)
    print(f"Loaded {spec.reasoning_repo} in {time.time()-t0:.0f}s; headline L{headline}, band "
          f"[{lo},{hi}]; post_count={post_count}; suffix={desc['post_instruction_suffix']!r}; "
          f"{len(harmful)} harmful / {len(harmless)} harmless")

    H = collect_positions(model, harmful, post_count, layers)
    S = collect_positions(model, harmless, post_count, layers)
    model.release()

    # --- diff-of-means directions at both sites, per layer ----------------------
    harm_dir = {L: _unit(H["t_inst"][L].mean(0) - S["t_inst"][L].mean(0)) for L in layers}
    ref_dir = {L: _unit(H["t_post_inst"][L].mean(0) - S["t_post_inst"][L].mean(0)) for L in layers}
    du.save_directions(out / "position_directions.npz",
                       {"harmfulness_t_inst": harm_dir, "refusal_t_post_inst": ref_dir})
    # headline per-prompt vectors for local reply-inversion / clustering reuse (no GPU later)
    np.savez(out / "position_headline_vectors.npz", headline=np.int64(headline),
             H_t_inst=H["t_inst"][headline].astype(np.float32),
             S_t_inst=S["t_inst"][headline].astype(np.float32),
             H_t_post=H["t_post_inst"][headline].astype(np.float32),
             S_t_post=S["t_post_inst"][headline].astype(np.float32))

    # --- separation: harmfulness@t_inst vs @t_post-inst; cross-site cosine -------
    L = headline
    hdir, rdir = harm_dir[L], ref_dir[L]
    sep = {
        "harmful_vs_harmless_at_t_inst": _separation(H["t_inst"][L] @ hdir, S["t_inst"][L] @ hdir),
        "harmful_vs_harmless_at_t_post": _separation(H["t_post_inst"][L] @ hdir, S["t_post_inst"][L] @ hdir),
        "cos_harmfulness_refusal": round(du.cosine(hdir, rdir), 4),
    }

    # --- refusal clustering at t_post-inst if behavior labels are available ------
    refusal_sep = None
    base = args.baseline or (out / "refusal_baseline.json")
    if Path(base).exists():
        b = json.loads(Path(base).read_text())
        rows = b.get("rows", [])
        # rows align with the first n harmful prompts (baseline uses harmful only)
        refused = [bool(r.get("refused_whole")) for r in rows][: len(harmful)]
        if any(refused) and not all(refused):
            ref_idx = [i for i, r in enumerate(refused) if r]
            acc_idx = [i for i, r in enumerate(refused) if not r]
            ht = H["t_post_inst"][L]
            refusal_sep = {
                "n_refused": len(ref_idx), "n_accepted": len(acc_idx),
                "refused_vs_accepted_at_t_post":
                    _separation(ht[ref_idx] @ rdir, ht[acc_idx] @ rdir),
                "refused_vs_accepted_at_t_inst":
                    _separation(H["t_inst"][L][ref_idx] @ hdir, H["t_inst"][L][acc_idx] @ hdir),
                "note": ("flip check: harmful prompts split by refusal should separate at "
                         "t_post-inst (refusal) but NOT cleanly at t_inst (harmfulness common)"),
            }

    payload = {
        "analysis": "phase2c_positions", "key": spec.key, "model": spec.reasoning_repo,
        "headline_layer": headline, "band": [lo, hi], "post_count": post_count,
        "post_instruction_suffix": desc["post_instruction_suffix"],
        "t_inst_token_example": desc["t_inst_token"],
        "n_harmful": len(harmful), "n_harmless": len(harmless),
        "separation": sep, "refusal_clustering": refusal_sep,
    }
    (out / "position_extraction.json").write_text(json.dumps(payload, indent=2))

    print(f"\nWrote {out}/position_directions.npz + position_extraction.json")
    print(f"  separation @L{L}: harmful/harmless d' at t_inst={sep['harmful_vs_harmless_at_t_inst']} "
          f"vs at t_post={sep['harmful_vs_harmless_at_t_post']}")
    print(f"  cos(harmfulness_t_inst, refusal_t_post)={sep['cos_harmfulness_refusal']}  "
          f"(Zhao: separate concepts -> expect modest)")
    if refusal_sep:
        print(f"  refusal flip: refused/accepted d' at t_post={refusal_sep['refused_vs_accepted_at_t_post']} "
              f"vs at t_inst={refusal_sep['refused_vs_accepted_at_t_inst']} "
              f"({refusal_sep['n_refused']} ref / {refusal_sep['n_accepted']} acc)")
    else:
        print("  refusal clustering: skipped (no mixed refusal labels in baseline)")


if __name__ == "__main__":
    main()
