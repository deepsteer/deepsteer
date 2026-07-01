#!/usr/bin/env python3
"""Phase 1 core: two-site (EOP vs CoT) refusal extraction for one reasoning model.

Extracts the Arditi/Heretic refusal direction at THREE pools per prompt, then
decomposes each against the per-model moral + persona subspace (reused Paper 5/6
math, so directions stay cosine-comparable):

  * **EOP** — last INPUT token (the reflexive boundary; the same position Paper 6
    used, captured from the full forward at index ``plen-1``).
  * **CoT-last** — last reasoning-trace token (just before ``</think>`` / the
    harmony final channel). MATCHED pooling to EOP (both last-token), so the
    EOP<->CoT cosine and moral-projection asymmetry are like-for-like, not a
    pooling artifact.
  * **CoT-mean** — mean over the reasoning-trace tokens (trace-level moral
    content; the setup for Phase 2 decision-sentence work). Carries a
    coherence/length control: incoherent rollouts are dropped and trace-length
    stats are reported so a trace-level moral signal is not a length artifact.

The **last-vs-mean divergence** (cosine + moral-fraction gap) is reported per
model as a substantive finding about how moral content distributes through the
trace; the interesting case is DISAGREEMENT, not agreement (see PAPER_PLAN).

GPT-OSS-20B is loaded bf16-dequantized (Mxfp4Config) for a clean read, exactly as
the Phase 0d gate settled.

Usage (driven by run_phase1; layer/band/subspace injected):
    python papers/7_reasoning/scripts/extract_two_site.py --key ds_r1_llama8b \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --moral-npz .../ds_r1_llama8b/exp1_probe_directions.npz \
        --persona-npz .../ds_r1_llama8b/persona_directions.npz \
        --layer 16 --band 15 31 --n 96 --output-dir .../ds_r1_llama8b
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

_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
_P6 = Path(__file__).resolve().parent.parent.parent / "6_cross_model" / "scripts"
# Paper 5/6 first for the reused tooling, then THIS paper's scripts at index 0 so
# ``import model_registry`` / ``import think_io`` resolve to Paper 7's, not Paper 6's
# same-named module (the reused Paper 5/6 modules don't import model_registry).
sys.path.insert(0, str(_P5))
sys.path.insert(0, str(_P6))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import model_registry as reg  # noqa: E402
import think_io  # noqa: E402
from deepsteer.directions import extraction as du  # noqa: E402
from extract_refusal import consolidation_at_layer  # noqa: E402  (Paper 6, reused)
from measure_refusal_decomposition import decompose_layer  # noqa: E402  (Paper 5)
from moral_dependency import build_subspace_basis  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER  # noqa: E402

logger = logging.getLogger(__name__)


_unit = du.unit_vector  # shared: deepsteer.directions.extraction.unit_vector


@torch.no_grad()
def _acts_from_ids(model, ids: torch.Tensor, layers: list[int]) -> dict[int, np.ndarray]:
    """Per-position residual activations ``{layer: (seq, hidden)}`` for token ids.

    Runs the model on the explicit id sequence (NOT re-tokenized text) so token
    positions align exactly with the boundary index computed in id space.
    """
    acts: dict[int, np.ndarray] = {}
    hooks = []

    def mk(L):
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            acts[L] = t[0].float().cpu().numpy()  # (seq, hidden)
        return hook

    for L in layers:
        hooks.append(model._get_layer_module(L).register_forward_hook(mk(L)))
    try:
        model.model(ids.to(model.model.device))
    finally:
        for h in hooks:
            h.remove()
    return acts


def collect_sites(model, prompts, cot_format, layers, max_new_tokens):
    """Per-prompt EOP / CoT-last / CoT-mean vectors per layer, plus coherence meta.

    Returns ``(site_rows, meta)`` where ``site_rows[site][L]`` is a list of
    per-prompt vectors (EOP over ALL prompts; CoT sites over COHERENT prompts
    only) and ``meta`` holds per-prompt coherence / trace-length / reached-final.
    """
    sites = {s: {L: [] for L in layers} for s in ("eop", "cot_last", "cot_mean")}
    meta = {"coherent": [], "reached_final": [], "trace_len": [], "n_prompts": len(prompts)}
    tok = model.tokenizer
    for p in prompts:
        prompt = think_io.think_prompt(tok, p)
        enc = tok(prompt, return_tensors="pt")
        plen = enc["input_ids"].shape[1]
        gen = model.model.generate(enc["input_ids"].to(model.model.device),
                                   attention_mask=enc["attention_mask"].to(model.model.device),
                                   max_new_tokens=max_new_tokens, do_sample=False)
        full_ids = gen[0].detach().cpu()
        generated = full_ids[plen:]
        boundary = think_io.cot_token_boundary(tok, generated, cot_format)  # in generated space
        reached_final = boundary < len(generated)
        rollout = think_io.decode_rollout(tok, generated, cot_format)
        reasoning, _final = think_io.split_rollout(rollout, cot_format)
        coherent = boundary >= 1 and not think_io.looks_degenerate(reasoning or rollout)
        meta["coherent"].append(bool(coherent))
        meta["reached_final"].append(bool(reached_final))
        meta["trace_len"].append(int(boundary))

        acts = _acts_from_ids(model, full_ids.unsqueeze(0), layers)
        cot_lo, cot_hi = plen, plen + boundary  # reasoning-trace positions in full
        for L in layers:
            a = acts[L]
            sites["eop"][L].append(a[plen - 1])  # last INPUT token
            if coherent:
                sites["cot_last"][L].append(a[cot_hi - 1])           # last reasoning token
                sites["cot_mean"][L].append(a[cot_lo:cot_hi].mean(0))  # mean over trace
    return sites, meta


def _direction(H: np.ndarray, S: np.ndarray) -> np.ndarray:
    return _unit(H.mean(0) - S.mean(0))


def _decompose_dir(r, H, S, basis_L, persona_L, foundations, found_vecs_L, gap_seed):
    """Energy decomposition + consolidation for one direction at one layer."""
    out = decompose_layer(r, basis_L, persona_L, foundations, found_vecs_L)
    out.update(consolidation_at_layer(H, S, r, gap_seed=gap_seed))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1 two-site refusal extraction (one model)")
    ap.add_argument("--key", required=True, help="registry key: " + ",".join(reg.PANEL_ORDER))
    ap.add_argument("--prompts", required=True, help='JSON {"harmful":[...],"harmless":[...]}.')
    ap.add_argument("--moral-npz", required=True)
    ap.add_argument("--persona-npz", required=True)
    ap.add_argument("--moral-kind", default="probe")
    ap.add_argument("--layer", type=int, required=True, help="headline layer (depth 0.5)")
    ap.add_argument("--band", type=int, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=96, help="prompts per class (generation-bounded).")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    spec = reg.get(args.key)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    headline, (lo, hi) = args.layer, args.band
    layers = list(range(lo, hi + 1))
    if headline not in layers:
        layers = sorted(set(layers) | {headline})

    ps = json.load(open(args.prompts))
    harmful, harmless = ps["harmful"][: args.n], ps["harmless"][: args.n]

    # ---- load (WhiteBoxModel auto-dequantizes the MoE primary's mxfp4 experts) --
    t0 = time.time()
    model = WhiteBoxModel(spec.reasoning_repo, device=args.device,
                          torch_dtype=torch.bfloat16, access_tier=AccessTier.WEIGHTS)
    cfg = model.model.config
    spec.assert_matches_model(model.info.n_layers, getattr(cfg, "hidden_size", None),
                              model_type_live=getattr(cfg, "model_type", None))
    print(f"Loaded {spec.reasoning_repo} ({model.info.n_layers}L) in {time.time()-t0:.0f}s; "
          f"headline L{headline}, band [{lo},{hi}]; cot={spec.cot_format.value}; "
          f"{len(harmful)} harmful / {len(harmless)} harmless @ {args.max_new_tokens} tok")

    # ---- moral subspace basis + persona + raw foundation vecs ------------------
    moral = du.load_directions(args.moral_npz)
    basis_by_layer, rank_by_layer, _names = build_subspace_basis(
        moral, kind=args.moral_kind, n_layers=model.info.n_layers)
    foundations = [f for f in FOUNDATION_ORDER if f in moral]
    persona_dirs = du.load_directions(args.persona_npz)
    persona = persona_dirs.get("persona", persona_dirs.get("persona_meandiff", {}))

    # ---- collect the three pools for harmful + harmless ------------------------
    print(">> collecting harmful rollouts ...")
    Hs, Hmeta = collect_sites(model, harmful, spec.cot_format, layers, args.max_new_tokens)
    print(f"   harmful coherent {sum(Hmeta['coherent'])}/{len(harmful)}, "
          f"reached_final {sum(Hmeta['reached_final'])}, "
          f"median trace_len {int(np.median(Hmeta['trace_len']))}")
    print(">> collecting harmless rollouts ...")
    Ss, Smeta = collect_sites(model, harmless, spec.cot_format, layers, args.max_new_tokens)
    print(f"   harmless coherent {sum(Smeta['coherent'])}/{len(harmless)}, "
          f"reached_final {sum(Smeta['reached_final'])}, "
          f"median trace_len {int(np.median(Smeta['trace_len']))}")

    # ---- per-site refusal directions per layer ---------------------------------
    site_dirs: dict[str, dict[int, np.ndarray]] = {}
    site_HS: dict[str, dict[int, tuple]] = {}
    for site in ("eop", "cot_last", "cot_mean"):
        site_dirs[site], site_HS[site] = {}, {}
        for L in layers:
            H = np.stack(Hs[site][L]) if Hs[site][L] else None
            S = np.stack(Ss[site][L]) if Ss[site][L] else None
            if H is None or S is None or len(H) < 2 or len(S) < 2:
                continue
            site_dirs[site][L] = _direction(H, S)
            site_HS[site][L] = (H, S)
    du.save_directions(out / "two_site_refusal_directions.npz", site_dirs)

    # Per-prompt HEADLINE-layer vectors, so bootstrap CIs run later LOCALLY (no
    # GPU): resample these rows, recompute the mean-diff directions + fractions.
    hv = {"headline": np.int64(headline)}
    for site in ("eop", "cot_last", "cot_mean"):
        if headline in site_HS[site]:
            H, S = site_HS[site][headline]
            hv[f"{site}_H"], hv[f"{site}_S"] = H.astype(np.float32), S.astype(np.float32)
    np.savez(out / "two_site_headline_vectors.npz", **hv)

    # ---- decomposition at headline + band per site -----------------------------
    def decomp_site(site: str) -> dict:
        per_layer = {}
        for L in layers:
            if L not in site_dirs[site] or L not in basis_by_layer or L not in persona:
                continue
            found_vecs_L = {f: moral[f][L] for f in foundations if L in moral[f]}
            H, S = site_HS[site][L]
            per_layer[L] = _decompose_dir(site_dirs[site][L], H, S, basis_by_layer[L],
                                          persona[L], foundations, found_vecs_L, gap_seed=42)
        band_layers = [L for L in range(lo, hi + 1) if L in per_layer]

        def bmean(k):
            vals = [per_layer[L][k] for L in band_layers if k in per_layer[L]]
            return round(float(np.mean(vals)), 6) if vals else None

        return {
            "headline_layer": headline,
            "headline": per_layer.get(headline),
            "band_mean": {k: bmean(k) for k in
                          ("mft_frac", "persona_unique_frac", "residual_frac", "auc_gap",
                           "margin_dprime", "single_dir_auc")},
            "per_layer": {str(L): per_layer[L] for L in sorted(per_layer)},
        }

    decomp = {s: decomp_site(s) for s in ("eop", "cot_last", "cot_mean")}

    # ---- cross-site comparisons (the headline numbers) -------------------------
    def cos_layer(a, b, L):
        if L in site_dirs[a] and L in site_dirs[b]:
            return round(du.cosine(site_dirs[a][L], site_dirs[b][L]), 4)
        return None

    def frac(site, L):
        h = decomp[site]["per_layer"].get(str(L))
        return h["mft_frac"] if h else None

    band_layers = [L for L in range(lo, hi + 1)]
    comparisons = {
        # matched-pooling EOP<->CoT (both last-token): cosine + moral asymmetry
        "eop_vs_cotlast_cosine_headline": cos_layer("eop", "cot_last", headline),
        "eop_vs_cotlast_cosine_band_mean": _band_cos(site_dirs, "eop", "cot_last", band_layers),
        "moral_asymmetry_headline": _diff(frac("cot_last", headline), frac("eop", headline)),
        "moral_asymmetry_band_mean": _band_frac_diff(decomp, "cot_last", "eop", band_layers),
        # last-vs-mean divergence: how moral content distributes through the trace
        "cotlast_vs_cotmean_cosine_headline": cos_layer("cot_last", "cot_mean", headline),
        "cotlast_vs_cotmean_cosine_band_mean": _band_cos(site_dirs, "cot_last", "cot_mean", band_layers),
        "trace_distribution_headline": _diff(frac("cot_mean", headline), frac("cot_last", headline)),
        "trace_distribution_band_mean": _band_frac_diff(decomp, "cot_mean", "cot_last", band_layers),
    }

    payload = {
        "analysis": "phase1_two_site", "key": spec.key, "model": spec.reasoning_repo,
        "provenance": spec.provenance.value, "cot_format": spec.cot_format.value,
        "n_layers": model.info.n_layers, "headline_layer": headline, "band": [lo, hi],
        "n_harmful": len(harmful), "n_harmless": len(harmless),
        "max_new_tokens": args.max_new_tokens,
        # coherence / length control for the mean-over-CoT pool
        "coherence": {
            "harmful_coherent": int(sum(Hmeta["coherent"])),
            "harmless_coherent": int(sum(Smeta["coherent"])),
            "harmful_reached_final": int(sum(Hmeta["reached_final"])),
            "harmless_reached_final": int(sum(Smeta["reached_final"])),
            "harmful_trace_len_median": int(np.median(Hmeta["trace_len"])),
            "harmless_trace_len_median": int(np.median(Smeta["trace_len"])),
            "harmful_trace_len_mean": round(float(np.mean(Hmeta["trace_len"])), 1),
            "harmless_trace_len_mean": round(float(np.mean(Smeta["trace_len"])), 1),
        },
        "decomposition": decomp,
        "comparisons": comparisons,
    }
    (out / "two_site_decomposition.json").write_text(json.dumps(payload, indent=2))
    model.release()

    print(f"\nWrote {out}/two_site_decomposition.json")
    print(f"  EOP      mft_frac (band): {decomp['eop']['band_mean']['mft_frac']}, "
          f"residual {decomp['eop']['band_mean']['residual_frac']}, gap {decomp['eop']['band_mean']['auc_gap']}")
    print(f"  CoT-last mft_frac (band): {decomp['cot_last']['band_mean']['mft_frac']}, "
          f"residual {decomp['cot_last']['band_mean']['residual_frac']}, gap {decomp['cot_last']['band_mean']['auc_gap']}")
    print(f"  CoT-mean mft_frac (band): {decomp['cot_mean']['band_mean']['mft_frac']}, "
          f"residual {decomp['cot_mean']['band_mean']['residual_frac']}")
    print(f"  EOP<->CoT-last cosine (band): {comparisons['eop_vs_cotlast_cosine_band_mean']}; "
          f"moral asymmetry (band): {comparisons['moral_asymmetry_band_mean']}")
    print(f"  CoT last-vs-mean cosine (band): {comparisons['cotlast_vs_cotmean_cosine_band_mean']}; "
          f"trace-distribution gap (band): {comparisons['trace_distribution_band_mean']}")


def _diff(a, b):
    return round(a - b, 6) if (a is not None and b is not None) else None


def _band_cos(site_dirs, a, b, band_layers):
    vals = [du.cosine(site_dirs[a][L], site_dirs[b][L])
            for L in band_layers if L in site_dirs.get(a, {}) and L in site_dirs.get(b, {})]
    return round(float(np.mean(vals)), 4) if vals else None


def _band_frac_diff(decomp, a, b, band_layers):
    diffs = []
    for L in band_layers:
        ha = decomp[a]["per_layer"].get(str(L))
        hb = decomp[b]["per_layer"].get(str(L))
        if ha and hb:
            diffs.append(ha["mft_frac"] - hb["mft_frac"])
    return round(float(np.mean(diffs)), 6) if diffs else None


if __name__ == "__main__":
    main()
