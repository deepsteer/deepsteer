#!/usr/bin/env python3
"""Phase 1 instruct-model pass: refusal direction + consolidation + geometry.

Extracts the Arditi/Heretic refusal direction (last-input-token diff-of-means,
``chat`` format) per layer on an instruct model, then computes:

  * **refusal-morality geometry** at the headline layer: cosine to each MFT
    foundation + fraction of the refusal direction lying in the 6-foundation
    moral subspace (reuses ``heretic_ablation.subspace_projection_fraction``).
  * **refusal consolidation** (the Phase-1 reduced sub-check): is the single
    ablatable direction the whole story, or is refusal diffuse? Reported as
      - margin d' = (mean proj_harmful - mean proj_harmless) / pooled-std, at the
        headline layer and as a band mean (how cleanly ONE direction separates),
      - single-direction AUC (projection-only classifier),
      - across-band effective dimensionality of the per-layer refusal directions
        (collinear across depth = consolidated; spread = diffuse), via 90%-var
        eff-dim and the participation ratio.

This pass is direction-only: it does NOT orthogonalize or save the model (that is
Phase 2's ``heretic_ablation.py --save-model``). Reuses Paper 5 tooling verbatim
so directions stay cosine-comparable; layer/band come from the Paper 6 registry.

Usage (driven by run_phase1; layer/band injected per model):
    python papers/6_cross_model/scripts/extract_refusal.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --prompts papers/5_moral_alignment/refusal_prompts.json \
        --moral-npz papers/6_cross_model/outputs/qwen25_base/exp1_probe_directions.npz \
        --layer 14 --band 13 27 \
        --output-dir papers/6_cross_model/outputs/qwen25_instruct
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_P5 = Path(__file__).resolve().parent.parent.parent / "5_moral_alignment" / "scripts"
sys.path.insert(0, str(_P5))

import direction_utils as du  # noqa: E402
from heretic_ablation import subspace_projection_fraction  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-math consolidation metrics (model-free; unit-tested on synthetic data)
# ---------------------------------------------------------------------------


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def separation_margin(proj_pos: np.ndarray, proj_neg: np.ndarray) -> float:
    """Cohen's d' separation of two 1-D projection sets (pooled-std normalised)."""
    vp, vn = float(np.var(proj_pos)), float(np.var(proj_neg))
    pooled = np.sqrt(0.5 * (vp + vn)) + 1e-12
    return float((np.mean(proj_pos) - np.mean(proj_neg)) / pooled)


def single_direction_auc(proj_pos: np.ndarray, proj_neg: np.ndarray) -> float:
    """Orientation-free AUC of a one-direction (projection) classifier."""
    scores = np.concatenate([proj_pos, proj_neg])
    y = np.concatenate([np.ones(len(proj_pos)), np.zeros(len(proj_neg))])
    auc = du.roc_auc(scores, y)
    return max(auc, 1.0 - auc)


def participation_ratio(direction_list: list[np.ndarray]) -> float:
    """Effective rank (Σσ²)² / Σσ⁴ of a set of (unit) direction vectors.

    UNCENTERED on purpose: the question is whether the per-layer refusal
    directions point the SAME way across depth (consolidated). Centering would
    subtract their common direction and leave only the residual noise, inflating
    the rank of a collinear set to full rank (the RepE-centering pitfall). On the
    raw vectors, all-collinear -> 1.0 (one direction does the job at every
    layer); independent directions -> ~min(n, d).
    """
    mat = np.stack(direction_list)
    s = np.linalg.svd(mat, compute_uv=False)
    s2 = s ** 2
    denom = float(np.sum(s2 ** 2)) + 1e-24
    return float((np.sum(s2) ** 2) / denom)


def effective_rank_uncentered(direction_list: list[np.ndarray], threshold: float = 0.9) -> int:
    """Number of (uncentered) singular directions holding >= ``threshold`` energy.

    Uncentered for the same reason as :func:`participation_ratio`: a consolidated
    (collinear-across-depth) refusal set should report rank ~1, which centering
    destroys.
    """
    mat = np.stack(direction_list)
    s = np.linalg.svd(mat, compute_uv=False)
    energy = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-24)
    return int(np.searchsorted(energy, threshold)) + 1


def consolidation_at_layer(
    H: np.ndarray, S: np.ndarray, r: np.ndarray
) -> dict:
    """Margin + single-direction AUC for harmful/harmless acts along ``r``."""
    r = _unit(r)
    pH, pS = H @ r, S @ r
    return {
        "margin_dprime": round(separation_margin(pH, pS), 4),
        "single_dir_auc": round(single_direction_auc(pH, pS), 4),
        "n_harmful": int(len(H)),
        "n_harmless": int(len(S)),
    }


# ---------------------------------------------------------------------------
# Per-prompt activation collection (last input token, like the refusal dir)
# ---------------------------------------------------------------------------


def last_token_acts(model, prompts, input_format, layers) -> dict[int, np.ndarray]:
    """Per-prompt last-input-token residual activations, ``{layer: (n, d)}``.

    Same position as ``heretic_ablation.last_token_means`` (the first generated
    token's source position), but kept per-prompt for the consolidation margin.
    """
    rows: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    chat = input_format == "chat" and getattr(model.tokenizer, "chat_template", None)
    for p in prompts:
        if chat:
            text = model.tokenizer.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = p
        acts = model.get_activations(text, layers=layers)
        for L in layers:
            rows[L].append(acts[L][0, -1, :].float().numpy())
    return {L: np.stack(rows[L]) for L in layers}


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: refusal direction + consolidation")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--prompts", required=True, help='JSON {"harmful":[...],"harmless":[...]}.')
    ap.add_argument("--moral-npz", default=None, help="Base exp1_probe_directions.npz (geometry).")
    ap.add_argument("--moral-kind", default="probe", help="Foundation key kind in moral npz.")
    ap.add_argument("--layer", type=int, required=True, help="Headline layer (depth ~0.5).")
    ap.add_argument("--band", type=int, nargs=2, required=True, help="Stable band (inclusive).")
    ap.add_argument("--input-format", choices=["raw", "chat"], default="chat")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-prompts", type=int, default=None, help="Cap per class (VALIDATE smoke).")
    ap.add_argument("--only-headline-layer", action="store_true",
                    help="Restrict collection to the headline layer (cheap smoke).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ps = json.load(open(args.prompts))
    harmful, harmless = ps["harmful"], ps["harmless"]
    if args.max_prompts:
        harmful, harmless = harmful[: args.max_prompts], harmless[: args.max_prompts]

    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device,
                          access_tier=AccessTier.WEIGHTS, revision=args.revision)
    n_layers = model.info.n_layers
    band = list(range(args.band[0], args.band[1] + 1))
    layers = [args.layer] if args.only_headline_layer else list(range(n_layers))
    print(f"Loaded {args.model} ({n_layers}L) in {time.time()-t0:.1f}s; "
          f"headline L{args.layer}, band {args.band}; "
          f"{len(harmful)} harmful / {len(harmless)} harmless; fmt={args.input_format}")

    # ---- per-prompt last-token activations -> refusal direction per layer ----
    H = last_token_acts(model, harmful, args.input_format, layers)
    S = last_token_acts(model, harmless, args.input_format, layers)
    refusal_by_layer = {L: _unit(H[L].mean(0) - S[L].mean(0)) for L in layers}
    du.save_directions(out / "refusal_directions.npz", {"refusal": refusal_by_layer})

    # ---- consolidation: margin + single-dir AUC per layer; eff-dim across band ----
    per_layer = {L: consolidation_at_layer(H[L], S[L], refusal_by_layer[L]) for L in layers}
    band_present = [L for L in band if L in refusal_by_layer]
    band_dirs = [refusal_by_layer[L] for L in band_present]
    across_band = {}
    if len(band_dirs) >= 2:
        across_band = {
            "n_layers_in_band": len(band_dirs),
            "effrank_90pct_uncentered": effective_rank_uncentered(band_dirs, 0.9),
            "participation_ratio": round(participation_ratio(band_dirs), 4),
        }

    def band_mean(key: str) -> float:
        vals = [per_layer[L][key] for L in band_present if L in per_layer]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    consolidation = {
        "headline_layer": args.layer,
        "headline": per_layer.get(args.layer),
        "band": list(args.band),
        "band_mean_margin_dprime": band_mean("margin_dprime"),
        "band_mean_single_dir_auc": band_mean("single_dir_auc"),
        "across_band_refusal_directions": across_band,
        "per_layer": {str(L): per_layer[L] for L in sorted(per_layer)},
    }

    # ---- refusal-morality geometry at the headline layer (reuse heretic 3.4) ----
    geom = None
    if args.moral_npz:
        moral = du.load_directions(args.moral_npz)
        foundations = [f for f in FOUNDATION_ORDER if f in moral]
        L = args.layer
        cos_at = {
            FOUNDATION_SHORT[f]: round(du.cosine(refusal_by_layer[L], moral[f][L]), 4)
            for f in foundations if L in moral[f] and L in refusal_by_layer
        }
        basis = [moral[f][L] for f in foundations if L in moral[f]]
        frac = subspace_projection_fraction(refusal_by_layer[L], basis) if basis else None
        geom = {
            "refusal_layer": L,
            "cosine_to_foundations": cos_at,
            "moral_subspace_projection_fraction": round(frac, 4) if frac is not None else None,
            "mean_abs_cosine": round(float(np.mean([abs(v) for v in cos_at.values()])), 4)
            if cos_at else None,
        }

    payload = {
        "analysis": "phase1_refusal_extraction",
        "model": args.model,
        "revision": args.revision,
        "n_layers": n_layers,
        "input_format": args.input_format,
        "n_harmful": len(harmful),
        "n_harmless": len(harmless),
        "refusal_morality_geometry": geom,
        "consolidation": consolidation,
    }
    with open(out / "refusal_extraction.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote {out}/refusal_directions.npz + refusal_extraction.json")
    if geom:
        frac = geom["moral_subspace_projection_fraction"]
        print(f"  geometry @L{args.layer}: frac={frac}, mean|cos|={geom['mean_abs_cosine']}")
    hd = consolidation["headline"]
    if hd:
        print(f"  consolidation @L{args.layer}: d'={hd['margin_dprime']}, "
              f"single-dir AUC={hd['single_dir_auc']}")
    if across_band:
        ab = across_band
        print(f"  across-band refusal dirs: eff-rank(90%)={ab['effrank_90pct_uncentered']}, "
              f"PR={ab['participation_ratio']} (of {ab['n_layers_in_band']} layers)")

    model.release()


if __name__ == "__main__":
    main()
