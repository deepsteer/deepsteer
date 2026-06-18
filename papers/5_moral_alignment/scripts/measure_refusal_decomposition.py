#!/usr/bin/env python3
"""Measurement 1: decompose the Instruct refusal direction onto the moral
subspace, the persona direction, and the residual.

This is a pure-analysis pass over cached directions (no model loading, no new
compute). It answers the single question that picks the pre-training coupling
target (Paper 5 §6.4 -> upstream intervention):

  * refusal mostly **persona-carried**  -> target persona<->MFT coupling.
  * refusal mostly **residual**         -> coupling must create overlap from
                                           scratch (hardest).
  * refusal partly **MFT already**      -> amplify existing structure (easiest).

Energy (squared-norm) decomposition, per layer, for the unit refusal direction
``r``:

  1. ``mft_frac``           = ||proj_S(r)||^2, S = orthonormal span of the 6
                              foundation directions (same basis the dependency
                              metric ablates, ``build_subspace_basis``).
  2. ``persona_unique_frac`` = (r . u)^2, u = unit(persona - proj_S(persona)),
                              i.e. the part of the persona direction orthogonal
                              to the moral subspace. Orthogonal to S by
                              construction, so it adds cleanly.
  3. ``residual_frac``      = 1 - mft_frac - persona_unique_frac.

Because S _|_ u, the three fractions partition the unit refusal direction and
sum to 1.0 exactly. ``sqrt(mft_frac)`` is the norm-ratio reported by Paper 5
§4.4 (``heretic_ablation.subspace_projection_fraction``); we cross-check it
against the cached 0.1044 at the ablation layer.

Conventions (must match the cached artifacts, confirmed from the arrays):
  * all directions are unit-norm float32, 4096-dim, layers 0..31;
  * MFT directions are the **base** probe-weight directions
    (``olmo3_base/exp1_probe_directions.npz``) -- the same ones §4.4 used, so
    the headline ``mft_frac`` reproduces the published geometry;
  * persona is the **Instruct** probe direction
    (``olmo3_instruct/persona_directions.npz``) to match the Instruct refusal
    direction; the base persona is reported alongside as a sensitivity check;
  * headline layer 16 = the refusal ablation / §4.4 geometry layer; the stable
    band 15..31 (Appendix B) is summarised as a band mean.

Usage:
    python papers/5_moral_alignment/scripts/measure_refusal_decomposition.py \
        --refusal-npz papers/5_moral_alignment/outputs/heretic/refusal_directions.npz \
        --moral-npz papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz \
        --persona-npz papers/5_moral_alignment/outputs/olmo3_instruct/persona_directions.npz \
        --output papers/5_moral_alignment/outputs/measurement/refusal_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402
from moral_dependency import build_subspace_basis  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402

logger = logging.getLogger(__name__)

# Defaults relative to the paper root (parent of scripts/).
_PAPER_ROOT = Path(__file__).resolve().parent.parent
_DEF_REFUSAL = _PAPER_ROOT / "outputs/heretic/refusal_directions.npz"
_DEF_MORAL = _PAPER_ROOT / "outputs/olmo3_base/exp1_probe_directions.npz"
_DEF_PERSONA = _PAPER_ROOT / "outputs/olmo3_instruct/persona_directions.npz"
_DEF_PERSONA_BASE = _PAPER_ROOT / "outputs/olmo3_base/persona_directions.npz"
_DEF_OUT = _PAPER_ROOT / "outputs/measurement/refusal_decomposition.json"


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def decompose_layer(
    r: np.ndarray,
    mft_basis: np.ndarray,
    persona: np.ndarray,
    foundations: list[str],
    foundation_vecs: dict[str, np.ndarray],
) -> dict:
    """Energy decomposition of one unit refusal vector at one layer.

    Args:
        r: Refusal direction (will be unit-normalised defensively).
        mft_basis: ``(k, hidden)`` orthonormal rows spanning the moral subspace.
        persona: Persona direction (unit-normalised defensively).
        foundations: Foundation names present at this layer (for per-cos report).
        foundation_vecs: ``{foundation: vec}`` at this layer (raw, for cosines).

    Returns:
        Dict of the partition fractions plus diagnostics. ``mft_frac +
        persona_unique_frac + residual_frac == 1.0``.
    """
    r = _unit(r)
    p = _unit(persona)

    # --- moral-subspace energy (orthonormal basis -> coeffs are the projection)
    coeff_m = mft_basis @ r              # (k,)
    mft_frac = float(coeff_m @ coeff_m)  # ||proj_S(r)||^2 (r is unit)

    # --- persona component orthogonal to the moral subspace ---
    p_in_s = mft_basis.T @ (mft_basis @ p)
    p_perp = p - p_in_s
    n_perp = float(np.linalg.norm(p_perp))
    if n_perp > 1e-6:
        u = p_perp / n_perp
        persona_unique_frac = float((r @ u) ** 2)
    else:  # persona lies entirely inside the moral subspace (not expected)
        persona_unique_frac = 0.0

    residual_frac = 1.0 - mft_frac - persona_unique_frac

    # --- diagnostics (overlapping, not part of the partition) ---
    cos_rp = float(r @ p)
    persona_total_frac = cos_rp ** 2          # energy along the raw persona dir
    cos_to_found = {
        FOUNDATION_SHORT.get(f, f): round(float(r @ _unit(foundation_vecs[f])), 4)
        for f in foundations
    }
    return {
        "mft_frac": round(mft_frac, 6),
        "persona_unique_frac": round(persona_unique_frac, 6),
        "residual_frac": round(residual_frac, 6),
        "mft_norm_ratio": round(float(np.sqrt(max(mft_frac, 0.0))), 6),
        "cos_refusal_persona": round(cos_rp, 6),
        "persona_total_frac": round(persona_total_frac, 6),
        "persona_in_subspace_norm": round(float(np.linalg.norm(p_in_s)), 6),
        "mean_abs_cos_to_foundations": round(
            float(np.mean([abs(v) for v in cos_to_found.values()])), 6
        ),
        "cos_to_foundations": cos_to_found,
        "mft_rank": int(mft_basis.shape[0]),
    }


def verdict(mft: float, persona_unique: float, residual: float, cos_rp: float) -> dict:
    """Map the partition to the §6.4 intervention-target decision."""
    # Thresholds are deliberately coarse; the JSON carries the raw numbers.
    if mft >= 0.10:
        target = "amplify_existing_mft_overlap"
        rationale = ("refusal already carries non-trivial moral-subspace energy "
                     "(>=0.10); amplify existing structure (easiest).")
    elif persona_unique >= mft and (persona_unique >= 0.10 or abs(cos_rp) >= 0.30):
        target = "couple_persona_to_mft"
        rationale = ("refusal is more persona-carried than moral; target the "
                     "persona<->MFT coupling (Approach 4c).")
    else:
        target = "create_overlap_from_scratch"
        rationale = ("refusal is dominated by the residual and only weakly "
                     "persona-carried; coupling must create overlap from scratch "
                     "(hardest).")
    return {"target": target, "rationale": rationale}


def _band_mean(per_layer: dict[int, dict], band: list[int], key: str) -> float:
    vals = [per_layer[L][key] for L in band if L in per_layer]
    return round(float(np.mean(vals)), 6) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measurement 1: refusal-direction decomposition.")
    ap.add_argument("--refusal-npz", default=str(_DEF_REFUSAL))
    ap.add_argument("--moral-npz", default=str(_DEF_MORAL))
    ap.add_argument("--persona-npz", default=str(_DEF_PERSONA),
                    help="Primary persona npz (Instruct, to match Instruct refusal).")
    ap.add_argument("--persona-base-npz", default=str(_DEF_PERSONA_BASE),
                    help="Sensitivity persona npz (base). Set '' to skip.")
    ap.add_argument("--persona-key", default="persona",
                    choices=["persona", "persona_meandiff"])
    ap.add_argument("--refusal-key", default="refusal")
    ap.add_argument("--refusal-layer", type=int, default=16,
                    help="Headline layer (ablation / §4.4 geometry layer).")
    ap.add_argument("--band", type=int, nargs=2, default=[15, 31],
                    help="Stable band (inclusive) summarised as a mean.")
    ap.add_argument("--output", default=str(_DEF_OUT))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    refusal = du.load_directions(args.refusal_npz)[args.refusal_key]
    moral = du.load_directions(args.moral_npz)
    persona_all = du.load_directions(args.persona_npz)
    if args.persona_key not in persona_all:
        raise KeyError(f"{args.persona_key!r} not in {args.persona_npz} "
                       f"(have {list(persona_all)})")
    persona = persona_all[args.persona_key]

    n_layers = max(refusal) + 1
    mft_basis_by_layer, mft_rank, mft_names = build_subspace_basis(
        moral, kind="probe", n_layers=n_layers
    )
    foundations = [f for f in FOUNDATION_ORDER if f in moral]

    band = list(range(args.band[0], args.band[1] + 1))

    def run(persona_dirs: dict[int, np.ndarray]) -> dict[int, dict]:
        per_layer: dict[int, dict] = {}
        for L in sorted(refusal):
            if L not in mft_basis_by_layer or L not in persona_dirs:
                continue
            found_at_L = {f: moral[f][L] for f in foundations if L in moral[f]}
            per_layer[L] = decompose_layer(
                refusal[L], mft_basis_by_layer[L], persona_dirs[L],
                list(found_at_L), found_at_L,
            )
        return per_layer

    primary = run(persona)

    hl = args.refusal_layer
    if hl not in primary:
        raise RuntimeError(f"Refusal layer {hl} missing from decomposition.")
    hd = primary[hl]
    v = verdict(hd["mft_frac"], hd["persona_unique_frac"],
                hd["residual_frac"], hd["cos_refusal_persona"])

    band_summary = {
        "band": [args.band[0], args.band[1]],
        "mft_frac_mean": _band_mean(primary, band, "mft_frac"),
        "persona_unique_frac_mean": _band_mean(primary, band, "persona_unique_frac"),
        "residual_frac_mean": _band_mean(primary, band, "residual_frac"),
        "cos_refusal_persona_mean": _band_mean(primary, band, "cos_refusal_persona"),
    }

    # Cross-check: sqrt(mft_frac) at headline layer vs cached §4.4 norm-ratio.
    f4_check = {
        "mft_norm_ratio_here": hd["mft_norm_ratio"],
        "f4_published_subspace_projection_fraction": 0.1044,
        "matches_f4": abs(hd["mft_norm_ratio"] - 0.1044) < 0.01,
    }

    payload = {
        "analysis": "refusal_decomposition",
        "description": "Energy decomposition of the Instruct refusal direction "
                       "onto {moral subspace, persona-unique, residual}.",
        "inputs": {
            "refusal_npz": args.refusal_npz,
            "refusal_key": args.refusal_key,
            "moral_npz": args.moral_npz,
            "moral_kind": "probe",
            "moral_directions": mft_names,
            "persona_npz": args.persona_npz,
            "persona_key": args.persona_key,
            "persona_source": "instruct",
        },
        "conventions": {
            "directions": "unit-norm float32, 4096-dim, layers 0..31",
            "partition": "mft_frac + persona_unique_frac + residual_frac == 1.0 "
                         "(squared-norm energy of the unit refusal vector)",
            "mft_basis": "SVD-orthonormalised span of the 6 base probe "
                         "foundation directions (same basis the dependency "
                         "metric ablates)",
            "note": "MFT directions are base; refusal+persona are Instruct. "
                    "F4 used base MFT directions against the Instruct refusal "
                    "direction, so the headline mft_frac is comparable.",
        },
        "headline_layer": hl,
        "headline": hd,
        "verdict": v,
        "f4_crosscheck": f4_check,
        "band_summary": band_summary,
        "per_layer": {str(L): primary[L] for L in sorted(primary)},
    }

    # --- base-persona sensitivity ---
    if args.persona_base_npz:
        pb = du.load_directions(args.persona_base_npz).get(args.persona_key)
        if pb is not None:
            base_pl = run(pb)
            payload["persona_base_sensitivity"] = {
                "persona_npz": args.persona_base_npz,
                "persona_source": "base",
                "headline": base_pl.get(hl),
                "band_summary": {
                    "persona_unique_frac_mean": _band_mean(base_pl, band, "persona_unique_frac"),
                    "cos_refusal_persona_mean": _band_mean(base_pl, band, "cos_refusal_persona"),
                },
            }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"\nWrote {out}")
    print(f"Refusal-direction energy decomposition @ layer {hl} "
          f"(Instruct refusal, base MFT, Instruct persona):")
    print(f"  MFT subspace      : {hd['mft_frac']:.4f}  "
          f"(norm-ratio {hd['mft_norm_ratio']:.4f}; F4 published 0.1044 "
          f"{'OK' if f4_check['matches_f4'] else 'MISMATCH'})")
    print(f"  persona (unique)  : {hd['persona_unique_frac']:.4f}  "
          f"(cos(refusal,persona) = {hd['cos_refusal_persona']:+.4f})")
    print(f"  residual          : {hd['residual_frac']:.4f}")
    print(f"  -> sum = {hd['mft_frac']+hd['persona_unique_frac']+hd['residual_frac']:.4f}")
    print(f"  band {args.band[0]}-{args.band[1]} means: "
          f"MFT {band_summary['mft_frac_mean']:.4f}, "
          f"persona {band_summary['persona_unique_frac_mean']:.4f}, "
          f"residual {band_summary['residual_frac_mean']:.4f}")
    print(f"  VERDICT: {v['target']} -- {v['rationale']}")


if __name__ == "__main__":
    main()
