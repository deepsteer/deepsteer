#!/usr/bin/env python3
"""TASK A (zero-GPU): bootstrap difference-CIs for Paper 3 headline geometry scalars.

Operates entirely on cached probe directions (no model load), identical to
dilemma_compositionality_baselines.py:
  - dilemma directions:  outputs/dilemma_probing/dilemma_probe_directions.npz
  - foundation directions: outputs/exp1_2_3/exp1_probe_directions.npz

Computes:
  1. Membership Δ-CI (matched - mismatched), bootstrapping over the 15 dilemmas,
     for three definitions: cross-layer mean, aggregate-peak layer, per-pair-peak
     (the 0.118 vs 0.044 headline; extremum-biased, flagged).
  2. Shared-component gap Δ-CI (share vs no-share dilemma-pair cosine, layer 13),
     node-bootstrap over the 15 dilemmas.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np

FOUNDATIONS = ["care", "fairness", "liberty", "loyalty", "authority", "sanctity"]
FULL = {
    "care": "care_harm", "fairness": "fairness_cheating", "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal", "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}
N_LAYERS = 16
B = 10000
SEED = 42
SHARED_LAYER = 13
BASE = Path("papers/3_moral_geometry/outputs")
DILEMMA_NPZ = BASE / "dilemma_probing" / "dilemma_probe_directions.npz"
FOUNDATION_NPZ = BASE / "exp1_2_3" / "exp1_probe_directions.npz"
OUT_DIR = BASE / "nonmoral_control"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def subspace_membership(w, a, b):
    """Fraction of w's variance in span{a,b} (identical to baselines script)."""
    e1 = a.copy()
    e2 = b - np.dot(b, e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-10:
        return float(np.dot(w, e1) ** 2)
    e2 = e2 / n2
    return float(np.dot(w, e1) ** 2 + np.dot(w, e2) ** 2)


def ci(arr, lo=2.5, hi=97.5):
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))


def main():
    dil = np.load(DILEMMA_NPZ)
    fnd = np.load(FOUNDATION_NPZ)
    dilemma_pairs = sorted({k.replace("dilemma_", "").rsplit("_layer", 1)[0] for k in dil.keys()})
    n = len(dilemma_pairs)
    all_found_pairs = list(combinations(FOUNDATIONS, 2))

    # Per-dilemma, per-layer matched + mismatched membership arrays.
    matched = np.zeros((n, N_LAYERS))     # [dilemma, layer]
    mismatched = np.zeros((n, N_LAYERS))
    for di, dp in enumerate(dilemma_pairs):
        f1, f2 = dp.split("-")
        comp = {f1, f2}
        for L in range(N_LAYERS):
            w = _unit(dil[f"dilemma_{dp}_layer{L}"])
            a = _unit(fnd[f"{FULL[f1]}_layer{L}"])
            b = _unit(fnd[f"{FULL[f2]}_layer{L}"])
            matched[di, L] = subspace_membership(w, a, b)
            mm = []
            for g1, g2 in all_found_pairs:
                if comp & {g1, g2}:
                    continue
                ga = _unit(fnd[f"{FULL[g1]}_layer{L}"])
                gb = _unit(fnd[f"{FULL[g2]}_layer{L}"])
                mm.append(subspace_membership(w, ga, gb))
            mismatched[di, L] = float(np.mean(mm))

    # --- Point values under three definitions ---
    # cross-layer mean per dilemma
    m_cl = matched.mean(axis=1)   # [dilemma]
    mm_cl = mismatched.mean(axis=1)
    agg_peak = int(np.argmax(matched.mean(axis=0)))
    m_ap = matched[:, agg_peak]
    mm_ap = mismatched[:, agg_peak]
    # per-pair-peak: each dilemma's own peak matched layer; mismatched at that same layer
    peakL = matched.argmax(axis=1)                          # [dilemma]
    m_pp = matched[np.arange(n), peakL]
    mm_pp = mismatched[np.arange(n), peakL]

    defs = {
        "cross_layer_mean": (m_cl, mm_cl),
        f"aggregate_peak_layer_{agg_peak}": (m_ap, mm_ap),
        "per_pair_peak_EXTREMUM": (m_pp, mm_pp),
    }

    rng = np.random.RandomState(SEED)
    boot_idx = rng.randint(0, n, size=(B, n))  # paired resample of the 15 dilemmas

    membership_out = {}
    for name, (mvec, mmvec) in defs.items():
        gap_pt = float(mvec.mean() - mmvec.mean())
        m_boot = mvec[boot_idx].mean(axis=1)
        mm_boot = mmvec[boot_idx].mean(axis=1)
        gap_boot = m_boot - mm_boot                    # difference bootstrap (paired)
        lo, hi = ci(gap_boot)
        membership_out[name] = {
            "matched_mean": float(mvec.mean()),
            "mismatched_mean": float(mmvec.mean()),
            "gap_point": gap_pt,
            "gap_CI95": [lo, hi],
            "gap_excludes_0": bool(lo > 0),
            "matched_CI95": list(ci(mvec[boot_idx].mean(axis=1))),
            "mismatched_CI95": list(ci(mmvec[boot_idx].mean(axis=1))),
            "frac_boot_gap_le_0": float(np.mean(gap_boot <= 0)),
        }

    # --- Shared-component gap Δ-CI at layer 13 (node bootstrap over 15 dilemmas) ---
    labels = [tuple(dp.split("-")) for dp in dilemma_pairs]
    mat = np.stack([_unit(dil[f"dilemma_{dp}_layer{SHARED_LAYER}"]) for dp in dilemma_pairs])
    cos = mat @ mat.T
    share_pt, noshare_pt = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (share_pt if set(labels[i]) & set(labels[j]) else noshare_pt).append(cos[i, j])
    gap_sc_pt = float(np.mean(share_pt) - np.mean(noshare_pt))

    gap_sc_boot = []
    for b in range(B):
        idx = rng.randint(0, n, size=n)
        sh, nsh = [], []
        for a in range(n):
            for c in range(a + 1, n):
                ia, ic = idx[a], idx[c]
                if ia == ic:      # skip self-pairs (cos = 1) — dyadic node bootstrap
                    continue
                (sh if set(labels[ia]) & set(labels[ic]) else nsh).append(cos[ia, ic])
        if sh and nsh:
            gap_sc_boot.append(np.mean(sh) - np.mean(nsh))
    gap_sc_boot = np.array(gap_sc_boot)
    lo_sc, hi_sc = ci(gap_sc_boot)

    shared_out = {
        "layer": SHARED_LAYER,
        "share_mean": float(np.mean(share_pt)),
        "noshare_mean": float(np.mean(noshare_pt)),
        "gap_point": gap_sc_pt,
        "gap_CI95": [lo_sc, hi_sc],
        "gap_excludes_0": bool(lo_sc > 0),
        "frac_boot_gap_le_0": float(np.mean(gap_sc_boot <= 0)),
        "n_share_pairs": len(share_pt),
        "n_noshare_pairs": len(noshare_pt),
        "existing_exact_permutation_p": 9.999e-05,
    }

    out = {
        "task": "A_bootstrap_difference_CIs",
        "n_bootstrap": B, "seed": SEED, "n_dilemmas": n,
        "membership_difference_CI": membership_out,
        "shared_component_gap_CI": shared_out,
        "note": "Difference-CIs per estimator-traps trap 1 (paired Δ bootstrap, not CI overlap). "
                "Unit of resampling = the 15 dilemmas.",
    }
    path = OUT_DIR / "task_a_difference_cis.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("=== MEMBERSHIP Δ-CI (matched - mismatched), bootstrap over 15 dilemmas ===")
    for name, r in membership_out.items():
        print(f"  {name}: matched {r['matched_mean']:.4f} mismatched {r['mismatched_mean']:.4f} "
              f"| Δ={r['gap_point']:.4f} CI95[{r['gap_CI95'][0]:.4f},{r['gap_CI95'][1]:.4f}] "
              f"excl0={r['gap_excludes_0']} p_le0={r['frac_boot_gap_le_0']:.4f}")
    print("\n=== SHARED-COMPONENT gap Δ-CI (layer 13) ===")
    print(f"  share {shared_out['share_mean']:.4f} noshare {shared_out['noshare_mean']:.4f} "
          f"| Δ={shared_out['gap_point']:.4f} CI95[{lo_sc:.4f},{hi_sc:.4f}] "
          f"excl0={shared_out['gap_excludes_0']} p_le0={shared_out['frac_boot_gap_le_0']:.4f}")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
