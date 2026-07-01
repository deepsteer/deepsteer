#!/usr/bin/env python3
"""Direction 1, Phase 2, Track-1 — σ* fragility (G5 input). Numpy-only; no model load.

Reuses G2's saved surf activations (g2_eval_acts.npz) and the extracted directions. For a
direction d, σ* = the smallest Gaussian-noise scale σ (grid S, max 10.0) at which the
transfer accuracy of d separating the eval moral/neutral pairs drops below τ=0.6 (Paper 1).
Compares the Base-V_moral primary direction against the MFT baseline (mean of the 6
foundation directions), RMS-normalized (raw σ* is activation-scale-confounded;
project memory). PREREGISTRATION §4: criterion form fixed (σ*_RMS(V_moral) ≥ σ*_RMS(MFT) − δ),
δ deferred -> reported, not gated. Also reports the eff-dim contrast (Phase 4.2).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
from deepsteer.directions import extraction as du  # noqa: E402

TAU = 0.6
GRID = np.concatenate([np.arange(0.0, 2.0, 0.25), np.arange(2.0, 10.5, 0.5)])  # max 10.0
N_TRIALS = 5
SEED = 0


def sigma_star(X: np.ndarray, y: np.ndarray, d: np.ndarray, rng) -> float:
    """Smallest σ in GRID where mean transfer accuracy under N(0,σ²I) drops below τ."""
    for sigma in GRID:
        if sigma == 0.0:
            acc = du.transfer_metrics(X, y, d)["acc_midpoint"]
        else:
            acc = np.mean([du.transfer_metrics(X + rng.normal(0, sigma, X.shape), y, d
                                               )["acc_midpoint"] for _ in range(N_TRIALS)])
        if acc < TAU:
            return float(sigma)
    return float(GRID[-1])  # never dropped below τ on the grid


def uncentered_eff_dim(dirs: list[np.ndarray], thresh: float = 0.9) -> int:
    M = np.stack(dirs)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), thresh)) + 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Track-1 σ* fragility (V_moral vs MFT).")
    ap.add_argument("--artifacts", default=str(HERE.parent / "outputs" / "phase2" / "base"))
    args = ap.parse_args()

    art = Path(args.artifacts)
    meta = json.load(open(art / "extract_meta.json"))
    layer = meta["match_layer"]
    acts = np.load(art / "g2_eval_acts.npz")
    layer = int(acts["layer"])

    d_vmoral = du.load_directions(art / "moral_directions.npz")["moral_stories"][layer]
    mft = du.load_directions(art / "mft_directions.npz")
    mft_dirs = [mft[f][layer] for f in mft]
    d_mft = np.mean(np.stack(mft_dirs), axis=0)
    d_mft = d_mft / (np.linalg.norm(d_mft) + 1e-12)

    rng = np.random.default_rng(SEED)
    report: dict = {"tau": TAU, "grid_max": float(GRID[-1]), "n_trials": N_TRIALS, "slices": {}}
    for sl in ("narrative", "declarative"):
        if f"X_{sl}" not in acts:
            continue
        X, y = acts[f"X_{sl}"].astype(np.float64), acts[f"y_{sl}"]
        rms = float(np.sqrt(np.mean(X ** 2)))
        s_v, s_m = sigma_star(X, y, d_vmoral, rng), sigma_star(X, y, d_mft, rng)
        report["slices"][sl] = {
            "n": int(X.shape[0]), "rms_activation": round(rms, 4),
            "sigma_star_raw": {"v_moral": s_v, "mft": s_m},
            "sigma_star_rms": {"v_moral": round(s_v / rms, 4), "mft": round(s_m / rms, 4)},
            "v_moral_minus_mft_rms": round((s_v - s_m) / rms, 4),
            "interpretation": ("V_moral no more fragile than MFT" if s_v >= s_m
                               else "V_moral MORE fragile than MFT -- investigate (plan §6)"),
        }

    # eff-dim contrast (Phase 4.2): how much richer is V_moral than the 6-MFT subspace?
    vm = np.load(art / "v_moral.npz", allow_pickle=True)
    report["eff_dim"] = {"v_moral": int(vm["eff_dim"]),
                         "mft_subspace_uncentered": uncentered_eff_dim(mft_dirs, 0.9)}

    with open(art / "track1_result.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print("Track-1 σ* (RMS-normalized; criterion form, δ deferred):")
    for sl, r in report["slices"].items():
        print(f"  [{sl}] σ*_RMS  V_moral={r['sigma_star_rms']['v_moral']} "
              f"MFT={r['sigma_star_rms']['mft']}  ({r['interpretation']})")
    print(f"  eff_dim: V_moral={report['eff_dim']['v_moral']} "
          f"MFT_subspace={report['eff_dim']['mft_subspace_uncentered']}")


if __name__ == "__main__":
    main()
