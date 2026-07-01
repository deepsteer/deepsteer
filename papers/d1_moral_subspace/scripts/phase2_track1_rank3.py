#!/usr/bin/env python3
"""Track-1 σ* on the RANK-3 headline instrument (not the abandoned single-source 385-dim).

The committed Track-1 (phase2_track1.py) measures σ* of the moral_stories DIRECTION (the
classification-primary of the rank-3 span) vs the mean-of-6 MFT direction. That is already a
direction-level robustness number, but Q3 asked for the SPAN characterized explicitly: this
re-measures σ* with the classifier constrained to each subspace -- the eval moral/neutral
mean-diff RESTRICTED to the span (the span's best in-subspace linear moral classifier) -- for
the rank-3 V_moral span vs the 6-foundation MFT span, same τ=0.6 / grid / RMS-normalization.

So the paper can state robustness on the PUBLISHED instrument, both as (a) the primary-direction
σ* (committed) and (b) this span-restricted σ*. Numpy-only; no model. Reuses the committed
g2_eval_acts + directions (base). See PREREGISTRATION Q3 (bundled with the GPT-OSS run's other
two numbers, but this one needs no GPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
from deepsteer.directions import extraction as du  # noqa: E402
from phase2_g3_respec import P2, _ortho, source_dirs  # noqa: E402
from phase2_track1 import GRID, SEED, sigma_star  # noqa: E402


def restricted_classifier(X: np.ndarray, y: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Best in-subspace linear moral classifier: the eval mean-diff restricted to span(Q)."""
    md = X[y == 1].mean(0) - X[y == 0].mean(0)
    d = Q @ (Q.T @ md)
    return d / (np.linalg.norm(d) + 1e-12)


def main() -> None:
    art = P2 / "base"
    acts = np.load(art / "g2_eval_acts.npz")
    layer = int(acts["layer"])

    # rank-3 V_moral span (published instrument) + 6-foundation MFT span, both orthonormalized.
    vdirs = source_dirs("base", P2 / "axis", layer)
    Qv = _ortho([vdirs[s] for s in ("moral_stories", "fables", "ethics")])
    mft = du.load_directions(art / "mft_directions.npz")
    Qm = _ortho([mft[f][layer] / np.linalg.norm(mft[f][layer]) for f in mft])

    rng = np.random.default_rng(SEED)
    report: dict = {"tau": 0.6, "grid_max": float(GRID[-1]),
                    "design": "span-restricted classifier (eval mean-diff projected onto the "
                              "span); rank-3 V_moral vs 6-foundation MFT",
                    "rank": {"v_moral": int(Qv.shape[1]), "mft": int(Qm.shape[1])}, "slices": {}}
    for sl in ("narrative", "declarative"):
        if f"X_{sl}" not in acts:
            continue
        X, y = acts[f"X_{sl}"].astype(np.float64), acts[f"y_{sl}"]
        rms = float(np.sqrt(np.mean(X ** 2)))
        dv, dm = restricted_classifier(X, y, Qv), restricted_classifier(X, y, Qm)
        s_v, s_m = sigma_star(X, y, dv, rng), sigma_star(X, y, dm, rng)
        report["slices"][sl] = {
            "n": int(X.shape[0]), "rms_activation": round(rms, 4),
            "sigma_star_rms": {"v_moral_rank3": round(s_v / rms, 4), "mft": round(s_m / rms, 4)},
            "v_moral_minus_mft_rms": round((s_v - s_m) / rms, 4),
            "interpretation": ("rank-3 V_moral no more fragile than MFT" if s_v >= s_m
                               else "rank-3 V_moral MORE fragile than MFT -- investigate"),
        }
    (art / "track1_rank3_result.json").write_text(json.dumps(report, indent=2))
    print("Track-1 σ* on the RANK-3 span (span-restricted classifier, RMS-normalized):")
    for sl, r in report["slices"].items():
        print(f"  [{sl}] σ*_RMS  rank-3 V_moral={r['sigma_star_rms']['v_moral_rank3']} "
              f"MFT={r['sigma_star_rms']['mft']}  ({r['interpretation']})")


if __name__ == "__main__":
    main()
