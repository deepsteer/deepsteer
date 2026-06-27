#!/usr/bin/env python3
"""Direction 1, Phase 2 (GPU), stage 3: realize the null + control -> FROZEN artifact.

This is the hard sequence point of the two-step null (PREREGISTRATION §3.3). It computes,
from `v_moral.npz` ALONE plus the activation sample and the persona axis:

  * q95  -- 95th percentile of the rank-matched, activation-covariance-matched null
            (K random directions with the layer's empirical covariance, projected onto
            V_moral). The honest null: does refusal project higher than a typical
            activation-space direction of the same rank?
  * c    -- the non-moral semantic control (persona axis) projected onto V_moral.
  * isotropic_floor = sqrt(eff_dim / hidden), the analytic reference.

It writes `null_artifact.json` and **never reads the refusal direction** -- by construction
the cutoffs cannot have been tuned to the result. phase2_g3.py consumes this artifact
read-only. The driver runs this BEFORE g3. Predates-the-result is enforced by structure,
not intent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_moral_alignment" / "scripts"))
import direction_utils as du  # noqa: E402
from heretic_ablation import subspace_projection_fraction  # noqa: E402

K = 1000
MARGIN_M = 0.05
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2 stage 3: frozen null + control.")
    ap.add_argument("--artifacts", default=str(HERE.parent / "outputs" / "phase2"))
    args = ap.parse_args()

    out = Path(args.artifacts)
    vm = np.load(out / "v_moral.npz", allow_pickle=True)
    basis = [vm["basis"][i] for i in range(vm["basis"].shape[0])]
    r, layer = int(vm["eff_dim"]), int(vm["layer"])

    sample = np.load(out / "act_sample.npz")
    X = sample["X"].astype(np.float64)          # (n, hidden)
    hidden = X.shape[1]
    Xc = X - X.mean(axis=0, keepdims=True)
    n = Xc.shape[0]

    # covariance-matched random directions: d = Xc^T w, w ~ N(0, I_n) has cov ~ Xc^T Xc
    rng = np.random.default_rng(SEED)
    fracs = np.empty(K)
    for k in range(K):
        w = rng.standard_normal(n)
        d = Xc.T @ w
        fracs[k] = subspace_projection_fraction(d, basis)
    q95 = float(np.percentile(fracs, 95))
    isotropic_floor = float(np.sqrt(r / hidden))

    persona = du.load_directions(out / "persona_direction.npz")["persona"][layer]
    c = subspace_projection_fraction(persona, basis)

    artifact = {
        "frozen": True, "consumed_by": "phase2_g3.py (read-only)",
        "predates_refusal_projection": True,
        "eff_dim": r, "hidden": hidden, "layer": layer,
        "null_method": "activation-covariance-matched (Xc^T w), K=%d, seed=%d" % (K, SEED),
        "q95": round(q95, 4), "null_mean": round(float(fracs.mean()), 4),
        "isotropic_floor_sqrt_r_over_d": round(isotropic_floor, 4),
        "control_c_persona_projection": round(float(c), 4),
        "margin_M": MARGIN_M,
        "decision_rule": "G3 POSITIVE iff for BOTH points A,B: p > q95+M AND p > c+M",
    }
    with open(out / "null_artifact.json", "w") as fh:
        json.dump(artifact, fh, indent=2)
    print(f"FROZEN null artifact written | eff_dim={r} q95={q95:.4f} "
          f"c={c:.4f} iso_floor={isotropic_floor:.4f} M={MARGIN_M}")
    print("  (refusal direction NOT read here; g3 consumes this read-only)")


if __name__ == "__main__":
    main()
