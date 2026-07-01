#!/usr/bin/env python3
"""A3 (calibration): refusal variance-percentile ("spare-channel" analysis). Descriptive,
no gate. Pre-registered in CALIBRATION_PREREG.md A3.

For each tag, using act_sample X (n, d), n < d: draw K covariance-matched random directions
(the same generator the null uses), and report where the refusal direction's activation
variance r^T Sigma_hat r sits in that distribution (percentile). V_moral axes + persona are
reported as references. Stay inside the sample-covariance machinery: var(u) = ||Xc u||^2/(n-1)
(never eigendecompose the rank-deficient covariance).

Pre-registered reading: refusal in a low-variance channel (<= q10) is consistent with a
"narrow add-on" implementation and would mechanistically explain both easy ablation and the
below-null projection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepsteer.directions import extraction as du  # noqa: E402
from calibration_a1_ladder import TAGS, source_dirs  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
OUT = P2 / "calibration"
K = 2000
SEED = 0

# Committed refusal vectors per tag. think: not saved as npz (only projections) -> logged.
REFUSAL_VECS = {
    "base": {"P_A_proto": "refusal_base.npz"},
    "instruct": {"P_B_gate": "refusal_instruct.npz"},
    "think": {},  # refusal vectors not committed for Think
    "gpt_oss": {"P0": "gpt_oss/refusal_think_P0.npz", "P1": "gpt_oss/refusal_think_P1.npz",
                "P2": "gpt_oss/refusal_think_P2.npz", "P2_FULL": "gpt_oss/refusal_think_P2_FULL.npz",
                "P3": "gpt_oss/refusal_think_P3.npz"},
}


def variance(Xc: np.ndarray, u: np.ndarray, denom: float) -> float:
    """u^T Sigma_hat u = ||Xc u||^2 / (n-1) for a unit direction u."""
    return float((Xc @ du.unit_vector(u)) @ (Xc @ du.unit_vector(u)) / denom)


def percentile_of(val: float, ref: np.ndarray) -> float:
    return round(float((ref < val).mean() * 100.0), 2)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    out = {}
    for tag, (layer, axis_sub, _mft, _g3) in TAGS.items():
        X = np.load(P2 / tag / "act_sample.npz")["X"].astype(np.float64)
        Xc = X - X.mean(0, keepdims=True)
        n = Xc.shape[0]
        denom = n - 1
        # Reference distribution of variances over covariance-matched random directions.
        refs = np.empty(K)
        for i in range(K):
            r = Xc.T @ rng.standard_normal(n)
            refs[i] = variance(Xc, r, denom)
        ref_q = {f"q{p}": round(float(np.percentile(refs, p)), 4) for p in (10, 50, 90)}

        dirs = source_dirs(tag, axis_sub, layer)
        persona = du.load_directions(P2 / tag / "persona_direction.npz")["persona"][layer]
        entries = {}
        for name, u in [(f"vmoral_{s}", v) for s, v in dirs.items()] + [("persona", persona)]:
            var = variance(Xc, u, denom)
            entries[name] = {"variance": round(var, 4), "percentile": percentile_of(var, refs)}
        # Refusal(s), where the vector is committed.
        missing = []
        for name, rel in REFUSAL_VECS[tag].items():
            path = P2 / rel
            if not path.exists():
                missing.append(name); continue
            u = np.load(path)["refusal"]
            var = variance(Xc, u, denom)
            entries[f"refusal_{name}"] = {"variance": round(var, 4),
                                          "percentile": percentile_of(var, refs),
                                          "low_variance_channel_q10": bool(var <= np.percentile(refs, 10))}
        out[tag] = {"layer": layer, "n": int(n), "hidden": int(Xc.shape[1]),
                    "ref_variance_percentiles": ref_q, "directions": entries,
                    "refusal_vectors_missing": bool(not REFUSAL_VECS[tag])}
        print(f"[{tag}] refs q10={ref_q['q10']} q50={ref_q['q50']} q90={ref_q['q90']}")
        for name, e in entries.items():
            tail = "  <= q10 (narrow channel)" if e.get("low_variance_channel_q10") else ""
            print(f"    {name:22} var={e['variance']:.3f} pct={e['percentile']:.1f}{tail}")
        if not REFUSAL_VECS[tag]:
            print(f"    (Think refusal vectors not committed -> logged to MISSING_ARTIFACTS.md)")

    (OUT / "a3_variance_percentile.json").write_text(json.dumps(out, indent=2))
    # Log the one true gap (Think refusal vectors).
    missing_md = P2.parent / "MISSING_ARTIFACTS.md"
    with open(missing_md, "a") as fh:
        fh.write("\n## A3 (2026-07-01): Think refusal vectors not saved\n\n"
                 "- OLMo-3-Think P0-P3 refusal directions exist only as projections in "
                 "`think_g3_result.json`, not as `.npz` vectors -> A3 variance-percentile and "
                 "A4 refusal-p bootstrap cannot run for Think. Re-extract with per-vector saves "
                 "in B3 if the Think spare-channel / refusal CI is wanted.\n")


if __name__ == "__main__":
    main()
