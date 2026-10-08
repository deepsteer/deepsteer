#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A12 fork (pushed 699d541 before this ran): C0-dm agreement predicted by T = 0.7
sampling of the forced primary readout's own per-permutation letter distributions.

    python3 papers/kdg_panel/scripts/analyze_gptoss_c0dm_fork.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

T, SIMS, PERMS = 0.7, 10_000, 4


def main() -> int:
    d = G.DEFAULT_DIR
    rows = A._rows(d / "c0_forced_primary.jsonl")
    z = np.load(d / "c0_forced_primary.npz")
    lp, tid = z["logp_decision"].astype(np.float64), z["option_token_ids"]
    dist: dict[str, list] = {}
    for k, r in enumerate(rows):
        if r["seed"] >= PERMS:
            continue
        letters = sorted(r["order"])
        p = np.exp(lp[k, [int(tid[k]["ABCDE".index(L)]) for L in letters]])
        p = p / p.sum()
        q = p ** (1 / T)
        opts = [r["order"][L] for L in letters]
        dist.setdefault(r["scenario_id"], []).append((opts, q / q.sum()))
    F = G.forced_argmax(d, "c0_forced_primary")
    obs = json.loads((A.DATA / "analysis_gptoss_c0dm.json").read_text())["c0_dm"]
    ids = sorted(dist)
    rng = np.random.default_rng(0)
    sims = np.empty(SIMS)
    for i in range(SIMS):
        hits = 0
        for s in ids:
            draws = [opts[rng.choice(len(opts), p=q)] for opts, q in dist[s]]
            hits += kh.strict_majority(draws) == F[s]
        sims[i] = hits / len(ids)
    lo, hi = (float(x) for x in np.percentile(sims, [2.5, 97.5]))
    a = obs["agreement"]
    verdict = "sampling_limited" if lo <= a <= hi else ("residual_beyond_sampling" if a < lo
                                                        else "above_prediction")
    rep = {"rule": "KDG_GPTOSS_SPEC G-A12 (699d541)", "n_scenarios": len(ids),
           "predicted_mean": float(sims.mean()), "predicted_95": [lo, hi],
           "p_valid_readout_passes_0_80": float(np.mean(sims >= 0.80)),
           "observed": a, "registered_verdict": obs["verdict"], "fork_verdict": verdict}
    (A.DATA / "analysis_gptoss_c0dm_fork.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
