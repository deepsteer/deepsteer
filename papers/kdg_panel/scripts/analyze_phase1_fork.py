#!/usr/bin/env python3
"""Amendment P1-A1 fork (KDG_PHASE1_SPEC.md; pushed d39deab before this ran): frame-specific scale
normalization of the Z1b(ii) contrast, from the committed per-scenario arrays of Z1.

    python3 papers/kdg_panel/scripts/analyze_phase1_fork.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
N_BOOT, SEED = 10_000, 0


def main() -> int:
    rows = list(csv.DictReader(open(DATA / "per_scenario_phase1_z.csv")))
    # "the fork reads the same 192 scenarios Z1 used": fail loudly otherwise
    assert len(rows) == 192, len(rows)
    g = {k: np.asarray([float(r[k]) for r in rows]) for k in rows[0] if k != "scenario_id"}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, 192, size=(N_BOOT, 192))

    def summ(x):
        lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
        return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)],
                "point_in_ci": bool(lo <= x.mean() <= hi)}

    out = {"amendment": "P1-A1 (d39deab)", "n": 192, "n_boot": N_BOOT, "seed": SEED}
    for m in ("base", "instruct"):
        out[m] = {
            "act_per_scale": summ(g[f"S_act_{m}"] / g[f"sigma_D_{m}"]),
            "judge_per_scale": summ(g[f"S_judge_{m}"] / g[f"sigma_J_{m}"]),
            "E_fs": summ(g[f"S_act_{m}"] / g[f"sigma_D_{m}"] - g[f"S_judge_{m}"] / g[f"sigma_J_{m}"]),
        }
    d_act = g["S_act_instruct"] / g["sigma_D_instruct"] - g["S_act_base"] / g["sigma_D_base"]
    d_jud = g["S_judge_instruct"] / g["sigma_J_instruct"] - g["S_judge_base"] / g["sigma_J_base"]
    d_fs = d_act - d_jud
    out["diff_instruct_minus_base"] = {"act_per_scale": summ(d_act), "judge_per_scale": summ(d_jud),
                                       "E_fs": summ(d_fs)}
    lo, hi = out["diff_instruct_minus_base"]["E_fs"]["ci95"]
    se = (hi - lo) / (2 * 1.96)
    out["mde_E_fs"] = 2.8 * se
    out["verdict"] = ("widening_survives_frame_specific_sharpening" if lo > 0
                      else "reversed" if hi < 0 else "unresolved")
    (DATA / "analysis_p1a1_fork.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
