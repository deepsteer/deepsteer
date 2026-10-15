#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 4: clustered sensitivity for the own-screen vs twin-screen overlap.

    python3 papers/kdg_panel/scripts/analyze_v2_overlap_cluster.py

The P1-A13 selection-matched null, E_sel = mean E over the own-screened set minus mean E_rev over
the twin-screened set, was bootstrapped with the two sets resampled independently, although a
scenario can sit in both (GPT-OSS-20B's dose-0 row, from analyze_gptoss, appended last so the
four panel models' draws are unchanged). Here each draw resamples scenario ids once from the
union of the two sets (10,000 draws, seed 0) and computes each arm over the drawn ids that belong
to it, so overlapping scenarios move together. Registered (independent) interval beside it;
letter-only readout.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    reg = json.loads((A.DATA / "analysis_own_screen.json").read_text())["selection_null"]
    rep: dict = {"rule": "KDG_PHASE1_SPEC P1-A17 item 4", "n_boot": A.N_BOOT, "seed": A.SEED,
                 "models": {}}
    sel_g = json.loads((A.DATA / "analysis_gptoss.json").read_text())["c3"]["own_screen"]
    reg["gpt_oss_20b_dose0"] = {"E_sel": {"ci95": sel_g["selection_null"]["ci95"]},
                                "verdict": ("survives_selection"
                                            if sel_g["selection_null"]["ci95"][0] > 0
                                            else "within_selection")}
    runs = [(k, SR.MODELS[k], status) for k in OS.MODELS]
    runs.append(("gpt_oss_20b_dose0", G.DEFAULT_DIR, G._status(scen)))
    rng = np.random.default_rng(A.SEED)
    for key, d, status in runs:
        T = A.four_cells([d], OS.CHAT, status)
        own = [s for s in OS.screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
        tw = [s for s in OS.twin_screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
        union = sorted(set(own) | set(tw))
        e = np.array([A.scales(T[s])["E_prob"] for s in union])
        in_own = np.array([s in set(own) for s in union])
        in_tw = np.array([s in set(tw) for s in union])
        draws = []
        for _ in range(A.N_BOOT):
            cnt = np.bincount(rng.integers(0, len(union), len(union)), minlength=len(union))
            wo, wt = cnt * in_own, cnt * in_tw
            if wo.sum() == 0 or wt.sum() == 0:
                continue
            draws.append((wo @ e) / wo.sum() - (wt @ -e) / wt.sum())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        point = e[in_own].mean() - (-e[in_tw]).mean()
        rep["models"][key] = {"n_own": int(in_own.sum()), "n_twin_screen": int(in_tw.sum()),
                              "n_overlap": int((in_own & in_tw).sum()),
                              "E_sel": float(point), "registered_ci95": reg[key]["E_sel"]["ci95"],
                              "clustered_ci95": [float(lo), float(hi)],
                              "registered_verdict": reg[key]["verdict"],
                              "clustered_verdict": ("survives_selection" if lo > 0
                                                    else "within_selection")}
        print(key, json.dumps(rep["models"][key]))
    (A.DATA / "analysis_v2_overlap_cluster.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
