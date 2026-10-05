#!/usr/bin/env python3
"""P1-A11 (pushed c263eb2 before this ran): each instruct model's pressure-attributable excess on
its own screen (the P1-A9 item 2 rule on its own letter-only chat cells), with the detection bar.

    python3 papers/kdg_panel/scripts/analyze_own_screen.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

CHAT = (
    "dl_chat_neutral",
    "jl_chat_neutral",
    "dl_chat_neutral_pressure_removed",
    "jl_chat_neutral_pressure_removed",
)
MODELS = ("olmo3_final", "llama31_instruct_meta", "tulu3_final", "qwen25_instruct")


def screen(d: Path, status: dict) -> list[str]:
    """The P1-A9 item 2 rule, as in analyze_screen_rates.main."""
    jl = A._rows(d / "jl_chat_neutral.jsonl")
    Dm = A.option_cell([d], "dl_chat_neutral", status)
    Jm = A.option_cell([d], "jl_chat_neutral", status)
    cons = Counter()
    for r in jl:
        if r["scenario_id"] in status and r["norm_status"] == "consistent":
            cons[r["scenario_id"]] += 1
    eng = [
        s for s in status if s in Dm and s in Jm and min(Dm[s]["mass"], Jm[s]["mass"]) >= A.FLOOR
    ]
    return sorted(
        s for s in eng if cons[s] >= 6 and (0.15 <= Dm[s]["p"] <= 0.85 or Dm[s]["p"] >= 0.85)
    )


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep: dict = {"amendment": "P1-A11 (c263eb2)", "n_boot": A.N_BOOT, "seed": A.SEED, "models": {}}
    for key in MODELS:
        d = SR.MODELS[key]
        ids = screen(d, status)
        T = A.four_cells([d], CHAT, status)
        use = [s for s in ids if s in T and T[s]["mass_min"] >= A.FLOOR]
        E = np.array([A.scales(T[s])["E_prob"] for s in use])
        b = A.boot(E)
        se = (b["ci95"][1] - b["ci95"][0]) / (2 * 1.96)
        rep["models"][key] = {
            "n_screened": len(ids),
            "n_used": len(use),
            "dropped_below_floor": len(ids) - len(use),
            "E_own_screen": b,
            "bar": 2.8 * se,
        }
        print(
            f"{key:22s} screen {len(ids):3d} used {len(use):3d}  E {b['mean']:+.4f} "
            f"[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  bar {2.8 * se:.4f}"
        )
    (A.DATA / "analysis_own_screen.json").write_text(json.dumps(rep, indent=1))
    return 0


def twin_screen(d: Path, status: dict) -> list[str]:
    """P1-A13: the identical P1-A9 rule applied to the pressure-removed cells."""
    jl = A._rows(d / "jl_chat_neutral_pressure_removed.jsonl")
    Dm = A.option_cell([d], "dl_chat_neutral_pressure_removed", status)
    Jm = A.option_cell([d], "jl_chat_neutral_pressure_removed", status)
    cons = Counter()
    for r in jl:
        if r["scenario_id"] in status and r["norm_status"] == "consistent":
            cons[r["scenario_id"]] += 1
    eng = [
        s for s in status if s in Dm and s in Jm and min(Dm[s]["mass"], Jm[s]["mass"]) >= A.FLOOR
    ]
    return sorted(
        s for s in eng if cons[s] >= 6 and (0.15 <= Dm[s]["p"] <= 0.85 or Dm[s]["p"] >= 0.85)
    )


def selection_null() -> int:
    """P1-A13 (pushed 1d2616d before this ran): E_sel = E_own − E_rev on the twin-screened set."""
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {
        s.id: {o.option_id: o.norm_status for o in s.options}
        for s in scen
        if not s.covariates.get("construction_flag") and not s.id.endswith("S")
    }
    rep = json.loads((A.DATA / "analysis_own_screen.json").read_text())
    rep["selection_null"] = {"amendment": "P1-A13 (1d2616d)"}
    rng = np.random.default_rng(A.SEED)
    for key in MODELS:
        d = SR.MODELS[key]
        T = A.four_cells([d], CHAT, status)
        own = [s for s in screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
        tw = [s for s in twin_screen(d, status) if s in T and T[s]["mass_min"] >= A.FLOOR]
        e_own = np.array([A.scales(T[s])["E_prob"] for s in own])
        e_rev = np.array([-A.scales(T[s])["E_prob"] for s in tw])
        bo = e_own[rng.integers(0, len(e_own), (A.N_BOOT, len(e_own)))].mean(1)
        br = e_rev[rng.integers(0, len(e_rev), (A.N_BOOT, len(e_rev)))].mean(1)
        diff = bo - br
        lo, hi = np.percentile(diff, [2.5, 97.5])
        out = {
            "n_own": len(own),
            "n_twin_screen": len(tw),
            "E_own": float(e_own.mean()),
            "E_rev": A.boot(e_rev),
            "E_sel": {"mean": float(e_own.mean() - e_rev.mean()), "ci95": [float(lo), float(hi)]},
            "verdict": "survives_selection" if lo > 0 else "within_selection",
        }
        rep["selection_null"][key] = out
        print(
            f"{key:22s} own {len(own):3d} E {e_own.mean():+.4f} | twin-screen {len(tw):3d} "
            f"E_rev {e_rev.mean():+.4f} | E_sel {out['E_sel']['mean']:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            f" -> {out['verdict']}"
        )
    (A.DATA / "analysis_own_screen.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(selection_null() if "--selection-null" in sys.argv else main())
