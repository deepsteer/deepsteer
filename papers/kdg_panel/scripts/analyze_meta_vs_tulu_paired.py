#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 2: Meta's minus Tulu 3's pressure-attributable excess, paired.

    python3 papers/kdg_panel/scripts/analyze_meta_vs_tulu_paired.py

Same Llama-3.1 base, two recipes. Per scenario on the scenarios both models engage (option mass at
least the floor in all four cells; the 586 screened by no model), E_prob under each model's own
template (letter-only readout, as the recipe table), then the paired difference Meta - Tulu 3 (final) with
the scenario bootstrap (10,000 draws, seed 0) and its MDE.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_own_screen as OS  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import analyze_screen_rates as SR  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    E = {}
    for key in ("llama31_instruct_meta", "tulu3_final"):
        T = A.four_cells([SR.MODELS[key]], OS.CHAT, status)
        E[key] = {s: A.scales(T[s])["E_prob"] for s in T if T[s]["mass_min"] >= A.FLOOR}
    common = sorted(set(E["llama31_instruct_meta"]) & set(E["tulu3_final"]))
    meta = np.array([E["llama31_instruct_meta"][s] for s in common])
    tulu = np.array([E["tulu3_final"][s] for s in common])
    rep = {"rule": "KDG_PHASE1_SPEC P1-A17 item 2", "n_common": len(common),
           "meta_minus_tulu": A.boot(meta - tulu), "meta_on_common": A.boot(meta),
           "tulu_on_common": A.boot(tulu)}
    (A.DATA / "analysis_meta_vs_tulu_paired.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
