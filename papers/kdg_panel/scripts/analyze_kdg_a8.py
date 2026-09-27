#!/usr/bin/env python3
"""KDG-A8 discriminator (KDG_PHASE1_SPEC.md P1-A7, pushed 8d8b44d before this ran): at-rest lean
across SFT -> DPO -> final on a final-model-free screen, on the probability scale and per unit of
output scale.

    python3 papers/kdg_panel/scripts/analyze_kdg_a8.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

P1A = A.OUT / "p1a"
NAMES = ("dl_chat_neutral", "jl_chat_neutral", "dl_chat_neutral_pressure_removed",
         "jl_chat_neutral_pressure_removed")
DIRS = {"sft": [P1A / "stages_chat_sft" / "olmo3_sft"], "dpo": [P1A / "stages_chat" / "olmo3_dpo"],
        "final": [P1A / "final_c1" / "olmo3_instruct"]}
STEPS = (("sft", "dpo", "DPO"), ("dpo", "final", "RL"))


def lean(r: dict) -> dict:
    a, j = A.logit(r["pDn"]), A.logit(r["pJn"])
    sig = 0.5 * (r["sD"] + r["sJ"])
    return {"g_null": r["pDn"] - r["pJn"], "lam": float((a - j) / sig),
            "lam_fs": float(a / r["sD"] - j / r["sJ"]), "E": A.scales(r)["E_prob"]}


def main() -> int:
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen
              if not s.covariates.get("construction_flag") and not s.id.endswith("S")}
    C = {st: A.four_cells(d, NAMES, status) for st, d in DIRS.items()}
    ids = sorted(s for s in status if all(s in C[st] and C[st][s]["mass_min"] >= A.FLOOR for st in C))
    L = {st: [lean(C[st][s]) for s in ids] for st in C}
    rep = {"amendment": "P1-A7 (8d8b44d)", "n": len(ids), "n_boot": A.N_BOOT, "seed": A.SEED,
           "per_stage": {}, "steps": {}}
    for q in ("g_null", "lam", "lam_fs", "E"):
        rep["per_stage"][q] = {st: A.boot(np.array([x[q] for x in L[st]])) for st in C}
        rep["steps"][q] = {n: A.boot(np.array([y[q] - x[q] for x, y in zip(L[a], L[b])]))
                           for a, b, n in STEPS}
    pos = [n for n in ("DPO", "RL") if A.sign_of(rep["steps"]["g_null"][n]["ci95"]) > 0]
    if not pos:
        v = "selection_R_b"
    else:
        scaled = [n for n in pos if A.sign_of(rep["steps"]["lam"][n]["ci95"]) > 0]
        v = "survives_selection_and_scale" if scaled else "survives_selection_sharpening_explained"
    rep["verdict"] = v
    rep["positive_steps"] = pos
    (A.DATA / "analysis_kdg_a8.json").write_text(json.dumps(rep, indent=1))
    f = lambda x: "%.3f [%.3f, %.3f]" % (x["mean"], *x["ci95"])  # noqa: E731
    print("n", len(ids), "verdict", v)
    for q in ("g_null", "lam", "lam_fs", "E"):
        print(q, {st: f(x) for st, x in rep["per_stage"][q].items()},
              {n: f(x) for n, x in rep["steps"][q].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
