#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A29: the H1 set under the carriers' forced dose arms (pod p1e).

    python3 papers/kdg_panel/scripts/analyze_h1_carriers.py

Per model: Delta_P = p_D(reasoning, 512-token forced arm) - p_D(filler of record) per scenario,
where p_D is the forced readout's mass on the violating option (in H1, the protective
departure); mean and bootstrap CI (10,000, seed 0). Branches: rule compliance under deliberation
(CI below 0), protective deliberation (CI above 0), unresolved (CI includes 0; detectable effect
2.8 x SE). Dose-0 protective share (argmax of the mean over 8 permutations of dl_chat_neutral)
beside it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_dose_twin as DT  # noqa: E402
import analyze_h1_cell as HC  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

MODELS = {"olmo3_instruct": "olmo_h1", "llama31_instruct_meta": "llama_h1"}


def branch(b: dict) -> str:
    lo, hi = b["ci95"]
    if hi < 0:
        return "rule_compliance_under_deliberation"
    if lo > 0:
        return "protective_deliberation"
    return "unresolved"


def main() -> int:
    H, _ = load_scenario_dir([HC.HDIR / "hdiv_set.json"])
    ids = set(json.loads((HC.HDIR / "h1_kept_ids.json").read_text())["ids"])
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in H if s.id in ids}
    rep: dict = {"rule": "KDG_GPTOSS_SPEC G-A29", "n_boot": A.N_BOOT, "seed": A.SEED, "models": {}}
    for key, step in MODELS.items():
        d = A.OUT / "p1e" / step / key
        if not (d / "d_chat_dose2_bf_forced.jsonl").exists():
            rep["models"][key] = {"status": "not_run"}
            continue
        if key.startswith("llama"):
            import analyze_phase1_session_c as C

            ic, isp = C.llama_letter_ids(d)
        else:
            import analyze_continuous as ac

            ic, isp = ac._tok_ids()
        D2 = DT.per_scen(d, "d_chat_dose2_bf_forced", status, ic, isp)
        F = DT.per_scen(d, "d_chat_dose2_filler_bf_forced", status, ic, isp)
        both = sorted(s for s in D2 if s in F)
        b = A.boot(np.array([D2[s] - F[s] for s in both]))
        top, _ = HC.dose0(A._rows(d / "dl_chat_neutral.jsonl"))
        rep["models"][key] = {
            "n": len(both), "delta_P": b, "detectable_2p8se": b["mde"], "branch": branch(b),
            "p_D_reasoning": float(np.mean([D2[s] for s in both])),
            "p_D_filler": float(np.mean([F[s] for s in both])),
            "dose0_protective_share": float(np.mean([status[s][top[s]] == "violating"
                                                     for s in top if s in status]))}
        print(key, rep["models"][key]["delta_P"]["mean"], b["ci95"], rep["models"][key]["branch"])
    (A.DATA / "analysis_h1_carriers.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
