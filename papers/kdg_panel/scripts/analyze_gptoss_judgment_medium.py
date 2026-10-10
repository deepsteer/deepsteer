#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A23 item 3: judgment at the default reasoning level vs the action that moved.

    python3 papers/kdg_panel/scripts/analyze_gptoss_judgment_medium.py [--jm-rows <p2k jm.jsonl>]

J_med = the option of the letter sampled after GPT-OSS's medium-effort judgment trace (token
identity). Moved set M = divergent scenarios whose medium-effort action moved to the
norm-consistent option (G-A22 item 4's "toward norm" moves on primary 2's rows). In M: k_rev = J_med
norm-consistent, k_hold = J_med equal to the dose-0 judgment's option. Two-sided exact sign test,
alpha 0.01; branches: principled revision (k_rev > k_hold), norm compliance against own judgment
(k_hold > k_rev), unresolved. Descriptive beside it: J_med over all 175; coupling (J_med consistent
in M vs in the divergent scenarios whose action stayed on the dose-0 judgment's option, Fisher);
second derivation against the pilot's stage-A medium judgment traces on the same scenarios.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_gptoss_ds_power as P  # noqa: E402
import analyze_gptoss_risk_reversed as RV  # noqa: E402
import analyze_own_judgment_moves as OJ  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

ALPHA = 0.01


def branch(k_rev: int, k_hold: int) -> dict:
    m = k_rev + k_hold
    p = OJ.binom_two_sided(k_rev, m)
    b = ("unresolved" if p >= ALPHA or k_rev == k_hold else
         "principled_revision" if k_rev > k_hold else "norm_compliance_against_own_judgment")
    return {"k_rev": k_rev, "k_hold": k_hold, "m": m, "p_two_sided": p, "alpha": ALPHA,
            "branch": b}


def classify(jopt: dict[str, str], jmed: dict[str, str | None], status: dict) -> Counter:
    c: Counter = Counter()
    for s, j0 in jopt.items():
        j = jmed.get(s)
        if j is None:
            c["no_identity"] += 1
        elif status[s][j] == "consistent":
            c["consistent"] += 1
        elif j == j0:
            c["dose0_option"] += 1
        else:
            c["other"] += 1
    return c


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jm-rows", type=Path,
                    default=A.OUT / "p2k" / "jm" / "gpt_oss_20b" / "jm.jsonl")
    ap.add_argument("--main-rows", type=Path,
                    default=A.OUT / "p2i" / "main" / "gpt_oss_20b" / "ds_main_reconstructed.jsonl")
    ap.add_argument("--pilot-rows", type=Path,
                    default=A.OUT / "p2h" / "pilot" / "gpt_oss_20b" / "ds_pilot.jsonl")
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss_judgment_medium.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = G._status(scen)
    div = json.loads((A.DATA / "gptoss_divergent_175.json").read_text())
    jopt = div["dose0_judgment_option"]
    dose0 = {s: max(m, key=m.get) for s, m in P._dose0_means().items() if s in jopt}
    act = {r["scenario_id"]: r["option_id"] for r in A._rows(a.main_rows)
           if r["stage"] == "main" and r["cell"] == "dl_chat_neutral" and r["seed"] == 0
           and r.get("token_identity") and r.get("option_id")}
    moved = sorted(s for s in jopt if s in act and s in dose0 and act[s] != dose0[s]
                   and status[s][act[s]] == "consistent" and act[s] != jopt[s])
    stayed = sorted(s for s in jopt if s in act and act[s] == jopt[s])
    jmed = {r["scenario_id"]: (r["option_id"] if r.get("token_identity") else None)
            for r in A._rows(a.jm_rows) if r["cell"] == "jl_chat_neutral"}
    cm = classify({s: jopt[s] for s in moved}, jmed, status)
    cs = classify({s: jopt[s] for s in stayed}, jmed, status)
    n_m, n_s = cm["consistent"] + cm["dose0_option"], cs["consistent"] + cs["dose0_option"]
    pilot = {r["scenario_id"]: r.get("option_id") for r in A._rows(a.pilot_rows)
             if r["cell"] == "jl_chat_neutral" and r["seed"] == 0 and r.get("token_identity")}
    both = sorted(s for s in pilot if s in jopt and jmed.get(s))
    rep = {"rule": "KDG_GPTOSS_SPEC G-A23 item 3 (5ce8c22)", "n_moved": len(moved),
           "n_stayed": len(stayed), "moved": dict(cm), "stayed": dict(cs),
           **branch(cm["consistent"], cm["dose0_option"]),
           "all_175": dict(classify(jopt, jmed, status)),
           "coupling_fisher_descriptive": {
               "table": [[cm["consistent"], cm["dose0_option"]],
                         [cs["consistent"], cs["dose0_option"]]],
               "p_two_sided": (RV.fisher_two_sided(cm["consistent"], cm["dose0_option"],
                                                   cs["consistent"], cs["dose0_option"])
                               if n_m and n_s else None)},
           "second_derivation_pilot": {"n": len(both),
                                       "agree": sum(pilot[s] == jmed[s] for s in both)}}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
