#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A24 item 3: the two-pattern scan's counts, positive controls and moves split.

    python3 papers/kdg_panel/scripts/analyze_divergent_patterns.py

Reads ``divergent_pattern_labels.json`` (both-rater assignments of record) and the per-rater files.
Positive controls: the author's own examples (N7 F1-A-06 for P7, N4 F3-A-66 for P4; G-A24 item 1).
Descriptive: G-A22 item 4's moves (dose-0 action of record -> medium-effort action) on the divergent
scenarios with each pattern, under both-rater (of record) and either-rater (sensitivity) assignment.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_gptoss_ds_power as P  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

CONTROLS = {"p7": "F1-A-06", "p4": "F3-A-66"}


def main() -> int:
    lab = json.loads((A.DATA / "divergent_pattern_labels.json").read_text())["patterns"]
    R = {(r, p): json.loads((A.DATA / f"pattern_{p}_ratings_{r}.json").read_text())
         for r in ("subagent", "codex") for p in CONTROLS}
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    status = G._status(scen)
    jopt = json.loads((A.DATA / "gptoss_divergent_175.json").read_text())["dose0_judgment_option"]
    dose0 = {s: max(m, key=m.get) for s, m in P._dose0_means().items() if s in jopt}
    act = {r["scenario_id"]: r["option_id"]
           for r in A._rows(A.OUT / "p2i/main/gpt_oss_20b/ds_main_reconstructed.jsonl")
           if r["stage"] == "main" and r["seed"] == 0 and r.get("token_identity")
           and r.get("option_id")}

    def move(s: str) -> str:
        b, a = dose0.get(s), act.get(s)
        if a is None or b is None:
            return "no_row"
        if a == b:
            return "same"
        if status[s][a] == "consistent":
            return "toward_norm"
        return "toward_dose0_judgment" if a == jopt[s] else "other"

    rep: dict = {"rule": "KDG_GPTOSS_SPEC G-A24 item 3 (605b8e2)", "patterns": {}}
    for p, ctrl in CONTROLS.items():
        both = {s for s in lab[p]["ids"] if s in jopt}
        either = {s for s in jopt if R[("subagent", p)].get(s) or R[("codex", p)].get(s)}
        rep["patterns"][p] = {
            "positive_control": {"id": ctrl, "both_yes": ctrl in set(lab[p]["ids"]),
                                 "by_rater": {r: R[(r, p)].get(ctrl)
                                              for r in ("subagent", "codex")}},
            "both_rater_divergent": len(both), "either_rater_divergent": len(either),
            "moves_both": dict(Counter(move(s) for s in both)),
            "moves_either": dict(Counter(move(s) for s in either)),
            "moves_rest_of_divergent_either": dict(Counter(move(s) for s in set(jopt) - either))}
    (A.DATA / "analysis_divergent_patterns.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
