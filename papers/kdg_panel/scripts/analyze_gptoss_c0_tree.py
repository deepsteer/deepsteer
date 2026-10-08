#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A10: C0 failure diagnosis tree (pushed 7b7ce85 before this ran).

    python3 papers/kdg_panel/scripts/analyze_gptoss_c0_tree.py [--dir <p2c cell dir>]

Root Δκ = A_fr − A_rr per scenario (forced argmax vs each rollout; rollout vs rollout pairs), paired
bootstrap over scenarios; then, if the forced readout departs, the direction split D over scenarios
whose strict generated majority differs from the forced argmax.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_gptoss as G  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402
import kdg_harmony as kh  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402


def ci(x: np.ndarray, f=np.mean) -> list[float]:
    rng = np.random.default_rng(A.SEED)
    b = [f(x[rng.integers(0, len(x), len(x))]) for _ in range(A.N_BOOT)]
    return [float(v) for v in np.percentile(b, [2.5, 97.5])]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=G.DEFAULT_DIR)
    ap.add_argument("--out", type=Path, default=A.DATA / "analysis_gptoss_c0_tree.json")
    a = ap.parse_args(argv)
    scen, _ = load_scenario_dir(sorted(A.DATA.glob("*_scenarios_*.json")))
    label = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    F = G.forced_argmax(a.dir, "c0_forced_primary")
    roll: dict[str, list] = defaultdict(list)
    for r in A._rows(a.dir / "c0_generate_low.jsonl"):
        if r["reasoning_trace"] == "completed":
            roll[r["scenario_id"]].append(r["option_id"])  # None = completed but unparsed
    ids = sorted(s for s in roll if s in F and len(roll[s]) >= 2)
    a_fr = np.array([np.mean([o == F[s] for o in roll[s]]) for s in ids])
    a_rr = np.array([np.mean([x == y and x is not None for x, y in combinations(roll[s], 2)])
                     for s in ids])
    dk = a_fr - a_rr
    root = {"n": len(ids), "A_fr": float(a_fr.mean()), "A_rr": float(a_rr.mean()),
            "dkappa": float(dk.mean()), "dkappa_ci95": ci(dk)}
    lo, hi = root["dkappa_ci95"]
    branch = "ceiling_limited" if hi >= 0 else "departs"

    # second derivation: a readout equal to each scenario's modal answer, scored like C0
    modal = []
    for s in ids:
        c = defaultdict(int)
        for o in roll[s]:
            if o is not None:
                c[o] += 1
        m = max(c, key=c.get) if c else None
        modal.append(kh.strict_majority(roll[s]) == m and m is not None)
    second = {"modal_readout_agreement": float(np.mean(modal)),
              "forced_readout_agreement": float(np.mean(
                  [kh.strict_majority(roll[s]) == F[s] for s in ids]))}

    dis = []
    for s in ids:
        g = kh.strict_majority(roll[s])
        if g is not None and g != F[s]:
            fs, gs = label[s].get(F[s]), label[s].get(g)
            dis.append(1 if (fs, gs) == ("violating", "consistent")
                       else -1 if (fs, gs) == ("consistent", "violating") else 0)
    dis = np.array(dis, float)
    direction = {"n_disagreeing": len(dis),
                 "forced_violating_gen_consistent": int((dis == 1).sum()),
                 "forced_consistent_gen_violating": int((dis == -1).sum()),
                 "other_pairs": int((dis == 0).sum()),
                 "D": float(dis.mean()) if len(dis) else None,
                 "D_ci95": ci(dis) if len(dis) > 1 else None}
    if branch == "departs" and direction["D_ci95"]:
        dlo, dhi = direction["D_ci95"]
        branch = "deliberation_brake" if dlo > 0 else "reverse" if dhi < 0 else "undirected"
    rep = {"rule": "KDG_GPTOSS_SPEC G-A10 (7b7ce85)", "root": root, "second_derivation": second,
           "direction": direction, "branch": branch}
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
