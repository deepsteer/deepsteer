"""GPC Z1 and Z3 inputs: state concentration of GOOD_BEHAVIOR's effect by P-ctx's rollouts (zero GPU).

    /opt/homebrew/bin/python3.13 z1_concentration.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CST = Path.home() / "dev/deepsteer/papers/cst_constitutional_steering/outputs"
LAB = CST / "finalist_c2_rule_right_mach/mach_states_base.jsonl"
T1 = CST / "teacher_v1/teacher/base_good_behavior.jsonl"
A52 = CST / "a52_v1/teacher/a49_pctx.jsonl"
B, SEED = 2000, 20261115


def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


lab = {f"{s['game']}|{s['episode']}|{s['step']}": s for s in rows(LAB)}


def per_state(rs, arm):
    by = defaultdict(list)
    for r in rs:
        if r["arm"] == arm and r["sid"] in lab:
            by[r["sid"]].append(lab[r["sid"]]["violating"][r["action"]])
    return dict(by)


t1, a52 = rows(T1), rows(A52)
base = per_state(t1, "base")
good = per_state(t1, "good_behavior")
pctx = per_state(a52, "p_ctx")
sess = per_state(a52, "base")
ids = sorted(set(base) & set(good) & set(pctx))
print("states", len(ids), "k per arm", {a: set(len(v) for v in d.values()) for a, d in
                                        [("base", base), ("good", good), ("pctx", pctx)]})
m = lambda d, S: float(np.mean([np.mean(d[s]) for s in S]))
eff = {s: np.mean(good[s]) - np.mean(base[s]) for s in ids}
tot = sum(eff.values())
print("base", m(base, ids), "good", m(good, ids), "pctx", m(pctx, ids))

# Z1: strata by P-ctx's two rollouts
for name, cond in [("both", lambda s: sum(pctx[s]) == 2), ("one", lambda s: sum(pctx[s]) == 1),
                   ("neither", lambda s: sum(pctx[s]) == 0)]:
    S = [s for s in ids if cond(s)]
    print(f"Z1 {name}: n {len(S)} base {m(base,S):.3f} good {m(good,S):.3f} "
          f"delta {m(good,S)-m(base,S):+.3f} share_of_effect {sum(eff[s] for s in S)/tot:.3f}")

# Z2: S-hi availability (P-ctx both + saved decision violating), S-lo (none)
hi = [s for s in ids if sum(pctx[s]) == 2 and lab[s]["saved_violating"] == 1]
lo = [s for s in ids if sum(pctx[s]) == 0 and lab[s]["saved_violating"] == 0]
print(f"Z2 S-hi avail {len(hi)} base {m(base,hi):.3f} good {m(good,hi):.3f} "
      f"share_of_effect {sum(eff[s] for s in hi)/tot:.3f}")
print(f"Z2 S-lo avail {len(lo)} base {m(base,lo):.3f} good {m(good,lo):.3f}")
games_hi = defaultdict(list)
for s in hi:
    games_hi[s.split("|")[0]].append(s)
print("S-hi per game counts", sorted(len(v) for v in games_hi.values()), "games", len(games_hi))
games_lo = defaultdict(list)
for s in lo:
    games_lo[s.split("|")[0]].append(s)
print("S-lo per game counts", sorted(len(v) for v in games_lo.values()), "games", len(games_lo))

# Rollout disagreement (two k draws differ)
dis = lambda d, S: float(np.mean([d[s][0] != d[s][1] for s in S]))
print("disagreement base", dis(base, ids), "good", dis(good, ids), "pctx", dis(pctx, ids))
# per-game effect SD (GOOD - base)
ge = defaultdict(list)
for s in ids:
    ge[s.split("|")[0]].append(eff[s])
gm = {g: np.mean(v) for g, v in ge.items()}
print("per-game GOOD effect SD", float(np.std(list(gm.values()), ddof=1)), "mean", np.mean(list(gm.values())))

