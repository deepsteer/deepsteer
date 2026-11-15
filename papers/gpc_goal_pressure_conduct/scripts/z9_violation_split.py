#!/usr/bin/env python3
"""GPC v0.2 item 9: GOOD_BEHAVIOR's effect on the S-hi pool split by the violation on offer.

    /opt/homebrew/bin/python3.13 z9_violation_split.py

A state's violation on offer is the union of MACHIAVELLI's labels on its violating options.
Violence: killing, physical_harm, intending_harm. Non-violent: deception, manipulation, promises,
fairness, stealing, trespassing, spying, vandalism. Bins: non-violent only, violence only, both,
neither (non_physical_harm or other alone). Shares of GOOD's effect (GOOD minus base of record, k 2)
and of states, on the 443-state pool and on the drawn 250; then the per-category weights (each
state's effect split evenly over the categories on offer). The screen's item 12 split uses
``violence_involved`` (violence only or both) against ``non_violent_only``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
exec(open(HERE / "z1_concentration.py").read().split("# Z1")[0])  # noqa: S102

from gpc_violation import NON_VIOLENT, VIOLENCE, bin_of, on_offer  # noqa: E402,F401


def split(states: list[str]) -> dict:
    eff = {s: np.mean(good[s]) - np.mean(base[s]) for s in states}  # noqa: F821
    tot = sum(eff.values())
    g = defaultdict(list)
    for s in states:
        g[bin_of(on_offer(lab[s]))].append(s)  # noqa: F821
    return {k: {"states": len(v) / len(states), "effect": float(sum(eff[x] for x in v) / tot)}
            for k, v in sorted(g.items())}


if __name__ == "__main__":
    pool = [s for s in ids if sum(pctx[s]) == 2 and lab[s]["saved_violating"] == 1]  # noqa: F821
    drawn = json.loads((HERE.parent / "data/gpc_screen_sets_v1.json").read_text())["S_hi"]
    eff = {s: np.mean(good[s]) - np.mean(base[s]) for s in pool}  # noqa: F821
    tot = sum(eff.values())
    w = defaultdict(float)
    for s in pool:
        c = on_offer(lab[s])  # noqa: F821
        for k in c:
            w[k] += eff[s] / tot / len(c)
    res = {"rule": "GPC v0.2 item 9", "pool": split(pool), "drawn_250": split(drawn),
           "category_weights": dict(sorted(w.items(), key=lambda kv: -kv[1]))}
    for k in ("pool", "drawn_250"):
        r = res[k]
        r["violence_involved"] = {m: r.get("both", {}).get(m, 0) + r.get("violence_only", {}).get(
            m, 0) for m in ("states", "effect")}
    print(json.dumps(res, indent=1))
