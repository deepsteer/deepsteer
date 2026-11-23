#!/usr/bin/env python3
"""Amendment 6's root, exploratory and post-bail (GPC_PLAN.md §14.4; the lead's 2026-11-23 12:20 PM item 2): the
picks by scenario and arm, Education-Assistant's clause and C1 differences paired by render, the named-choice rate
against the depth cell's base by scenario and variant (the judgment-action gap), the picks by render, and the
pressure factor paired by render. Zero GPU, from the saved rows. Labeled exploratory: the bail is the record.

    python z_gpc4_a6_explore.py   # writes outputs/gpc4_a6/a6/explore.json
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_pod as P4  # noqa: E402

GPC = HERE.parent
A6 = GPC / "outputs/gpc4_a6/a6"
D1 = GPC / "outputs/gpc4_v1_depth1"
B = 10000


def ci(xs):
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def main() -> None:
    rows = [r for r in P4.P1._jl(A6 / "root.jsonl") if r.get("choice") is not None]  # noqa: SLF001
    picks = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        k = (r["scenario"], r["arm"])
        picks[k][0] += r["choice"] == "goal"
        picks[k][1] += 1
    rec = {"label": "exploratory, post-bail (the bail is the record)",
           "picks": {f"{s}|{a}": v for (s, a), v in sorted(picks.items())}}
    # Education-Assistant: by render (pressure pooled), paired differences, a bootstrap over renders within variant
    ed = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["scenario"] == "Ai-Education-Assistant":
            ed[r["arm"]][r["render_id"]].append(float(r["choice"] == "goal"))
    rend = sorted(ed["base"])
    strata = collections.defaultdict(list)
    for k in rend:
        strata[k.rsplit("-", 1)[1][0]].append(k)
    rng = random.Random(20261123)
    m = lambda arm, ks: sum(sum(ed[arm][k]) / len(ed[arm][k]) for k in ks) / len(ks)
    for a in ("cctx", "c1", "sum"):
        pt = m(a, rend) - m("base", rend)
        bs = []
        for _ in range(B):
            ks = [rng.choice(v) for v in strata.values() for _ in v]
            bs.append(m(a, ks) - m("base", ks))
        rec[f"education_{a}_minus_base"] = {"diff": pt, "ci": ci(bs), "renders": len(rend)}
    # picks by render: how many renders carry the base's picks
    base_by_render = collections.Counter(r["render_id"] for r in rows if r["arm"] == "base" and r["choice"] == "goal")
    rec["base_picks_by_render"] = {"total": sum(base_by_render.values()), "renders_with_any": len(base_by_render),
                                   "by_render": dict(sorted(base_by_render.items()))}
    # the named-choice rate against the depth cell's base by scenario and variant (the judgment-action gap)
    judged = {r["run_id"]: r for r in P4.P1._jl(D1 / "odcv/judged_depth_a.jsonl")}  # noqa: SLF001
    depth = collections.defaultdict(list)
    for f in sorted((D1 / "vm_results/odcv_depth_base").glob("*/run.json")):
        r = json.loads(f.read_text())
        j = judged.get(r["run_id"])
        if j and j.get("score") is not None:
            depth[(r["scenario"], r["variant"])].append(float(j["score"] >= 3))
    named = collections.defaultdict(list)
    for r in rows:
        if r["arm"] == "base":
            named[(r["scenario"], r["variant"])].append(float(r["choice"] == "goal"))
    rec["judgment_action_gap"] = {f"{s}|{v}": {"named_choice": sum(named[(s, v)]) / len(named[(s, v)]),
                                               "loop_judged": sum(depth[(s, v)]) / len(depth[(s, v)]),
                                               "gap": sum(depth[(s, v)]) / len(depth[(s, v)]) - sum(named[(s, v)]) / len(named[(s, v)])}
                                  for (s, v) in sorted(depth)}
    # the pressure factor: pressured minus unpressured, paired by render, per arm (no direction claim)
    pf = {}
    for a in ("base", "cctx", "c1", "sum"):
        by = collections.defaultdict(lambda: {True: [], False: []})
        for r in rows:
            if r["arm"] == a:
                by[r["render_id"]][r["pressure"]].append(float(r["choice"] == "goal"))
        ks = sorted(k for k, v in by.items() if v[True] and v[False])
        st = collections.defaultdict(list)
        for k in ks:
            st[k.rsplit("-", 1)[0] + k.rsplit("-", 1)[1][0]].append(k)
        d = lambda kk: sum(sum(by[k][True]) / len(by[k][True]) - sum(by[k][False]) / len(by[k][False]) for k in kk) / len(kk)
        bs = []
        for _ in range(B):
            kk = [rng.choice(v) for v in st.values() for _ in v]
            bs.append(d(kk))
        pf[a] = {"pressured_minus_unpressured": d(ks), "ci": ci(bs), "renders": len(ks)}
    rec["pressure_factor"] = pf
    (A6 / "explore.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
