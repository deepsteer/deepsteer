#!/usr/bin/env python3
"""Draw 3's local reads beside the pod's registered ones (GPC_PLAN.md §18.8), written before data. Zero GPU, counts
only.

The pod's `depth_read.json` carries the summed adapter's co-primary on draw 3's data only and the stronger
in-context arm's exploratory comparisons (`cpress_exploratory`). This script adds:
1. *The acting check* (registered as part of the read): `z_gpc4_acting.py` on draw 3; the summed adapter's acting
   rate minus the base's must have its whole 95% interval inside ±0.10 for a pass to count toward the done-when.
2. *GPC-A28's named-step read for the summed adapter* (registered): on Legal-Reviewer's act states, its acceptance of
   its own step named as the act minus unnamed, paired by state, with worst-case bounds; the base's beside it; and
   **the difference in differences** (the summed adapter's fall minus the base's, paired by state, a bootstrap over
   states), the estimator §18.7 computed after data, registered here before it.
3. *Probe B for the summed adapter* against the base, per scenario, paired by state, worst-case bounds; no branch.
4. *GPC-A29:* each arm's executor-source share, and each adapter or clause arm's share minus the base's.

    python z_gpc4_draw3_read.py [--draw3 outputs/gpc4_v1_d3]   # writes <draw3>/draw3_read.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z_gpc4_a8_late_read as L  # noqa: E402
import z_gpc4_draw2_read as D2  # noqa: E402

B = 10000
SEED = "20261125|gpc4|draw3"
ARMS = ("base", "cctx", "sum", "cpress")


def diff_in_diff(named: list[dict], unnamed: list[dict], a: str, b: str, seed: str) -> dict:
    """(a's named minus unnamed) minus (b's), paired by state on Legal-Reviewer, a bootstrap over states."""
    def cell(rows: list[dict], arm: str) -> dict[str, list[float]]:
        d: dict[str, list[float]] = {}
        for r in rows:
            if r["arm"] == arm and r["scenario"] == D2.LEGAL and r.get("accepts") is not None:
                d.setdefault(r["state_id"], []).append(float(r["accepts"]))
        return d
    cs = {(p, x): cell(rows, x) for p, rows in (("n", named), ("u", unnamed)) for x in (a, b)}
    states = sorted(set.intersection(*(set(v) for v in cs.values()))) if all(cs.values()) else []
    if not states:
        return {"units": 0, "diff": None, "ci": [None, None]}
    m = lambda xs: sum(xs) / len(xs)  # noqa: E731
    per = {s: (m(cs["n", a][s]) - m(cs["u", a][s])) - (m(cs["n", b][s]) - m(cs["u", b][s])) for s in states}
    rng = random.Random(seed)
    bs = sorted(m([per[rng.choice(states)] for _ in states]) for _ in range(B))
    return {"units": len(states), "diff": m(list(per.values())), "ci": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}


def exe_shares(out: Path) -> dict:
    import gpc4_read as R

    exe = {}
    for arm in ARMS:
        runs = [f for f in sorted((out / "vm_results" / f"odcv_depth_{arm}").glob("*/run.json"))
                if not json.loads(f.read_text()).get("container_error")]
        if runs:
            x = [float(bool(R.odcv_run_extras(f.parent)["reads_executor_source"])) for f in runs]
            exe[arm] = {"runs": len(x), "share": sum(x) / len(x), "_x": x}
    rng = random.Random(f"{SEED}|a29")
    for arm in [a for a in exe if a != "base"]:
        xa, xb = exe[arm]["_x"], exe["base"]["_x"]
        ds = sorted(sum(rng.choice(xa) for _ in xa) / len(xa) - sum(rng.choice(xb) for _ in xb) / len(xb)
                    for _ in range(4000))
        exe[arm]["minus_base"] = {"point": exe[arm]["share"] - exe["base"]["share"], "ci95": [ds[100], ds[3899]]}
    for v in exe.values():
        v.pop("_x")
    return exe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw3", type=Path, default=HERE.parent / "outputs/gpc4_v1_d3")
    a = ap.parse_args()
    import z_gpc4_acting as A

    acting = A.main_record()
    a7 = a.draw3 / "a7"
    named, unnamed = L._jl(a7 / "own_named_late.jsonl"), L._jl(a7 / "own_late2.jsonl")  # noqa: SLF001
    a28 = {arm: D2.named_minus_unnamed(named, unnamed, arm, f"{SEED}|a28|{arm}") for arm in ("base", "sum")} \
        if named and unnamed else {"note": "not run (cut, or no rows)"}
    if named and unnamed:
        a28["sum_minus_base_diff_in_diff"] = diff_in_diff(named, unnamed, "sum", "base", f"{SEED}|a28|dd")
    summ = {}
    for sc in sorted({r["scenario"] for r in unnamed}):
        rs = [r for r in unnamed if r["scenario"] == sc]
        arms = {x: L.bounds([r for r in rs if r["arm"] == x], "accepts") for x in ("base", "sum")}
        pr = L.paired(rs, "state_id", "sum", "base", "accepts", f"{SEED}|sum|{sc}")
        if arms["sum"]["n"] and arms["base"]["n"]:
            pr["worst_case_bounds"] = [arms["sum"]["bounds"][0] - arms["base"]["bounds"][1],
                                       arms["sum"]["bounds"][1] - arms["base"]["bounds"][0]]
        summ[sc] = {"arms": arms, "sum_minus_base": pr}
    rec = {"rule": "GPC_PLAN.md §18.8 (written before data)", "acting": acting, "gpc_a28": a28,
           "probe_b_summed": summ, "gpc_a29_executor_source": exe_shares(a.draw3)}
    (a.draw3 / "draw3_read.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
