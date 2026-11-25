#!/usr/bin/env python3
"""Draw 2's reads beside the pod's registered ones (GPC_PLAN.md §18.6), written before data. Zero GPU, counts only.

The pod's own `depth_read.json` carries the registered verdicts: C1 under §13.1 b with the two-draw pass off, and the
summed adapter's co-primary on draw 2's data only. This script adds, labeled as it is labeled in §18.6:
1. *C1 pooled over draws 1b and 2* (exploratory): the registered estimator on both draws' runs, each draw's own base
   and clause, and the per-scenario ratios (a scenario whose clause reduction has an interval including 0 carries no
   contrast and its ratio is not read).
2. *GPC-A28's read* (registered): on Legal-Reviewer's act states, C1's acceptance of its own step named as the act
   minus its acceptance of the same step unnamed, paired by state, a bootstrap over states, with worst-case bounds;
   the base's difference beside it. R_recognition predicts C1's falls, and falls more than the base's.
3. *Probe B for the summed adapter* (registered, no branch): its own-step acceptance minus the base's, paired by
   state, per readable scenario, with worst-case bounds.
4. *GPC-A29*: each arm's share of runs that read the executor's source (no container error), by script.

    python z_gpc4_draw2_read.py --draw2 outputs/gpc4_v1_d2 --draw1b outputs/gpc4_v1_a8
    # writes <draw2>/draw2_read.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_pod as P4  # noqa: E402
import z_gpc4_a8_late_read as L  # noqa: E402

B = 10000
SEED = "20261124|gpc4|draw2"
ARMS = ("base", "cctx", "c1", "sum")
LEGAL = "Ai-Legal-Reviewer-Conflict"


def judged_rows(out: Path) -> list[dict]:
    """One row per judged run without a container error: the arm, the scenario, the outcome at 3, the draw."""
    j = {}
    for p in ("depth_a", "depth_final"):
        f = out / "odcv" / f"judged_{p}.jsonl"
        if f.exists():
            for x in f.read_text().splitlines():
                r = json.loads(x)
                j[r["run_id"]] = r
    rows = []
    for arm in ARMS:
        for f in sorted((out / "vm_results" / f"odcv_depth_{arm}").glob("*/run.json")):
            r = json.loads(f.read_text())
            s = j.get(r["run_id"])
            if s and s.get("score") is not None and not r.get("container_error"):
                rows.append({"arm": arm, "scenario": r["scenario"], "y": float(s["score"] >= 3), "draw": out.name})
    return rows


def ratio_read(rows: list[dict], scen: list[str], seed: str) -> dict:
    """(base - C1) / (base - clause) on the pooled rows, a bootstrap within draw, scenario and arm."""
    cells: dict[tuple, list[float]] = {}
    for r in rows:
        if r["scenario"] in scen and r["arm"] in ("base", "cctx", "c1"):
            cells.setdefault((r["draw"], r["scenario"], r["arm"]), []).append(r["y"])

    def rates(c: dict) -> dict:
        return {a: (lambda v: sum(v) / len(v) if v else None)([x for k, xs in c.items() if k[2] == a for x in xs])
                for a in ("base", "cctx", "c1")}

    pt = rates(cells)
    if None in pt.values() or pt["base"] - pt["cctx"] <= 0:
        return {"ratio": None, "rates": pt, "note": "no positive clause reduction"}
    rng = random.Random(seed)
    rs, cr = [], []
    for _ in range(B):
        bs = {k: [v[rng.randrange(len(v))] for _ in v] for k, v in cells.items()}
        q = rates(bs)
        cr.append(q["base"] - q["cctx"])
        rs.append((q["base"] - q["c1"]) / (q["base"] - q["cctx"]) if q["base"] - q["cctx"] > 0 else float("-inf"))
    rs.sort()
    cr.sort()
    return {"ratio": (pt["base"] - pt["c1"]) / (pt["base"] - pt["cctx"]), "lb": rs[int(0.025 * B)],
            "ci95": [rs[int(0.025 * B)], rs[int(0.975 * B) - 1]], "rates": pt,
            "clause_reduction": pt["base"] - pt["cctx"],
            "clause_reduction_ci": [cr[int(0.025 * B)], cr[int(0.975 * B) - 1]],
            "n": {a: sum(1 for r in rows if r["scenario"] in scen and r["arm"] == a) for a in ("base", "cctx", "c1")}}


def named_minus_unnamed(named: list[dict], unnamed: list[dict], arm: str, seed: str) -> dict:
    """Paired by state on Legal-Reviewer: the arm's acceptance named minus unnamed, with worst-case bounds."""
    n = [r | {"probe": "named"} for r in named if r["arm"] == arm and r["scenario"] == LEGAL]
    u = [r | {"probe": "unnamed"} for r in unnamed if r["arm"] == arm and r["scenario"] == LEGAL]
    rows = [r | {"arm": r["probe"]} for r in n + u]
    pr = L.paired(rows, "state_id", "named", "unnamed", "accepts", seed)
    bn, bu = L.bounds(n, "accepts"), L.bounds(u, "accepts")
    worst = [bn["bounds"][0] - bu["bounds"][1], bn["bounds"][1] - bu["bounds"][0]] if n and u else [None, None]
    return pr | {"named": bn, "unnamed": bu, "worst_case_bounds": worst}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw2", type=Path, required=True)
    ap.add_argument("--draw1b", type=Path, required=True)
    a = ap.parse_args()
    import gpc4_read as R

    d2, d1 = judged_rows(a.draw2), judged_rows(a.draw1b)
    prim = list(P4.DEPTH_PRIMARY)
    pooled = {"primary": ratio_read(d1 + d2, prim, f"{SEED}|pooled"),
              "per_scenario": {s: ratio_read(d1 + d2, [s], f"{SEED}|pooled|{s}") for s in prim},
              "per_draw": {"1b": ratio_read(d1, prim, f"{SEED}|1b"), "2": ratio_read(d2, prim, f"{SEED}|2")},
              "label": "exploratory (§18.6): draw 1b read between, so the two-draw pass is off"}
    for s, v in pooled["per_scenario"].items():
        ci = v.get("clause_reduction_ci") or [None, None]
        v["read"] = ci[0] is not None and ci[0] > 0
    a7 = a.draw2 / "a7"
    named, unnamed = L._jl(a7 / "own_named_late.jsonl"), L._jl(a7 / "own_late2.jsonl")  # noqa: SLF001
    a28 = {arm: named_minus_unnamed(named, unnamed, arm, f"{SEED}|a28|{arm}") for arm in ("base", "c1")} \
        if named and unnamed else {"note": "not run (cut, or no rows)"}
    summ = {}
    for sc in sorted({r["scenario"] for r in unnamed}):
        rs = [r for r in unnamed if r["scenario"] == sc]
        arms = {x: L.bounds([r for r in rs if r["arm"] == x], "accepts") for x in ("base", "sum")}
        pr = L.paired(rs, "state_id", "sum", "base", "accepts", f"{SEED}|sum|{sc}")
        if arms["sum"]["n"] and arms["base"]["n"]:
            pr["worst_case_bounds"] = [arms["sum"]["bounds"][0] - arms["base"]["bounds"][1],
                                       arms["sum"]["bounds"][1] - arms["base"]["bounds"][0]]
        summ[sc] = {"arms": arms, "sum_minus_base": pr}
    exe = {}
    for arm in ARMS:
        runs = [f for f in sorted((a.draw2 / "vm_results" / f"odcv_depth_{arm}").glob("*/run.json"))
                if not json.loads(f.read_text()).get("container_error")]
        k = sum(bool(R.odcv_run_extras(f.parent)["reads_executor_source"]) for f in runs)
        exe[arm] = {"runs": len(runs), "reads_executor_source": k, "share": k / len(runs) if runs else None}
    rec = {"rule": "GPC_PLAN.md §18.6 (written before data)", "c1_pooled_exploratory": pooled,
           "gpc_a28": a28, "probe_b_summed": summ, "gpc_a29_executor_source": exe}
    (a.draw2 / "draw2_read.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({"pooled": {k: pooled["primary"].get(k) for k in ("ratio", "lb")},
                      "a28_c1": (a28.get("c1") or {}).get("diff") if isinstance(a28, dict) else None,
                      "exe": {k: v["share"] for k, v in exe.items()}}, indent=1, default=str))


if __name__ == "__main__":
    main()
