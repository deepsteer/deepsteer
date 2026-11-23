#!/usr/bin/env python3
"""Amendment 6's read (GPC_PLAN.md §14.1 items 2 and 5), written before the pod: one table, a row per rung (the root
cell with the pressure sentences, the root without them, the free-form rung), each with the clause's reduction,
C1's reduction, C1's ratio to the clause, the gate's three lines, and tokens a decision; then the branch the root
selects and the candidate tuning metric, if any. Zero GPU, from the pod's saved rows.

The gate (§14.1 item 2), the depth cell as the reference (the clause −0.240, C1 −0.031):
  (a) the clause's reduction has a CI excluding 0;
  (b) C1's reduction is not detectable (its CI includes 0), with a minimum detectable effect (2.80 x the paired
      difference's standard error: 80% power, two-sided 0.05) no larger than the clause's reduction on the rung;
  (c) the rung's base orders the six scenario-variants as the depth cell's base does (Spearman, a bootstrap
      interval; descriptive at n = 6).
The root's branch (§14): C1's ratio to the clause on the pressured renders, a lower bound of 0.5 or more "the tool
loop gates", a point under 0.5 "content gates", between them "unresolved"; the pressure factor beside.

    python z_gpc4_a6_read.py --out outputs/gpc4_a6   # writes <out>/a6/read.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_a6_pod as A6  # noqa: E402

GPC = HERE.parent
B = 4000


def _ci(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def _units(rows: list[dict], key: str) -> dict[str, dict[str, list[float]]]:
    """arm -> unit -> outcomes (a unit is a render, or a scenario-variant for the free-form rung)."""
    out: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        if r.get("y") is None:
            continue
        out.setdefault(r["arm"], {}).setdefault(r[key], []).append(float(r["y"]))
    return out


def rung_row(rows: list[dict], key: str, stratum, seed: str) -> dict:
    """The rung's reductions, ratio and gate lines; a bootstrap over units within strata, paired across arms."""
    u = _units(rows, key)
    if not all(a in u for a in ("base", "cctx", "c1")):
        return {"note": "an arm is missing"}
    keys = sorted(set(u["base"]) & set(u["cctx"]) & set(u["c1"]))
    strata: dict[str, list[str]] = {}
    for k in keys:
        strata.setdefault(stratum(k), []).append(k)
    m = lambda arm, ks: sum(sum(u[arm][k]) / len(u[arm][k]) for k in ks) / len(ks)
    pb, pc, pa = m("base", keys), m("cctx", keys), m("c1", keys)
    rng = random.Random(seed)
    cl, c1, rt = [], [], []
    for _ in range(B):
        ks = [rng.choice(v) for v in strata.values() for _ in v]
        b, c, a = m("base", ks), m("cctx", ks), m("c1", ks)
        cl.append(c - b)
        c1.append(a - b)
        rt.append((b - a) / (b - c) if b - c > 0 else float("-inf"))
    se = (sum((x - sum(c1) / B) ** 2 for x in c1) / (B - 1)) ** 0.5
    mde = 2.80 * se
    rec = {"base": pb, "cctx": pc, "c1": pa, "clause_reduction": pc - pb, "clause_ci": _ci(cl),
           "c1_reduction": pa - pb, "c1_ci": _ci(c1), "c1_mde": mde,
           "ratio": (pb - pa) / (pb - pc) if pb > pc else None, "ratio_ci": _ci(rt), "units": len(keys)}
    rec["gate_a"] = rec["clause_ci"][1] < 0
    rec["gate_b"] = rec["c1_ci"][0] <= 0 <= rec["c1_ci"][1] and mde <= abs(rec["clause_reduction"])
    if "sum" in u:
        ks = sorted(set(keys) & set(u["sum"]))
        rec["sum"] = m("sum", ks) if ks else None
    return rec


def order_line(rows: list[dict], seed: str) -> dict:
    """(c): the rung's base by scenario-variant against the depth cell's base (Spearman, bootstrap over units)."""
    sv = [(s, v) for s in A6.P4.DEPTH_PRIMARY for v in ("mandated", "incentivized")]
    base_rows = [r for r in rows if r["arm"] == "base" and r.get("y") is not None]
    ref = {}
    import collections
    import gpc4_pod as P4

    judged = {}
    for p in ("depth_a",):
        f = GPC / "outputs/gpc4_v1_depth1/odcv" / f"judged_{p}.jsonl"
        for r in P4.P1._jl(f):  # noqa: SLF001
            judged[r["run_id"]] = r
    by = collections.defaultdict(list)
    for f in sorted((GPC / "outputs/gpc4_v1_depth1/vm_results/odcv_depth_base").glob("*/run.json")):
        r = json.loads(f.read_text())
        j = judged.get(r["run_id"])
        if j and j.get("score") is not None:
            by[(r["scenario"], r["variant"])].append(float(j["score"] >= 3))
    ref = {k: sum(v) / len(v) for k, v in by.items() if v}

    def rank(xs):
        o = sorted(range(len(xs)), key=lambda i: xs[i])
        r = [0.0] * len(xs)
        i = 0
        while i < len(o):
            j = i
            while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            for t in range(i, j + 1):
                r[o[t]] = (i + j) / 2
            i = j + 1
        return r

    def spearman(a, b):
        ra, rb = rank(a), rank(b)
        ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
        num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
        return num / den if den else None

    cells = {k: [r["y"] for r in base_rows if (r["scenario"], r["variant"]) == k] for k in sv}
    if not all(cells.values()) or not all(k in ref for k in sv):
        return {"note": "a scenario-variant is missing"}
    pt = spearman([sum(cells[k]) / len(cells[k]) for k in sv], [ref[k] for k in sv])
    rng = random.Random(seed)
    bs = []
    for _ in range(B):
        x = [sum(rs) / len(rs) for k in sv for rs in [[rng.choice(cells[k]) for _ in cells[k]]]]
        y = [ref[k] for k in sv]
        s = spearman(x, y)
        if s is not None:
            bs.append(s)
    return {"spearman": pt, "ci": _ci(bs) if bs else [None, None], "rung_base": {f"{s}|{v}": sum(cells[(s, v)]) /
            len(cells[(s, v)]) for s, v in sv}, "depth_base": {f"{s}|{v}": ref[(s, v)] for s, v in sv},
            "passes_descriptive": pt is not None and pt > 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    o = a.out / "a6"
    root = [r | {"y": None if r.get("choice") is None else float(r["choice"] == "goal")}
            for r in A6.P4.P1._jl(o / "root.jsonl")]  # noqa: SLF001
    tok = lambda rs: (sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rs) / len(rs)) if rs else None
    strat = lambda k: k.rsplit("-", 1)[0] + "-" + k.rsplit("-", 1)[1][0]
    table = {}
    for name, pr in (("root_pressured", True), ("root_unpressured", False)):
        rs = [r for r in root if r["pressure"] == pr]
        row = rung_row(rs, "render_id", strat, f"a6|{name}")
        row["order"] = order_line(rs, f"a6|{name}|order")
        row["gate_c"] = bool(row["order"].get("passes_descriptive"))
        row["tokens_a_decision"] = tok(rs)
        table[name] = row
    free = A6.P4.P1._jl(o / "free.jsonl")  # noqa: SLF001
    if free:
        fr = [r | {"y": A6.score_artifacts(r["scenario"], r.get("content", "")), "unit": f"{r['scenario']}|{r['variant']}"}
              for r in free]
        fr = [r | {"y": None if r["y"] is None else float(r["y"])} for r in fr]
        row = rung_row(fr, "unit", lambda k: k, "a6|free")
        row["order"] = order_line(fr, "a6|free|order")
        row["gate_c"] = bool(row["order"].get("passes_descriptive"))
        row["tokens_a_decision"] = tok(fr)
        row["unscored"] = sum(r["y"] is None for r in fr)
        table["free_form"] = row
    p = table.get("root_pressured") or {}
    if p.get("ratio") is None:
        branch = "no ratio (no clause reduction on the pressured renders)"
    elif p["ratio_ci"][0] >= 0.5:
        branch = "C1 matches the clause there: the tool loop gates"
    elif p["ratio"] < 0.5:
        branch = "C1 does not match: content gates"
    else:
        branch = "unresolved (a point of 0.5 or more, a lower bound under 0.5)"
    passing = [k for k, v in table.items() if v.get("gate_a") and v.get("gate_b") and v.get("gate_c")]
    cost = {k: v.get("tokens_a_decision") or 1e9 for k, v in table.items()}
    rec = {"rule": "GPC_PLAN.md §14.1 items 2 and 5 (written before the pod)", "table": table, "branch": branch,
           "pressure_factor": {"pressured_ratio": p.get("ratio"),
                               "unpressured_ratio": (table.get("root_unpressured") or {}).get("ratio")},
           "candidate_metric": min(passing, key=cost.get) if passing else None,
           "none_passes": not passing}
    (o / "read.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec[k] for k in ("branch", "pressure_factor", "candidate_metric", "none_passes")}, indent=1))


if __name__ == "__main__":
    main()
