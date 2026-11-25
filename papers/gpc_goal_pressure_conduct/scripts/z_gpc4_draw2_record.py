#!/usr/bin/env python3
"""Draw 2's record (GPC_PLAN.md §18.7; the lead's 2026-11-25 7:40 AM item 1), zero GPU, counts only.

Beside the pod's registered verdicts (`depth_read.json`), from the same judged runs at 3:
1. *Per scenario:* each arm's count of runs judged a violation, its n, the clause's reduction with its interval,
   and the ratio to the clause for C1 and for the summed adapter, each with its interval. A scenario whose clause
   reduction has an interval including 0 carries no contrast, and its ratios are not read.
2. *The summed adapter's rate minus the clause's,* pooled over the three scenarios, with its interval, and the
   summed adapter's two-sided ratio interval with P(ratio >= 0.5).

Bootstraps: 10,000, within scenario and arm (the registered estimator's strata).

    python z_gpc4_draw2_record.py [--draw2 outputs/gpc4_v1_d2]   # writes <draw2>/draw2_record.json
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
import z_gpc4_draw2_read as D2  # noqa: E402

B = 10000
SEED = "20261125|gpc4|draw2|record"


def _rate(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _cells(rows: list[dict], scen: list[str]) -> dict[tuple, list[float]]:
    c: dict[tuple, list[float]] = {}
    for r in rows:
        if r["scenario"] in scen:
            c.setdefault((r["scenario"], r["arm"]), []).append(r["y"])
    return c


def _pooled(c: dict, arm: str) -> float | None:
    return _rate([x for k, xs in c.items() if k[1] == arm for x in xs])


def _ci(xs: list[float]) -> list[float]:
    xs = sorted(xs)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]]


def read(rows: list[dict], scen: list[str], seed: str) -> dict:
    """Counts, the clause's reduction, C1's and the summed adapter's ratios, and summed minus clause."""
    c = _cells(rows, scen)
    pt = {a: _pooled(c, a) for a in D2.ARMS}
    rec = {"counts": {a: int(sum(x for k, xs in c.items() if k[1] == a for x in xs)) for a in D2.ARMS},
           "n": {a: sum(len(xs) for k, xs in c.items() if k[1] == a) for a in D2.ARMS}, "rates": pt}
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        bs = {k: [v[rng.randrange(len(v))] for _ in v] for k, v in c.items()}
        draws.append({a: _pooled(bs, a) for a in D2.ARMS})
    red = pt["base"] - pt["cctx"]
    rec["clause_reduction"] = red
    rec["clause_reduction_ci"] = _ci([q["base"] - q["cctx"] for q in draws])
    rec["contrast"] = rec["clause_reduction_ci"][0] > 0
    for a in ("c1", "sum"):
        rs = [(q["base"] - q[a]) / (q["base"] - q["cctx"]) if q["base"] - q["cctx"] > 0 else float("-inf")
              for q in draws]
        rec[f"{a}_ratio"] = {"point": (pt["base"] - pt[a]) / red if red > 0 else None, "ci95": _ci(rs),
                             "p_ge_0.5": sum(r >= 0.5 for r in rs) / B,
                             "read": rec["contrast"]}
    rec["sum_minus_cctx"] = {"point": pt["sum"] - pt["cctx"], "ci95": _ci([q["sum"] - q["cctx"] for q in draws])}
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw2", type=Path, default=HERE.parent / "outputs/gpc4_v1_d2")
    a = ap.parse_args()
    rows = D2.judged_rows(a.draw2)
    prim = list(P4.DEPTH_PRIMARY)
    rec = {"rule": "GPC_PLAN.md §18.7 (the lead's 2026-11-25 7:40 AM item 1); judge at 3; "
                   f"bootstrap {B}, within scenario and arm",
           "primary": read(rows, prim, f"{SEED}|primary"),
           "per_scenario": {s: read(rows, [s], f"{SEED}|{s}") for s in prim}}
    (a.draw2 / "draw2_record.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
