#!/usr/bin/env python3
"""§15.3 item 4: the amended probe-artifact checks simulated at the built sizes, through the read's own functions
(`z_gpc4_a7_read.order_check`, `early_check`). Zero GPU; synthetic letters only.

Sizes: each scenario has 16 states (the order check: 2 probes an order a state, the base) and 16 cold renders (the
earliest-state check: 4 probes a state against 4 a render). Within-unit correlation 0.58 at k = 2 (the measured
within-render correlation): each unit draws its rate from a Beta with that intraclass correlation.

Conditions, at brink rates 0.05, 0.25 and 0.50:
- no artifact: the false-drop rate;
- a real effect of 0.30 (the order check: the act's rate in position A minus in position B is 0.30, centred on the
  unit's rate where the bounds allow; the earliest-state check: the earliest probe's rate 0.30 above the cold rate);
- always one letter (the subject answers A whatever the order).

    python z_gpc4_a7_check_sim.py [--reps 400]   # writes outputs/gpc4_prep/a7_check_sim.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z_gpc4_a7_read as R  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/gpc4_prep/a7_check_sim.json"
RHO, N_STATES, N_RENDERS, K = 0.58, 16, 16, 2
SC = "S"


def _unit_rate(rng: random.Random, p: float) -> float:
    if p <= 0 or p >= 1:
        return p
    s = (1 - RHO) / RHO
    return rng.betavariate(p * s, (1 - p) * s)


def _shift(q: float, eff: float) -> tuple[float, float]:
    """(rate with the act as A, rate with the act as B): a difference of ``eff`` centred on q where bounds allow."""
    lo = max(0.0, q - eff / 2)
    hi = lo + eff
    if hi > 1:
        hi, lo = 1.0, 1.0 - eff
    return hi, lo


def named_rows(rng: random.Random, p: float, cond: str) -> list[dict]:
    rows = []
    for s in range(N_STATES):
        q = _unit_rate(rng, p)
        pa, pb = (_shift(q, 0.30) if cond == "effect" else (1.0, 0.0) if cond == "one_letter" else (q, q))
        for order, pr in (("act_A", pa), ("act_B", pb)):
            for _ in range(K):
                rows.append({"arm": "base", "scenario": SC, "state_id": f"s{s}", "order": order,
                             "accepts": rng.random() < pr})
    return rows


def early_cold_rows(rng: random.Random, p: float, cond: str) -> tuple[list[dict], list[dict]]:
    early, cold = [], []
    for s in range(N_STATES):
        q = _unit_rate(rng, p)
        if cond == "effect":
            q = min(1.0, q + 0.30)
        for order in ("act_A", "act_B"):
            pr = (1.0 if order == "act_A" else 0.0) if cond == "one_letter" else q
            for _ in range(K):
                early.append({"arm": "base", "scenario": SC, "state_id": f"e{s}", "accepts": rng.random() < pr})
    for r in range(N_RENDERS):
        q = _unit_rate(rng, p)
        for order in ("act_A", "act_B"):
            for _ in range(K):
                cold.append({"arm": "base", "scenario": SC, "render_id": f"r{r}", "accepts": rng.random() < q})
    return early, cold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rng = random.Random("20261123|gpc4|a7|checksim")
    res = {}
    for p in (0.05, 0.25, 0.50):
        for cond in ("none", "effect", "one_letter"):
            oc = ec = 0
            mde_o, mde_e = [], []
            for i in range(a.reps):
                o = R.order_check(named_rows(rng, p, cond), SC, B=a.boot, seed=f"sim|{p}|{cond}|{i}")
                e_rows, c_rows = early_cold_rows(rng, p, cond)
                e = R.early_check(e_rows, c_rows, SC, B=a.boot, seed=f"sim|{p}|{cond}|{i}")
                oc += o["artifact"]
                ec += e["artifact"]
                mde_o.append(o["mde"])
                mde_e.append(e["mde"])
            res[f"p={p}|{cond}"] = {"order_drop": oc / a.reps, "early_drop": ec / a.reps,
                                    "order_mde_median": sorted(mde_o)[len(mde_o) // 2],
                                    "early_mde_median": sorted(mde_e)[len(mde_e) // 2]}
            print(f"p={p} {cond:10s}", json.dumps(res[f"p={p}|{cond}"]), flush=True)
    rec = {"rule": "GPC_PLAN.md §15.3 item 4 (simulated at the built sizes, through the read's functions)",
           "sizes": {"states": N_STATES, "renders": N_RENDERS, "k": K, "rho": RHO}, "reps": a.reps,
           "boot": a.boot, "results": res}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
