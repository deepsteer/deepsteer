#!/usr/bin/env python3
"""The affordability rate (GPC_PLAN.md §11.23, the lead's ruling 1): the lowest ODCV-Bench base rate at which
the registered read's two draws fit §11.1's third condition (12 A100-h or less), at a clause effect of half the
base rate, from z_gpc4_power.py's own simulation (the registered bar: the adapter's reduction over the clause's,
lower bound 0.5, 80% power at a true ratio of 1.0; 39 scenarios, two variants each, clustered by scenario).

The cost of one pass at k runs a scenario-variant is k times the three arms' measured cost (launch 6: the base's
80 runs took 66 min of pod time at concurrency 8; the clause arm costs about the base's; an adapter carrying
C1's update about 2.2 A100-h, from C1's run at about 4 times the clause arms'). Two draws at k cost 2 k times
that, so the read fits 12 A100-h only at the k it allows; the affordability rate is the lowest base rate whose
needed k fits. The spread between scenarios (the intra-class correlation of the two variants) is the
registered one from Qwen3.6-27B's saved runs (0.53); launch 6's own two variants under the two zero-GPU reads
(Codex, the file trace) are reported beside as a sensitivity. §11.25 item 2 adds the best case (concurrency
16 at twice launch 6's rate at 8), the rate of record from launch 7 on.

    python z_gpc4_affordability.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import z_gpc4_power as PW  # noqa: E402

GPC = HERE.parent
OUT = GPC / "outputs/gpc4_prep/affordability.json"
CAP_H = 12.0
PASS_H = {"measured": {"base": 1.1, "clause": 1.1, "adapter": 2.2},  # launch 6 (§11.21's table)
          "adapter_at_the_base_cost": {"base": 1.1, "clause": 1.1, "adapter": 1.1},
          # §11.25 item 2: the best case, concurrency 16 at twice launch 6's rate at 8 (rehearsal 9's stub ran
          # all 80 at 16 cleanly, about 20 s a run, so the host does not bind; the server is the limit)
          "best_case_16": {"base": 0.55, "clause": 0.55, "adapter": 1.1},
          # §12 item 7 (the lead, 2026-11-22): the pilot's measured cell, 78 runs at 16 under 16,384 in 0.517 h
          # (pilot.json, cell A); the clause arm at the cell's cost, C1 at 1.1 times it (GPC-A14's token ratio,
          # PILOT_RATIO); beside, C1 at launch 6's measured ratio to the clause arm (2.0, 1.1 against 2.2)
          "pilot_measured": {"base": 0.52, "clause": 0.52, "adapter": 0.57},
          "pilot_measured_c1_at_launch6_ratio": {"base": 0.52, "clause": 0.52, "adapter": 1.04}}
GRID = [round(x, 3) for x in np.arange(0.20, 0.801, 0.025)]


def icc_of(m: list[bool], i: list[bool]) -> float:
    y = np.array([m, i], float)
    mean = float(y.mean())
    cov = float(np.mean((y[0] - mean) * (y[1] - mean)))
    return max(min(cov / (mean * (1 - mean)), 0.95), 0.02) if 0 < mean < 1 else 0.02


def launch6_icc() -> dict:
    """The two variants' agreement in launch 6's base under each zero-GPU read (39 scenarios)."""
    import gpc4_pod as P4

    broken = P4.odcv_route_broken()
    o = GPC / "outputs/gpc4_v1"
    out = {}
    ft = {r["run_id"]: r for r in json.loads((o / "odcv/file_trace_launch6.json").read_text())["runs"]}
    cx = {}
    for x in (o / "odcv/codex_odcv_base.jsonl").read_text().splitlines():
        r = json.loads(x)
        cx[r["run_id"]] = r
    for name, val in (("file_trace", lambda rid: ft.get(rid, {}).get("changed_shipped")),
                      ("codex", lambda rid: (cx.get(rid) or {}).get("misaligned"))):
        scen = sorted({r["scenario"] for r in ft.values()} - broken)
        pairs = [(val(f"base_m_{s}"), val(f"base_i_{s}")) for s in scen]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        out[name] = {"n_scenarios": len(pairs), "icc": icc_of([a for a, _ in pairs], [b for _, b in pairs]),
                     "rate": float(np.mean([a for a, _ in pairs] + [b for _, b in pairs]))}
    return out


def k_needed(p: float, icc: float, kmax: int = 4) -> int | None:
    for k in range(1, kmax + 1):
        _, lb = PW.simulate(39, 2, p, icc, 0.5, 1.0, k)
        if np.nanmean(lb >= 0.5) >= 0.8:
            return k
    return None


def main() -> None:
    reg = json.loads((GPC / "outputs/gpc4_prep/power_39.json").read_text())["odcv_saved_qwen3.6_27b"]
    iccs = {"registered (Qwen3.6-27B's saved runs)": reg["icc"]} | {
        f"launch 6, {k}": v["icc"] for k, v in launch6_icc().items()}
    rec = {"rule": "GPC_PLAN.md §11.23 ruling 1 (§11.1's third condition: the registered read's two draws at "
                   "12 A100-h or less; a clause effect of half the base rate; z_gpc4_power.py's simulation)",
           "cap_h": CAP_H, "pass_h": PASS_H, "icc": iccs, "grid": {}, "affordability_rate": {}}
    for label, icc in iccs.items():
        ks = {p: k_needed(p, icc) for p in GRID}
        rec["grid"][label] = ks
        for pk, ph in PASS_H.items():
            per_k = sum(ph.values())
            fits = [p for p, k in ks.items() if k is not None and 2 * k * per_k <= CAP_H]
            rec["affordability_rate"][f"{label} | {pk}"] = {
                "rate": min(fits) if fits else None, "k_that_fits": int(CAP_H // (2 * per_k)),
                "pass_h": per_k,
                "two_draws_h_at_k1": 2 * per_k}
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec["icc"]))
    for k, v in rec["affordability_rate"].items():
        print(k, v)
    for label, ks in rec["grid"].items():
        print(label, {p: k for p, k in ks.items()})


if __name__ == "__main__":
    main()
