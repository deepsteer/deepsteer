#!/usr/bin/env python3
"""CST Amendment 16 item 1(b): the trace-length-stratified read on S0a's saved rollouts.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/s0a_length_strata.py

Descriptive only: trace length responds to the arm (A0s traces are 2.6 times longer, CST-A8), so
conditioning on it conditions on a post-treatment variable and this read never separates the
length rival (A0n does, Amendment 16 item 1(a)).

Operator cell, 458 clear items, rollouts with an action (capped excluded, as in every action
share). Quintile edges from the pooled A0 and A0s rollouts' ``trace_len``. Per bin and arm: the
ordered-option share (per-scenario means, scenario bootstrap). Per bin: the scenario-paired
A0s - A0 difference on the scenarios with at least one rollout of each arm in the bin. Writes
outputs/s0a_length_strata.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402

CST = HERE.parent
SEED, N_BOOT = 20261109, 10000


def boot(x: list[float]) -> dict:
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    idx = np.random.default_rng(SEED).integers(0, len(x), size=(N_BOOT, len(x)))
    lo, hi = np.percentile(x[idx].mean(axis=1), [2.5, 97.5])
    return {"mean": float(x.mean()), "ci95": [float(lo), float(hi)], "n": int(len(x))}


def rows(lab: dict) -> list[tuple[str, str, int, int]]:
    """(arm, scenario, trace_len, ordered) for every operator-cell clear rollout with an action."""
    out = []
    for a in ("a0", "a0s"):
        for r in A.load_jsonl(f"gen/{a}/panel_operator_*.jsonl"):
            if r["condition"] != "operator":  # the glob also matches operator_pr's chunks
                continue
            rec = lab["record"][r["scenario_id"]]
            if rec["stratum_of_record"] != "clear_refuse" or r["hit_cap"] or r["option"] is None:
                continue
            out.append((a, r["scenario_id"], int(r["trace_len"]),
                        int(r["option"] == rec["ordered_option"])))
    return out


def main() -> None:
    lab = A.labels()
    rr = rows(lab)
    # most probable failure: reading another cell than the headline's; the unstratified shares
    # must equal s0a_reads' actions on panel|operator|clear_refuse
    ref = json.loads((CST / "outputs/s0a_reads.json").read_text())["actions"][
        "panel|operator|clear_refuse"]["ordered"]
    for a in ("a0", "a0s"):
        per = defaultdict(list)
        for arm, s, _, o in rr:
            if arm == a:
                per[s].append(o)
        assert abs(np.mean([np.mean(v) for v in per.values()]) - ref[a]) < 1e-9, (a, ref[a])
    edges = np.quantile([t for _, _, t, _ in rr], [0.2, 0.4, 0.6, 0.8]).tolist()
    bins = []
    for b in range(5):
        lo = -np.inf if b == 0 else edges[b - 1]
        hi = np.inf if b == 4 else edges[b]
        per = {a: defaultdict(list) for a in ("a0", "a0s")}
        for arm, s, t, o in rr:
            if lo < t <= hi or (b == 0 and t <= hi):
                per[arm][s].append(o)
        paired = sorted(set(per["a0"]) & set(per["a0s"]))
        bins.append({
            "bin": b + 1, "trace_len_range": [None if b == 0 else edges[b - 1],
                                              None if b == 4 else edges[b]],
            "rollouts": {a: sum(len(v) for v in per[a].values()) for a in per},
            "ordered_share": {a: boot([float(np.mean(v)) for v in per[a].values()]) for a in per},
            "paired_scenarios": len(paired),
            "paired_d_a0s_minus_a0": boot([float(np.mean(per["a0s"][s]) - np.mean(per["a0"][s]))
                                           for s in paired]),
        })
    med = {a: float(np.median([t for arm, _, t, _ in rr if arm == a])) for a in ("a0", "a0s")}
    res = {"rule": "CST Amendment 16 item 1(b), descriptive", "edges": edges,
           "trace_len_median": med, "bins": bins}
    (CST / "outputs/s0a_length_strata.json").write_text(json.dumps(res, indent=1) + "\n")
    print("edges", [round(e) for e in edges], "median", med)
    for x in bins:
        s, d = x["ordered_share"], x["paired_d_a0s_minus_a0"]
        sh = {a: None if s[a]["mean"] is None else round(s[a]["mean"], 3) for a in s}
        pd = None if d["mean"] is None else (round(d["mean"], 3), [round(c, 3) for c in d["ci95"]])
        print(x["bin"], x["rollouts"], sh, "paired", x["paired_scenarios"], pd)


if __name__ == "__main__":
    main()
