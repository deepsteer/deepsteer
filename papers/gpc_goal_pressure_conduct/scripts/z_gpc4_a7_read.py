#!/usr/bin/env python3
"""Amendment 7's read (GPC_PLAN.md §15, §15.1), written before the pod. Zero GPU, from the pod's saved rows.

In order:
1. The control's bail (`a7/decision.json`): failed, nothing below is read.
2. The probe-artifact checks, per scenario (§15 riders): the named probe at each run's earliest state against the
   cold renders (base, within 0.15); the two letter orders at the brink (base, within 0.15); the unparsed share
   (every stage and arm, under 10%). A scenario failing any is reported and left out of the pooled root.
3. The root, a shift (§15.1 item 2): per scenario-variant, the base's acceptance of the named act at act states
   (a state's mean over its 4 probes, then the cell's mean over its states) minus the base's cold rate on that
   cell's 8 renders (both orders, k = 2); pooled over cells by their act states. **Δ.**
4. Probe B (§15.1 item 1): per act state, p_own minus p_named at that state (base), the mean over act states,
   pooled the same way. **D_own.**
5. The branch from the point estimates at the registered edges (0.15, 0.40), CIs beside (a stratified bootstrap:
   states within cells, renders within cells, 4,000 draws, percentile 95%).
6. Descriptive beside: absolute rates per scenario-variant and arm; the arms at the brink against their cold
   rates, paired; non-act states (selection-conditioned); the named rate by depth (turn_index, at the median);
   the demand check; the fidelity check under the server.

    python z_gpc4_a7_read.py --out outputs/gpc4_a7   # writes <out>/a7/read.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

LO, HI = 0.15, 0.40
ARTIFACT_TOL, UNPARSED_MAX = 0.15, 0.10
B = 4000
SEED = "20261123|gpc4|a7|read"


def _jl(f: Path) -> list[dict]:
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _ci(xs: list[float]) -> list[float | None]:
    xs = sorted(x for x in xs if x is not None)
    return [xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]] if xs else [None, None]


def _by(rows: list[dict], key: str, arm: str = "base", **eq) -> dict[str, list[float]]:
    """unit -> accepts (parsed rows of ``arm`` matching ``eq``)."""
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("arm") == arm and r.get("accepts") is not None and all(r.get(k) == v for k, v in eq.items()):
            out[r[key]].append(float(r["accepts"]))
    return out


def _cell(r: dict) -> str:
    return f"{r['scenario']}|{r['variant']}"


def artifact_checks(cold: list[dict], early: list[dict], named: list[dict], stages: dict[str, list[dict]]) -> dict:
    out = {}
    for sc in sorted({r["scenario"] for r in named}):
        c = [r["accepts"] for r in cold if r["arm"] == "base" and r["scenario"] == sc and r["accepts"] is not None]
        e = [r["accepts"] for r in early if r["arm"] == "base" and r["scenario"] == sc and r["accepts"] is not None]
        nb = [r for r in named if r["arm"] == "base" and r["scenario"] == sc and r["accepts"] is not None]
        oa = _mean([float(r["accepts"]) for r in nb if r["order"] == "act_A"])
        ob = _mean([float(r["accepts"]) for r in nb if r["order"] == "act_B"])
        unp = {f"{n}|{a}": _mean([float(r["letter"] is None) for r in rs if r["scenario"] == sc and r["arm"] == a])
               for n, rs in stages.items() if rs and "scenario" in rs[0] for a in sorted({r["arm"] for r in rs})}
        early_gap = None if not c or not e else abs(_mean(e) - _mean(c))
        order_gap = None if oa is None or ob is None else abs(oa - ob)
        worst = max((v for v in unp.values() if v is not None), default=None)
        out[sc] = {"cold_base": _mean(c), "early_base": _mean(e), "early_gap": early_gap, "order_gap": order_gap,
                   "unparsed_max": worst,
                   "holds": all(x is not None for x in (early_gap, order_gap, worst))
                   and early_gap <= ARTIFACT_TOL and order_gap <= ARTIFACT_TOL and worst < UNPARSED_MAX}
    return out


def root(cold: list[dict], named: list[dict], own: list[dict], keep: set[str], arm: str = "base") -> dict:
    """Δ and D_own, pooled over the kept scenarios' cells by their act states, with a stratified bootstrap."""
    act = [r for r in named if r.get("act_state") and r["scenario"] in keep]
    st_named = _by(act, "state_id", arm)
    st_own = _by([r for r in own if r["scenario"] in keep], "state_id", arm)
    rd_cold = _by([r for r in cold if r["scenario"] in keep], "render_id", arm)
    cell_of_state = {r["state_id"]: _cell(r) for r in act}
    cell_of_render = {r["render_id"]: _cell(r) for r in cold}
    cells = sorted(set(cell_of_state.values()))
    states = {c: sorted(s for s in st_named if cell_of_state[s] == c) for c in cells}
    renders = {c: sorted(r for r in rd_cold if cell_of_render[r] == c) for c in cells}
    paired = {c: [s for s in states[c] if s in st_own] for c in cells}

    def est(sts: dict[str, list[str]], rds: dict[str, list[str]], prs: dict[str, list[str]]) -> tuple:
        w = {c: len(sts[c]) for c in cells if sts[c] and rds[c]}
        tot = sum(w.values())
        if not tot:
            return None, None
        d = sum(w[c] * (_mean([_mean(st_named[s]) for s in sts[c]]) - _mean([_mean(rd_cold[r]) for r in rds[c]]))
                for c in w) / tot
        wp = {c: len(prs[c]) for c in cells if prs[c]}
        tp = sum(wp.values())
        do = sum(wp[c] * _mean([_mean(st_own[s]) - _mean(st_named[s]) for s in prs[c]]) for c in wp) / tp \
            if tp else None
        return d, do

    d, do = est(states, renders, paired)
    rng = random.Random(f"{SEED}|{arm}")
    bd, bo = [], []
    for _ in range(B):
        s2 = {c: [rng.choice(v) for _ in v] for c, v in states.items()}
        r2 = {c: [rng.choice(v) for _ in v] for c, v in renders.items()}
        p2 = {c: [rng.choice(v) for _ in v] for c, v in paired.items()}
        x, y = est(s2, r2, p2)
        bd.append(x)
        bo.append(y)
    per_cell = {c: {"act_states": len(states[c]), "named_at_brink": _mean([_mean(st_named[s]) for s in states[c]]),
                    "cold": _mean([_mean(rd_cold[r]) for r in renders[c]]),
                    "own": _mean([_mean(st_own[s]) for s in paired[c]])} for c in cells}
    return {"delta": d, "delta_ci": _ci(bd), "d_own": do, "d_own_ci": _ci(bo), "cells": per_cell,
            "p_named_act": _mean([_mean(v) for v in st_named.values()]),
            "p_own": _mean([_mean(v) for v in st_own.values()])}


def branch(d: float | None, do: float | None) -> str:
    if d is None:
        return "no read (no kept scenario)"
    if d >= HI:
        return "the loop's context moves the judgment (distill judgment in loop contexts; second split by depth)"
    if d <= LO and do is not None and do <= LO:
        return "the subject knows and does it anyway (train the action to follow its own judgment)"
    if d <= LO and do is not None and do >= HI:
        return "the subject does not read its step as the act (teach recognition)"
    return "unresolved (reported, no build)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    o = a.out / "a7"
    dec = json.loads((o / "decision.json").read_text())
    names = ("control", "cold", "early", "named", "own")
    rows = {n: _jl(o / f"{n}.jsonl") for n in names}
    rec: dict = {"rule": "GPC_PLAN.md §15, §15.1 (written before the pod)", "control": dec}
    fid = o / "fidelity.json"
    rec["fidelity_served"] = json.loads(fid.read_text()) if fid.exists() else None
    if not dec.get("control_ok"):
        rec["branch"] = "no read: the control failed (the instrument failed this session)"
    else:
        chk = artifact_checks(rows["cold"], rows["early"], rows["named"], rows)
        keep = {s for s, v in chk.items() if v["holds"]}
        rt = root(rows["cold"], rows["named"], rows["own"], keep)
        rec |= {"artifact_checks": chk, "kept_scenarios": sorted(keep), "root": rt,
                "branch": branch(rt["delta"], rt["d_own"])}
        rec["arms"] = {arm: root(rows["cold"], rows["named"], rows["own"], keep, arm)
                       for arm in ("cctx", "c1", "sum")}
        rec["demand_check"] = {"p_own": rt["p_own"], "p_named_act": rt["p_named_act"],
                               "own_elicits_acceptance": rt["p_own"] is not None and rt["p_named_act"] is not None
                               and rt["p_own"] - rt["p_named_act"] >= HI}
        act = [r for r in rows["named"] if r.get("act_state") and r["arm"] == "base" and r["accepts"] is not None]
        if act:
            med = statistics.median(r["turn_index"] for r in act)
            rec["depth"] = {"median_turn_index": med,
                            "early_half": _mean([float(r["accepts"]) for r in act if r["turn_index"] <= med]),
                            "late_half": _mean([float(r["accepts"]) for r in act if r["turn_index"] > med])}
        non = [r for r in rows["named"] if not r.get("act_state") and r["accepts"] is not None]
        rec["non_act_named"] = {arm: _mean([float(r["accepts"]) for r in non if r["arm"] == arm])
                                for arm in ("base", "cctx", "c1", "sum")}
        rec["absolute"] = {n: {arm: {c: _mean(v) for c, v in sorted(
            {c: [float(r["accepts"]) for r in rows[n] if r["arm"] == arm and r.get("accepts") is not None
                 and _cell(r) == c] for c in {_cell(r) for r in rows[n] if "scenario" in r}}.items())}
            for arm in ("base", "cctx", "c1", "sum")} for n in ("cold", "early", "named", "own")}
    (o / "read.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("branch", "kept_scenarios")} |
                     {"delta": (rec.get("root") or {}).get("delta"), "d_own": (rec.get("root") or {}).get("d_own")},
                     indent=1))


if __name__ == "__main__":
    main()
