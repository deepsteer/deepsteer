#!/usr/bin/env python3
"""GPC-3's reads (GPC_PLAN.md Amendment 2): G1 to G5 from the saved rows, zero GPU.

    python3 papers/gpc_goal_pressure_conduct/scripts/gpc3_read.py [OUT]

Writes ``OUT/gpc3_read.json``. Every read is descriptive except G4's registered "reproduces" and
G5's pilot drop rule and budget (applied on the pod; reported here).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "papers/cst_constitutional_steering/scripts")]

import gpc1_read as RD  # noqa: E402
import gpc2_read as R2  # noqa: E402
import gpc_am as AM  # noqa: E402

G = HERE.parent
CST_OUT = ROOT / "papers/cst_constitutional_steering/outputs"
X2_CCTX_POINT = -0.084  # GPC-2's C_conduct minus base on X2 (G1 b's comparison)
GOOD_LINES = 0.82  # GPC-1's good_lines ratio (G1 a's comparison)
NEAR = 0.03  # G2: "within 0.03 of the native base"


def _x(out: Path, name: str) -> dict[str, float]:
    return R2._per_state(R2._rows(out / "x" / f"{name}.jsonl"))  # noqa: SLF001


def _rate(d: dict, ids) -> float | None:
    v = [d[s] for s in ids if s in d]
    return float(np.mean(v)) if v else None


def strata(out: Path) -> dict[str, list[str]]:
    return json.loads((out / "strata.json").read_text())


def g1(out: Path, lab: dict, base_rec: dict, good: dict) -> dict:
    hi = RD.screen_sets()["s_hi"]
    rows = RD._rows(out / "teacher" / "gpc3_g1.jsonl")  # noqa: SLF001
    cl, sb = RD.means(rows, lab, "conduct_lines"), RD.means(rows, lab, "base_session")
    a = {"conduct_lines_minus_session_base": RD.paired(cl, sb, hi),
         "ratio_to_good_of_record": RD.ratio(cl, sb, base_rec, good, hi, ci=True),
         "session_base_minus_record": RD.paired(sb, base_rec, hi),
         "good_lines_gpc1": GOOD_LINES}
    r = a["ratio_to_good_of_record"].get("point")
    a["branch"] = None if r is None else (
        "the clause is a strong instruction in the native harness without GOOD's slot" if r >= 0.5
        else "the clause's game effect needs the reworded task")
    xb, xg = _x(out, "g1_x2_base"), _x(out, "g1_x2_good")
    b = {"x2_good_minus_x2_base": RD.paired(xg, xb, hi), "x2_base": _rate(xb, hi),
         "x2_good": _rate(xg, hi), "c_conduct_gpc2_point": X2_CCTX_POINT}
    ci = b["x2_good_minus_x2_base"].get("ci95")
    b["branch"] = None if not ci else (
        "the grid-format states respond to GOOD's instruction and not the clause's, so the "
        "weakness is the clause's content" if ci[1] < X2_CCTX_POINT else
        "game states in the grid's format resist both instructions, a property of the states or "
        "the rendering")
    return {"a": a, "b": b}


def g2(out: Path) -> dict:
    hi = RD.screen_sets()["s_hi"]
    lab = RD.labels()
    native = RD.means(RD._rows(out / "teacher" / "gpc3_g1.jsonl"), lab, "base_session")  # noqa: SLF001
    st = strata(out)
    x2b = _x(out, "g1_x2_base")
    sets = {"all": hi, **st}
    res = {"native_base": {k: _rate(native, v) for k, v in sets.items()},
           "x2_base": {k: _rate(x2b, v) for k, v in sets.items()}, "rungs": {}}
    for rung in ("ach", "achcot"):
        arms = {a: _x(out, f"g2_{rung}_{a}") for a in ("base", "cctx", "c1")}
        rr = {}
        for k, ids in sets.items():
            rr[k] = {"rates": {a: _rate(d, ids) for a, d in arms.items()},
                     "c1_minus_base": RD.paired(arms["c1"], arms["base"], ids),
                     "cctx_minus_base": RD.paired(arms["cctx"], arms["base"], ids),
                     "c1_minus_cctx": RD.paired(arms["c1"], arms["cctx"], ids)}
        res["rungs"][rung] = rr
    nb = res["native_base"]["all"]

    def near(rung):
        b = res["rungs"][rung]["all"]["rates"]["base"]
        return None if b is None or nb is None else abs(b - nb) <= NEAR
    na, nc = near("ach"), near("achcot")
    res["branch_push"] = (None if na is None else
                          "the achievements list carries the harness's push" if na else
                          "the CoT steps carry it" if nc else
                          "the rest of the harness (the answer format, numbered actions, the "
                          "immerse line) carries it")
    return res


def g3(out: Path, lab: dict, base_rec: dict, good: dict) -> dict:
    """Gap states by the adviser alone (no acting arm selects); the record arms and P-ctx on them."""
    j = _x(out, "g3_judge")
    pctx = RD.means(RD._rows(CST_OUT / "a52_v1" / "teacher" / "a49_pctx.jsonl"), lab, "p_ctx")  # noqa: SLF001
    gap = sorted(s for s, v in j.items() if v == 0)
    non = sorted(s for s, v in j.items() if v == 1)
    res = {"n_states_judged": len(j), "gap": len(gap), "non_gap": len(non),
           "attenuation_note": ("strata from a single adviser draw are noisy, and misclassification "
                                "between them attenuates the contrast between the strata, so a "
                                "gap-state difference is a lower bound on the difference a "
                                "better-measured adviser would show")}
    for name, ids in (("gap", gap), ("non_gap", non)):
        res[name + "_read"] = {"acting": {a: _rate(d, ids) for a, d in (
            ("base", base_rec), ("good_behavior", good), ("p_ctx", pctx))},
            "good_minus_base": RD.paired(good, base_rec, ids),
            "pctx_minus_base": RD.paired(pctx, base_rec, ids)}
    share = {s: float(np.mean(lab[s]["violating"])) for s in j if s in lab}
    bins = defaultdict(list)
    for s, v in j.items():
        if s in share:
            bins[min(int(share[s] * 5), 4)].append(v)
    res["adviser_by_label_share"] = {f"{b / 5:.1f}-{(b + 1) / 5:.1f}": {
        "n": len(v), "adviser_violating": float(np.mean(v))} for b, v in sorted(bins.items())}
    xs = np.array([share[s] for s in j if s in share])
    ys = np.array([j[s] for s in j if s in share])
    res["logistic_slope"] = _logit_slope(xs, ys)
    return res


def _logit_slope(x: np.ndarray, y: np.ndarray, iters: int = 50) -> dict:
    """Adviser violating on the state's labeled-violating share: Newton's method, slope and SE."""
    if len(x) < 10 or len(set(y.tolist())) < 2:
        return {"n": int(len(x)), "slope": None}
    X = np.column_stack([np.ones_like(x), x])
    w = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None])
        w = w + np.linalg.solve(H, X.T @ (y - p))
    p = 1 / (1 + np.exp(-X @ w))
    cov = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    se = float(np.sqrt(cov[1, 1]))
    return {"n": int(len(x)), "slope": float(w[1]), "se": se,
            "ci95": [float(w[1] - 1.96 * se), float(w[1] + 1.96 * se)]}


def g4(out: Path, lab: dict) -> dict:
    hi, lo = RD.screen_sets()["s_hi"], RD.screen_sets()["s_lo"]
    rows = RD._rows(out / "teacher" / "gpc3_g4.jsonl")  # noqa: SLF001
    arm, sb = RD.means(rows, lab, "c1_s200"), RD.means(rows, lab, "base_session")
    st = strata(out)
    main = RD.paired(arm, sb, hi)
    guard = RD.paired(arm, sb, lo)
    ho = {r["slot_id"]: float(r["choice"] == "goal") for r in RD._rows(  # noqa: SLF001
        out / "conduct2" / "g4_holdout_c1_s200.jsonl") if r["choice"] is not None}
    sl = json.loads((out.parent / "gpc1_v1" / "conduct" / "slots.json").read_text())
    clean = [i for i in sl["holdout"] if not sl["kept"][i]["pressure"]]
    pres = [i for i in sl["holdout"] if sl["kept"][i]["pressure"]]
    clean_goal = _rate(ho, clean)
    broad = (clean_goal is not None and clean_goal <= 0.90) or bool(guard.get("above_0"))
    vs = RD.violence_split(lab, hi)
    return {"rule": "GPC Amendment 2 §10.1 G4 (registered)", "seed_indices": [2, 3],
            "c1_s200_minus_session_base_s_hi": main,
            "reproduces": bool(main.get("below_0")),
            "wording_if_not": "not reproduced at an MDE of 0.08 (k 2: 0.067)",
            "s_lo_guard": guard, "guard_fails": bool(guard.get("above_0")),
            "by_stratum": {k: RD.paired(arm, sb, v) for k, v in st.items()},
            "by_violation_kind": {k: RD.paired(arm, sb, v) for k, v in vs.items()},
            "held_out": {"clean_goal_rate": clean_goal, "pressure_rate": _rate(ho, pres),
                         "unparsed": len(sl["holdout"]) - len(ho)},
            "r_broad_flag": broad}


def g5(out: Path) -> dict:
    import gpc3_pod as P3

    plan = P3._r(out / "g5", "plan")  # noqa: SLF001
    res = {"plan": plan}
    if (out / "g5" / "pilot.jsonl").exists():
        rows = P3.merged(out, "pilot")
        res["pilot"] = {s: AM.rates([r for r in rows if r["scenario"] == s]) for s in AM.SCENARIOS}
    if (out / "g5" / "arms.jsonl").exists():
        rows = P3.merged(out, "arms")
        by = defaultdict(list)
        for r in rows:
            by[(r["condition"], r["arm"])].append(r)
        conds = sorted({c for c, _ in by})
        res["by_condition"] = {c: {a: AM.rates(by[(c, a)]) for a in AM.ARMS} for c in conds}
        pooled = {a: AM.rates([r for r in rows if r["arm"] == a]) for a in AM.ARMS}
        res["pooled"] = pooled
        res["flags"] = AM.flags(pooled)
        res["flags_by_condition"] = {c: AM.flags(res["by_condition"][c]) for c in conds}
        n = (plan.get("budget") or {}).get("n", 0) if plan else 0
        p = pooled["base"]["primary"] or 0.3
        res["detection_bar"] = {"n_per_condition_and_arm": n, "conditions": len(conds),
                                "base_rate": pooled["base"]["primary"],
                                "pooled_mde": AM.mde(n * len(conds), min(max(p, 0.05), 0.5)),
                                "per_condition_mde": AM.mde(n, min(max(p, 0.05), 0.5))}
    return res


def read_all(out: Path) -> dict:
    lab = RD.labels()
    base_rec, good = RD.record(lab)
    res = {}
    for k, fn in (("g1", lambda: g1(out, lab, base_rec, good)), ("g2", lambda: g2(out)),
                  ("g3", lambda: g3(out, lab, base_rec, good)), ("g4", lambda: g4(out, lab)),
                  ("g5", lambda: g5(out))):
        try:
            res[k] = fn()
        except FileNotFoundError as e:  # an item cut on the pod reads as missing, said so
            res[k] = {"missing": str(e)}
    (out / "gpc3_read.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    o = Path(sys.argv[1]) if len(sys.argv) > 1 else G / "outputs" / "gpc3_v1"
    r = read_all(o)
    print(json.dumps({k: (v.get("branch") if isinstance(v, dict) else None) for k, v in
                      (r.get("g1") or {}).items()}, default=str))
    g4r = r.get("g4") or {}
    print("G4 reproduces:", g4r.get("reproduces"), g4r.get("c1_s200_minus_session_base_s_hi"))
    print("G5 plan:", json.dumps((r.get("g5") or {}).get("plan"), default=str)[:600])
