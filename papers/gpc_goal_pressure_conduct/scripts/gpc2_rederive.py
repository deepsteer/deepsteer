"""GPC-2's second derivations and the results commit's added reads (zero GPU, saved rows only).

Re-derives the lead's scratch numbers for the GPC-2 results commit from the saved rows: C1 against
its in-context teacher in the matched cells (held-out, X1, X2), X1's base against the held-out
cell's base on the same slots, B1's adviser strata on S-hi with the acting rates of the base and
GOOD_BEHAVIOR of record, C1, and C1 at 1.5 and 2.0 times, the share of S-hi's options labeled
violating, and the session checks the ledger entries need (X2's base against the same-session base,
C1 at 2.0 against both bases). Writes ``outputs/gpc2_v1/rederive.json``.

Usage: python3 papers/gpc_goal_pressure_conduct/scripts/gpc2_rederive.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import gpc1_read as RD  # noqa: E402
import gpc2_read as R2  # noqa: E402

G = HERE.parent
O1, O2 = G / "outputs" / "gpc1_v1", G / "outputs" / "gpc2_v1"


def _holdout(p: Path) -> dict[str, float]:
    return {r["slot_id"]: float(r["choice"] == "goal") for r in RD._rows(p)  # noqa: SLF001
            if r["choice"] is not None}


def _rate(d: dict, ids: list[str]) -> float:
    return float(np.mean([d[i] for i in ids if i in d]))


def _ratio(arm: float, base: float, good: float) -> float:
    return (arm - base) / (good - base)


def _cluster_boot(a: dict, b: dict, ids: list[str], seed: int = 7, n: int = 4000) -> dict:
    """An independent game-clustered bootstrap (mean of the resampled states' differences), the
    second derivation of ``gpc1_read.paired``'s ratio-of-sums CI."""
    by: dict[str, list[float]] = {}
    for s in ids:
        if s in a and s in b:
            by.setdefault(s.split("|")[0], []).append(a[s] - b[s])
    gs = list(by)
    rng = np.random.default_rng(seed)
    draws = [np.mean([x for i in rng.integers(0, len(gs), len(gs)) for x in by[gs[i]]])
             for _ in range(n)]
    d = [x for v in by.values() for x in v]
    return {"delta": float(np.mean(d)), "ci95": [float(np.percentile(draws, 2.5)),
                                                 float(np.percentile(draws, 97.5))],
            "n": len(d), "games": len(gs)}


def main() -> dict:
    sl = json.loads((O1 / "conduct" / "slots.json").read_text())
    pres = [i for i in sl["holdout"] if sl["kept"][i]["pressure"]]
    ho = {a: _holdout(O1 / "conduct" / f"holdout_{a}.jsonl") for a in ("base", "c_conduct", "c1")}
    x1 = {a: R2._x1(O2, a) for a in ("base", "cctx", "c1")}  # noqa: SLF001
    hi = RD.screen_sets()["s_hi"]
    x2 = {a: R2._per_state(R2._rows(O2 / "x" / f"x2_{a}.jsonl"))  # noqa: SLF001
          for a in ("base", "cctx", "c1")}
    lab = RD.labels()
    base, good = RD.record(lab)
    c1 = RD.means(RD._rows(O1 / "teacher" / "screen_c1.jsonl"), lab, "c1")  # noqa: SLF001
    c15 = RD.means(RD._rows(O2 / "teacher" / "screen_c1_s150.jsonl"), lab, "c1_s150")  # noqa: SLF001
    c20 = RD.means(RD._rows(O2 / "teacher" / "screen_c1_s200.jsonl"), lab, "c1_s200")  # noqa: SLF001
    f_rows = RD._rows(O2 / "teacher" / "gpc2_f.jsonl")  # noqa: SLF001
    sbase = RD.means(f_rows, lab, "base_session")
    judged = R2._per_state(R2._rows(O2 / "x" / "b1_base.jsonl"))  # noqa: SLF001
    strata = {"adviser_non_violating": [s for s in hi if judged.get(s) == 0],
              "adviser_violating": [s for s in hi if judged.get(s) == 1]}
    arms = {"base": base, "good_behavior": good, "c1": c1, "c1_s150": c15, "c1_s200": c20}
    b1 = {}
    for k, st in strata.items():
        r = {a: _rate(d, st) for a, d in arms.items()}
        b1[k] = {"n": len(st), "acting": r,
                 "ratio": {a: _ratio(r[a], r["base"], r["good_behavior"])
                           for a in ("c1", "c1_s150", "c1_s200")},
                 "c1_s200_minus_base": RD.paired(c20, base, st),
                 "good_minus_base": RD.paired(good, base, st),
                 "x2": {a: _rate(x2[a], st) for a in x2}}
    res = {
        "c1_against_teacher": {
            "held_out": {"c1": _rate(ho["c1"], pres), "teacher": _rate(ho["c_conduct"], pres),
                         "c1_minus_teacher": R2._slot_paired(ho["c1"], ho["c_conduct"], pres,  # noqa: SLF001
                                                             "gpc2|rd|ho")},
            "x1": {"c1": _rate(x1["c1"], pres), "teacher": _rate(x1["cctx"], pres),
                   "c1_minus_teacher": R2._slot_paired(x1["c1"], x1["cctx"], pres,  # noqa: SLF001
                                                       "gpc2|rd|x1")},
            "x2": {"c1": _rate(x2["c1"], hi), "teacher": _rate(x2["cctx"], hi),
                   "c1_minus_teacher": RD.paired(x2["c1"], x2["cctx"], hi)}},
        "x2_c1_minus_base_second_derivation": _cluster_boot(x2["c1"], x2["base"], hi),
        "x1_base_against_held_out_base": {
            "x1_base": _rate(x1["base"], pres), "held_out_base": _rate(ho["base"], pres),
            "paired": R2._slot_paired(x1["base"], ho["base"], pres, "gpc2|rd|x1b")},  # noqa: SLF001
        "x2_base_against_session_base": {
            "x2_base": _rate(x2["base"], hi), "session_base": _rate(sbase, hi),
            "record_base": _rate(base, hi), "paired": RD.paired(x2["base"], sbase, hi)},
        "c1_s200": {"minus_record_base": RD.paired(c20, base, hi),
                    "minus_session_base": RD.paired(c20, sbase, hi),
                    "minus_c1_cross_session": RD.paired(c20, c1, hi)},
        "b1_strata": b1,
        "s_hi_violating_label_share": {
            "per_state_mean": float(np.mean([np.mean(lab[s]["violating"]) for s in hi])),
            "pooled_over_options": float(np.mean([v for s in hi for v in lab[s]["violating"]]))},
    }
    (O2 / "rederive.json").write_text(json.dumps(res, indent=1, default=float))
    return res


if __name__ == "__main__":
    r = main()
    ca = r["c1_against_teacher"]
    print("C1 against its teacher:",
          {k: (round(v["c1"], 3), round(v["teacher"], 3)) for k, v in ca.items()},
          "X2 C1 minus teacher", ca["x2"]["c1_minus_teacher"])
    print("X2 C1 minus base, second derivation:", r["x2_c1_minus_base_second_derivation"])
    print("X1 base against held-out base:", r["x1_base_against_held_out_base"])
    print("X2 base against session base:", r["x2_base_against_session_base"])
    print("C1 at 2.0:", r["c1_s200"])
    for k, v in r["b1_strata"].items():
        print(k, v["n"], {a: round(x, 3) for a, x in v["acting"].items()},
              {a: round(x, 3) for a, x in v["ratio"].items()})
    print("S-hi label share:", r["s_hi_violating_label_share"])
