#!/usr/bin/env python3
"""L-beh-2's local read (Amendment 31 item 3 and its addition; zero GPU), by T1.5's estimators on
the pod's rows (the ``none`` re-run on the same engine), with Amendment 31's sign classes.

    python3 papers/st_live_reference/scripts/analyze_st2lb.py \
        --st2lb papers/st_live_reference/outputs/st2lb/gpt_oss_20b [--out st2lb_read.json]

Branches (Amendment 31 item 3, written before data): passes ΔP (H1 criterion, +0.15 with its CI
above 0) and specificity, capability and the guard holding: L-beh's non-specificity was a
training-set property; passes ΔP, specificity non-specific: the recipe carries it as well as the
data; fails ΔP with specificity specific or discriminating: the rule-right pairs cancel the protect
signal; fails both: the weight route needs a different objective. L-beh (T1.5) is read beside.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2ml as AML  # noqa: E402
import analyze_t15 as A15  # noqa: E402
import t13_driver as T13  # noqa: E402

OUTS = HERE.parent / "outputs"
DATA = HERE.parent / "data"
ARM = "beh2"


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def branch(x: dict) -> str:
    dp_ok = x["category_record"] == "H1 criterion met"
    cls = x["specificity"]["class"]
    held = x["capability"]["passes"] and not (x.get("guard") or {}).get("fails", False)
    if dp_ok and cls == "specific":
        return ("passes ΔP and specificity: L-beh's non-specificity was a training-set property"
                if held else "passes ΔP and specificity; capability or the guard fails")
    if dp_ok and cls == "non-specific":
        return "passes ΔP, fails specificity: the recipe carries the non-specificity as well"
    if dp_ok:
        return "passes ΔP, specificity discriminating (flagged, read with its fill margin)"
    if cls in ("specific", "discriminating"):
        return "fails ΔP, specificity holds: the rule-right pairs cancel the protect signal"
    return "fails both: the weight route needs a different objective"


def read(t15: Path, panel_none: Path) -> dict:
    rec = json.loads((t15 / "t15_record.json").read_text())
    engaged = json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"]
    held = json.loads((HERE.parent / "split_v1.json").read_text())["H1_held"]
    spec_ids = json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"]
    half = set(json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"])
    arms = A15.ARMS
    A15.ARMS = (ARM,)  # T1.5's parity and rows-of-record functions read this arm
    try:
        par = A15.parity(t15)
        f, src = A15.rows_of_record(t15, ARM, par, "gen")
    finally:
        A15.ARMS = arms
    if f is None:
        return {"status": "no H1_held rows for beh2", "parity": par}
    none = _jsonl(t15 / "gen" / "h1_none.jsonl")
    rows = _jsonl(f)
    nr, nx = A15.p_record(none), T13.p_by_scenario(none)
    ar, ax = A15.p_record(rows), T13.p_by_scenario(rows)
    d_rec, d_exc = A15.paired(ar, nr, engaged), A15.paired(ax, nx, engaged)
    spec = A15.specificity(t15, panel_none, ARM, spec_ids)
    spec["class"] = AML.sign_class(spec["dP_violating"]["mean"])
    x: dict = {"dP_record": d_rec, "dP_truncation_excluded": d_exc,
               "category_record": A15.category(d_rec), "category_excluded": A15.category(d_exc),
               "dP_all132_record": A15.paired(ar, nr, held),
               "truncation_arm": A15.truncation_share(rows),
               "truncation_none": A15.truncation_share(none), "rows_of_record": src,
               "specificity": spec, "capability": A15.capability(t15, ARM, par)}
    if (t15 / "panel" / ARM).exists():
        x["guard"] = A15.guard(t15, panel_none, ARM, half)
    x["verdict"] = A15.verdict(ARM, x, ARM)
    x["branch"] = branch(x)
    lb = json.loads((DATA / "t15_read_v1.json").read_text())
    beh = (lb.get("arms") or {}).get("beh") or {}
    x["l_beh_beside"] = {k: beh.get(k) for k in ("dP_record", "specificity_dP_violating",
                                                 "capability_d_accuracy", "guard_dE", "verdict")}
    return {"arm": x, "parity": par, "engine": rec["phases"].get("base_vllm")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--st2lb", type=Path, required=True)
    ap.add_argument("--panel-none", type=Path, default=OUTS / "t13" / "gpt_oss_20b" / "panel")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.st2lb, a.panel_none)
    p = a.out or a.st2lb / "st2lb_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    x = res.get("arm")
    if x:
        d, s = x["dP_record"], x["specificity"]["dP_violating"]
        print(f"beh2 dP {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"({x['category_record']}); spec {s['mean']:+.3f} ({x['specificity']['class']}); "
              f"capability {x['capability']['d_accuracy']['mean']:+.3f}; guard "
              f"{(x.get('guard') or {}).get('fails')}; verdict {x['verdict']}")
        print(f">> {x['branch']}")
    else:
        print(res.get("status"))
    print(f">> wrote {p}")


if __name__ == "__main__":
    main()
