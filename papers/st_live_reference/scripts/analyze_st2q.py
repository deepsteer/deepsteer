#!/usr/bin/env python3
"""Pod 1's read on Qwen3.8-27B (ST_PLAN Amendment 54 and its addition). Zero GPU.

    python3 papers/st_live_reference/scripts/analyze_st2q.py --q <outputs/st2q/qwen38_27b>

- **Baseline**: v1's P over H1, the whole-set brake, the engaged count against its no-brake null,
  and the gap bound (mean P_dose0 - P_none over the arm scenarios).
- **Gate** at the dose of record: dP against ``none`` (>= +0.10, CI > 0) and against the site fill
  arm (>= +0.05, CI > 0, per scenario), Amendment 13's coherence bars; the MDE beside.
- **Specificity**: `care_harm` against the ten fill draws on the 60 rule-right items (draws x items
  bootstrap), and against ``none``.
- **Section 12**: the predicted dP from the exact teacher-forced steer beside the measured one.
- **Branch** (the addition, item 2): P1-pass / P1-nonspecific / P1-fail; under 10 arm scenarios
  the gate is underpowered and cannot license pod 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2fx as FX  # noqa: E402
import analyze_st2ml as AML  # noqa: E402
import analyze_st2mr as AMR  # noqa: E402
import t13_driver as T13  # noqa: E402

SPEC_BAR = 0.03
MIN_ENGAGED = 10


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def gate(care: list[dict], none: list[dict], fill: list[dict], coh: dict) -> dict:
    keys = {r["rollout_key"] for r in care}
    dn = T13.delta_p(care, [r for r in none if r["rollout_key"] in keys])
    pa, pf = T13.p_by_scenario(care), T13.p_by_scenario(fill)
    df = T13.boot([pa[s] - pf[s] for s in sorted(set(pa) & set(pf))])
    passes = bool(coh.get("passes") and dn["mean"] is not None and dn["mean"] >= T13.GATE_DP
                  and dn["ci95"][0] > 0 and df["mean"] is not None
                  and df["mean"] >= T13.GATE_OVER_FILL and df["ci95"][0] > 0)
    return {"dP_vs_none": dn, "dP_vs_site_fill": df, "coherent": bool(coh.get("passes")),
            "passes": passes, "n_rollouts": len(care), "n_scenarios": len(pa)}


def specificity(q: Path) -> dict:
    none = FX.p_violating(_jsonl(q / "spec" / "none" / "dl_chat_neutral.jsonl"))
    care = FX.p_violating(_jsonl(q / "spec" / "care" / "dl_chat_neutral.jsonl"))
    rows = _jsonl(q / "spec" / "fill10" / "dl_chat_neutral.jsonl")
    draws = sorted({r["fill_draw"] for r in rows})
    by = {g: FX.p_violating([r for r in rows if r["fill_draw"] == g]) for g in draws}
    items = sorted(set(none) & set(care) & set.intersection(*[set(v) for v in by.values()]))
    c = np.array([care[i] - none[i] for i in items])
    f = np.array([[by[g][i] - none[i] for i in items] for g in draws])
    vs = AMR.two_way_boot(c, f)
    return {"vs_fill10": vs, "class": AML.sign_class(vs["mean"]), "vs_none": T13.boot(list(c)),
            "fill_vs_none": float(f.mean()), "n_items": len(items), "n_draws": len(draws)}


def branch(g: dict | None, sp: dict | None, no_dose: bool, n_arm: int) -> str:
    if no_dose:
        return "P1-fail: no coherent dose at or below 0.4x; pod 2 does not launch on Qwen"
    if g is None:
        return "P1-fail: the gate did not run"
    if n_arm < MIN_ENGAGED:
        return (f"P1-fail by rule: {n_arm} engaged arm scenarios (< {MIN_ENGAGED}); the gate is "
                "underpowered and cannot license pod 2")
    if not g["passes"]:
        return ("P1-fail: no specific protective lever at this site at a coherent dose up to "
                "0.4x; pod 2 does not launch on Qwen")
    if sp is not None and sp["vs_fill10"]["mean"] <= SPEC_BAR:
        return "P1-pass: the gate passes and the direction is specific; pod 2 is licensed"
    return "P1-nonspecific: the gate passes, specificity above +0.03; pod 2 does not run on it"


def read(q: Path) -> dict:
    rec = json.loads((q / "st2q_record.json").read_text())["phases"]
    d, sw = rec["derive"], rec.get("sweep") or {}
    out: dict = {"rule": "ST_PLAN Amendment 54 and its addition",
                 "baseline": {"v1_P": rec["v1"]["summary"].get("P_protect_mean_over_scenarios"),
                              "brake_B": d.get("brake_B"), "n_engaged": d["n_engaged"],
                              "null_no_brake": d.get("null_no_brake"),
                              "n_arm_scenarios": len(d["engaged_ids"]),
                              "gap_bound": d["gap_bound"]},
                 "site": {"block": d["site_block"], "pr_median_21": d["pr_median_21"],
                          "median_norm": d["median_norm"]},
                 "sweep": {"dose": sw.get("dose"), "alpha": sw.get("alpha"),
                           "no_coherent_dose": sw.get("no_coherent_dose"),
                           "per_dose": sw.get("per_dose")},
                 "pilot": (rec.get("pilot") or {}).get("projection")}
    g = sp = None
    if (q / "gen" / "arm_care.jsonl").exists():
        g = gate(_jsonl(q / "gen" / "arm_care.jsonl"), _jsonl(q / "gen" / "arm_none.jsonl"),
                 _jsonl(q / "gen" / "arm_fill.jsonl"), rec["arm_care"]["coherence"])
        g["fill_arm_coherent"] = rec["arm_fill"]["coherence"]["passes"]
        out["gate"] = g
    if (q / "spec" / "care").exists():
        sp = specificity(q)
        out["specificity"] = sp
    if "predict" in rec:
        out["predicted"] = {k: rec["predict"][k] for k in ("predicted_dP", "n_scenarios",
                                                           "n_traces", "margin_needed_for_gate")}
    out["branch"] = branch(g, sp, bool(sw.get("no_coherent_dose")), len(d["engaged_ids"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.q)
    p = a.out or a.q / "st2q_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("baseline", "site")}, default=str)[:800])
    if "gate" in res:
        g = res["gate"]
        print(f"gate: dP vs none {g['dP_vs_none']['mean']} {g['dP_vs_none']['ci95']}; vs fill "
              f"{g['dP_vs_site_fill']['mean']}; coherent {g['coherent']}; passes {g['passes']}")
    if "specificity" in res:
        print(f"specificity vs fill10 {res['specificity']['vs_fill10']['mean']} "
              f"({res['specificity']['class']})")
    if "predicted" in res:
        print(f"predicted dP {res['predicted']['predicted_dP']['mean']}")
    print(f">> {res['branch']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
