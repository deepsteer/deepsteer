#!/usr/bin/env python3
"""The ``care_harm`` ladder's local read (Amendment 31 items 1 and 4; ST-A20; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_st2ml.py \
        --ladder papers/st_live_reference/outputs/st2ml/muse_glimmer_30b [--out st2ml_read.json]

- **Ladder**: ``care_harm`` at 0.2x (Cell 4's arm), 0.3x and 0.4x Muse's median norm, each against
  Cell 4's ``none`` and ``fill`` (§5.0's gate, Amendment 13's relative coherence bars); each rung's
  rise over 0.2x paired by scenario (P per scenario), scenario-bootstrap CI.
- **Specificity** against Cell 4's ``spec_none`` (the baseline of record; the ladder pod re-reads it
  for parity), with Amendment 31's sign classes (> +0.03 non-specific; < -0.03 discriminating, a
  flagged class read with the fill margin, the arm's difference from ``spec_fill`` at its rung).
- **Branch** (Amendment 31 item 4): R_a if protection rises with dose (the highest coherent rung
  above 0.2x, CI above 0) and the rule-right read is discriminating at every coherent rung; R_b if
  neither moves with dose (the rise's CI includes 0 at every coherent rung and no rung is
  discriminating); otherwise unresolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import analyze_st2 as A2  # noqa: E402
import analyze_st2dl as ADL  # noqa: E402
import t13_driver as T13  # noqa: E402

OUTS = HERE.parent / "outputs"
BAR = 0.03
RUNGS = ("03", "04")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def sign_class(mean: float | None) -> str | None:
    """Amendment 31 item 1."""
    if mean is None:
        return None
    return ("non-specific" if mean > BAR else "discriminating" if mean < -BAR else "specific")


def rise(arm: list[dict], base: list[dict]) -> dict:
    pa, pb = T13.p_by_scenario(arm), T13.p_by_scenario(base)
    s = sorted(set(pa) & set(pb))
    return T13.boot([pa[x] - pb[x] for x in s])


def read(lad: Path, c4: Path) -> dict:
    g4 = lambda n: _jsonl(c4 / "gen" / f"arm_{n}.jsonl")  # noqa: E731
    none, fill, c02 = g4("none"), g4("fill"), g4("care_harm")
    ids = json.loads((HERE.parent / "data" / "t15_specificity_v1.json").read_text())["ids"]
    rec = json.loads((lad / "st2ml_record.json").read_text())["phases"]
    base = c4 / "spec" / "none"

    def spec(d: Path, arm: str, fill_arm: str) -> dict:
        a, f = ADL.spec_items(d, base, arm, ids), ADL.spec_items(d if fill_arm != "fill" else c4,
                                                                 base, fill_arm, ids)
        s = sorted(set(a) & set(f))
        d_ = T13.boot(list(a.values()))
        return {"dP_violating": d_, "class": sign_class(d_["mean"]),
                "fill_margin": T13.boot([a[i] - f[i] for i in s]),
                "fill_itself": T13.boot(list(f.values()))}
    ladder = {"02": {**A2.arm_read(c02, none, fill, "fill"), "alpha_factor": 0.2,
                     "spec": spec(c4, "care_harm", "fill"), "source": "Cell 4"}}
    for r in RUNGS:
        p = lad / "gen" / f"arm_care_{r}.jsonl"
        if not p.exists():
            ladder[r] = {"status": (rec.get(f"arm_care_{r}") or {}).get("status", "not run")}
            continue
        rows = _jsonl(p)
        ladder[r] = {**A2.arm_read(rows, none, fill, "fill"), "alpha_factor": int(r) / 10,
                     "rise_over_02": rise(rows, c02),
                     "spec": spec(lad, f"care_{r}", f"fill_{r}")}
    coh = [r for r in RUNGS if "coherence" in ladder[r] and ladder[r]["coherence"]["passes"]]
    top = coh[-1] if coh else None
    lo = lambda d: d["ci95"][0]  # noqa: E731
    rises = bool(top and lo(ladder[top]["rise_over_02"]) is not None
                 and lo(ladder[top]["rise_over_02"]) > 0)
    disc = bool(coh) and all(ladder[r]["spec"]["class"] == "discriminating" for r in coh)
    flat = bool(coh) and all(lo(ladder[r]["rise_over_02"]) is not None
                             and lo(ladder[r]["rise_over_02"]) <= 0 <= ladder[r]["rise_over_02"]
                             ["ci95"][1] for r in coh) and not any(
        ladder[r]["spec"]["class"] == "discriminating" for r in coh)
    branch = ("R_a (conditional content): protection rises with dose and the rule-right read stays "
              "on the rule side" if rises and disc else
              "R_b (deliberation or noise): neither moves with dose" if flat else
              "unresolved" if coh else "no coherent rung")
    parity = (rec.get("spec_none") or {}).get("parity")
    return {"ladder": ladder, "coherent_rungs": coh, "branch": branch, "parity": parity}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", type=Path, required=True)
    ap.add_argument("--cell4", type=Path, default=OUTS / "st2m" / "muse_glimmer_30b")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.ladder, a.cell4)
    p = a.out or a.ladder / "st2ml_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    for r, x in res["ladder"].items():
        if "dP_vs_none" not in x:
            print(f"{r}: {x.get('status')}")
            continue
        d, s = x["dP_vs_none"], x["spec"]["dP_violating"]
        ri = x.get("rise_over_02")
        print(f"care_harm {r}: dP {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
              f"vs fill {x['dP_vs_fill']['mean']:+.3f}; coherent {x['coherence']['passes']}; "
              f"spec {s['mean']:+.3f} ({x['spec']['class']}, fill margin "
              f"{x['spec']['fill_margin']['mean']:+.3f})"
              + (f"; rise over 0.2x {ri['mean']:+.3f} [{ri['ci95'][0]:+.3f}, {ri['ci95'][1]:+.3f}]"
                 if ri else ""))
    print(f">> {res['branch']}; parity {res['parity']}\n>> wrote {p}")


if __name__ == "__main__":
    main()
