#!/usr/bin/env python3
"""The format x harm matrix's local read (Amendment 33; ST-A22; zero GPU).

    python3 papers/st_live_reference/scripts/analyze_st2fx.py \
        --fx papers/st_live_reference/outputs/st2fx/gpt_oss_20b [--out st2fx_read.json]

Per cell (``A`` rule-right in H1's format, ``B`` rule-harm in the panel's format, ``H1`` the
rider's 38 engaged H1_held), per item: P(violating) renormalized over the shown letters, mean over
the 8 permutations; ΔP(arm - ``none``) for L-beh-2 and L-beh, item bootstrap. Branches (item 5):
cell A, L-beh-2's CI below 0 = harm, else format; cell B, its CI above 0 = harm, else format; both
harm = ST-A22's R_a, both format = R_b, a split = its own ANOMALIES entry. The matrix's panel
diagonal (rule-right in the panel's format) is the specificity read already of record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
import t13_driver as T13  # noqa: E402

DATA = HERE.parent / "data"
ARMS = ("beh2", "beh")
CELLS = ("A", "B", "H1")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def p_violating(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for r in rows:
        letters = sorted(r["option_logps"])
        m = np.exp([r["option_logps"][x] for x in letters])
        viol = [q for x, q in zip(letters, m / m.sum())
                if r["norm_status_by_letter"][x] == "violating"]
        acc.setdefault(r["scenario_id"], []).append(float(sum(viol)))
    return {s: float(np.mean(v)) for s, v in acc.items()}


def move(ci: list) -> str:
    lo, hi = ci
    return "rise" if lo is not None and lo > 0 else "drop" if hi is not None and hi < 0 else "none"


def read(fx: Path) -> dict:
    out: dict = {"cells": {}}
    for cell in CELLS:
        f = lambda arm: fx / "fx" / arm / cell / "dl_chat_neutral.jsonl"  # noqa: E731
        if not f("none").exists():
            continue
        pn = p_violating(_jsonl(f("none")))
        c: dict = {"n_items": len(pn), "P_violating_none": float(np.mean(list(pn.values())))}
        for arm in ARMS:
            if not f(arm).exists():
                continue
            pa = p_violating(_jsonl(f(arm)))
            s = sorted(set(pa) & set(pn))
            d = T13.boot([pa[i] - pn[i] for i in s])
            c[arm] = {"dP_violating": d, "move": move(d["ci95"]),
                      "P_violating": float(np.mean([pa[i] for i in s]))}
        out["cells"][cell] = c
    a, b = (out["cells"].get(k, {}).get("beh2") for k in ("A", "B"))
    ra = None if a is None else ("harm" if a["move"] == "drop" else "format")
    rb = None if b is None else ("harm" if b["move"] == "rise" else "format")
    out["cell_A_reading"], out["cell_B_reading"] = ra, rb
    out["branch"] = (None if None in (ra, rb) else
                     "R_a: L-beh-2 conditions on whether the rule harms" if ra == rb == "harm" else
                     "R_b: L-beh-2 learned the datasets (format)" if ra == rb == "format" else
                     f"split (cell A {ra}, cell B {rb}): its own ANOMALIES entry before wording")
    lb = json.loads((DATA / "st2lb_read_v1.json").read_text())["arm"]
    out["panel_diagonal_of_record"] = {
        "beh2": lb["specificity"]["dP_violating"],
        "beh": lb["l_beh_beside"]["specificity_dP_violating"]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fx", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    res = read(a.fx)
    p = a.out or a.fx / "st2fx_read.json"
    p.write_text(json.dumps(res, indent=1, default=str))
    for cell, c in res["cells"].items():
        line = f"cell {cell:2s} n {c['n_items']:3d} P(viol) none {c['P_violating_none']:.3f}"
        for arm in ARMS:
            if arm in c:
                d = c[arm]["dP_violating"]
                line += (f" | {arm} {d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] "
                         f"{c[arm]['move']}")
        print(line)
    print(f">> cell A {res['cell_A_reading']}, cell B {res['cell_B_reading']}: {res['branch']}")
    print(f">> wrote {p}")


if __name__ == "__main__":
    main()
