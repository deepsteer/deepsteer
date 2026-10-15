#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 6: audit of Appendix D's crossing sentence from the saved masks.

    python3 papers/kdg_panel/scripts/analyze_v2_crossing_audit.py

analyze_paper8.py's A17 E2 computed the predicted binary excess (net crossings of p_D = 0.5, twin
to primary) on the paired screened scenarios with a non-violating greedy reference (88), but the
observed excess and the up/down crossing counts on all paired screened scenarios (100). This
recomputes every piece on each set, from data/per_scenario_union.csv, with the pipeline's own
bootstrap.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_paper8 as P8  # noqa: E402

DATA = HERE.parent / "data"


def _f(x: str) -> float | None:
    return None if x in ("", "None") else float(x)


def main() -> int:
    rows = list(csv.DictReader(open(DATA / "per_scenario_union.csv")))
    both = [r for r in rows if r["screened"] == "True" and all(
        _f(r[k]) is not None for k in ("kdg_binary", "kdg_binary_null", "pD", "pD_null"))]
    sets = {"all_paired_screened": both,
            "non_violating_reference": [r for r in both if r["j_greedy_status"] != "violating"]}
    rep: dict = {"rule": "KDG_PHASE1_SPEC P1-A17 item 6", "sets": {}}
    for name, S in sets.items():
        pred = [(_f(r["pD"]) > 0.5) - (_f(r["pD_null"]) > 0.5) for r in S]
        obs = [_f(r["kdg_binary"]) - _f(r["kdg_binary_null"]) for r in S]
        rep["sets"][name] = {
            "n": len(S), "predicted_from_pD": P8.boot_mean(pred), "observed": P8.boot_mean(obs),
            "crossings_up": sum(_f(r["pD"]) > 0.5 >= _f(r["pD_null"]) for r in S),
            "crossings_down": sum(_f(r["pD_null"]) > 0.5 >= _f(r["pD"]) for r in S)}
        s = rep["sets"][name]
        pr, ob = s["predicted_from_pD"], s["observed"]
        print(name, s["n"], round(pr["mean"], 3), [round(x, 3) for x in pr["ci95"]],
              round(ob["mean"], 3), [round(x, 3) for x in ob["ci95"]], s["crossings_up"],
              s["crossings_down"])
    (DATA / "analysis_v2_crossing_audit.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
