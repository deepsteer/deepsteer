#!/usr/bin/env python3
"""KDG_PHASE1_SPEC P1-A17 item 1 (exact bounds): every Appendix D slice, full precision.

    python3 papers/kdg_panel/scripts/analyze_v2_slice_check.py

Per scenario on the 192 shared above-floor scenarios (raw frame, union run of
data/per_scenario_raw_union.csv): Delta_E = (g - g_null) on base minus (g - g_null) on instruct;
E_base, E_instruct and Delta_E for the pooled set and each slice (prompt version via the pilot
files, role, family), scenario bootstrap 10,000, seed 0, the estimator that reproduces the pooled
row of record. v1's slice rows came from an uncommitted computation that this replaces.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_phase1_session_a as A  # noqa: E402


def _f(x: str) -> float | None:
    return None if x in ("", "None") else float(x)


def main() -> int:
    by: dict[str, dict[str, float]] = {}
    for r in csv.DictReader(open(A.DATA / "per_scenario_raw_union.csv")):
        if r["run"] != "three_cell_union" or r["above_floor"] != "True":
            continue
        if r["above_floor_null"] != "True" or _f(r["g"]) is None or _f(r["g_null"]) is None:
            continue
        by.setdefault(r["scenario_id"], {})[r["model"]] = _f(r["g"]) - _f(r["g_null"])
    both = {s: v for s, v in by.items() if len(v) == 2}
    pilot = set()
    for f in sorted(A.DATA.glob("pilot_scenarios_*.json")):
        pilot |= {s["id"] for s in json.loads(f.read_text())["scenarios"]}
    slices = [("pooled", lambda s: True), ("pilot_written", lambda s: s in pilot),
              ("later_written", lambda s: s not in pilot),
              ("primaries", lambda s: not s.endswith("T")),
              ("harm_twins", lambda s: s.endswith("T"))]
    slices += [(f, (lambda f: lambda s: s.startswith(f + "-"))(f))
               for f in ("F1", "F3", "F4", "F5")]
    rep = {"rule": "KDG_PHASE1_SPEC P1-A17 item 1", "slices": {}}
    for name, keep in slices:
        ids = [s for s in both if keep(s)]
        b = {"E_base": A.boot(np.array([both[s]["base"] for s in ids])),
             "E_instruct": A.boot(np.array([both[s]["instruct"] for s in ids])),
             "Delta_E": A.boot(np.array([both[s]["base"] - both[s]["instruct"] for s in ids]))}
        rep["slices"][name] = b
        print(name, len(ids), {k: (round(v["mean"], 4), [round(c, 4) for c in v["ci95"]])
                               for k, v in b.items()})
    (A.DATA / "analysis_v2_slice_check.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
