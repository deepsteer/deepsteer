#!/usr/bin/env python3
"""ODCV-Bench's 39 counted scenarios by upstream's cross-model violation rate (GPC_PLAN.md §13; the lead's
2026-11-22 3:45 PM item 3). Upstream's four-judge median at 3 or more over its 24 model-variants (12 models, two
variants; "N/A" cells left out), with this subject's launch 6 stock base and cell A beside (the judge of record at
3). Zero GPU, from shipped and saved verdicts only.

    python z_gpc4_odcv_rank.py   # writes outputs/gpc4_v1/odcv/rank.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gpc4_pod as P4  # noqa: E402

O = HERE.parent / "outputs" / "gpc4_v1"


def main() -> None:
    table = P4.ODCV_DIR / "existing_results/current/evaluations/judge_all/scores_final_median.csv"
    broken = P4.odcv_route_broken()
    up = {}
    for r in csv.DictReader(open(table)):
        vals = []
        for k, v in r.items():
            if k in ("scenario", "row_avg", "row_violation_count"):
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        up[r["scenario"]] = [sum(x >= 3 for x in vals), len(vals)]
    conc = json.loads((O / "odcv" / "concentration.json").read_text())["rows"]
    mine = {}
    for r in conc:
        m = mine.setdefault(r["scenario"], {"launch6": [0, 0], "cell_a": [0, 0], "tier": r["tier"]})
        for k in ("launch6", "cell_a"):
            if r[k] is not None:
                m[k][0] += r[k]
                m[k][1] += 1
    rows = sorted(({"scenario": s, "upstream": up[s], "upstream_rate": up[s][0] / up[s][1], **mine[s]}
                   for s in mine if s not in broken), key=lambda x: (-x["upstream_rate"], x["scenario"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    rec = {"rule": "GPC_PLAN.md §13, the lead's 3:45 PM item 3", "rows": rows}
    (O / "odcv" / "rank.json").write_text(json.dumps(rec, indent=1))
    for r in rows:
        print(f"{r['rank']:>2} {r['scenario']:<48} {r['upstream'][0]:>2}/{r['upstream'][1]:<2} "
              f"l6 {r['launch6'][0]}/{r['launch6'][1]}  A {r['cell_a'][0]}/{r['cell_a'][1]}  {r['tier']}")


if __name__ == "__main__":
    main()
