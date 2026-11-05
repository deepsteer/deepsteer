#!/usr/bin/env python3
"""CST T0.1: the T_conf artifact with provenance (CST_PLAN Amendment 1, D7).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_tconf.py

Members today are document-internal (Claude's conflict pass on the v2 tenets, the narrowed rule:
explicit, specific gating or competing directions at overlapping conditions); each pair carries
codex's check when it has run. Panel-detected members join at S0 under the root-rate rule
(``deepsteer.judges.root_rate_tags``). Writes ``data/tconf_v1.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deepsteer.constitutions import TenetSet, document_conflicts, tconf_record  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"


def main() -> None:
    r = json.loads((CST / "outputs" / "t01_conflicts_v2.json").read_text())
    out: dict = {"rule": "CST_PLAN §2.2 as narrowed at G0; Amendment 1 D7 (root-rate rule for the "
                         "panel half, at S0)", "tenets": "v2", "docs": {}}
    for d in ("C_anth", "C_spec"):
        ts = TenetSet.load(CST / "data" / f"tenets_{d}_v2.json")
        ids = set(ts.ids())
        by = ts.by_id()
        for c in r[d]["conditioned_on"]:
            if c["id"] in ids and c["on"] in ids:
                by[c["id"]].conditioned_on.append(c["on"])
        comp = [tuple(c["pair"].split("|")) for c in r[d]["competing"]]
        comp = [p for p in comp if all(x in ids for x in p)]
        rec = tconf_record(document_conflicts(ts, comp))
        check = {c["pair"]: c["competing"] for c in (r[d].get("codex_check") or [])}
        roots = {i.rstrip("bcdefgh") for i in rec}
        out["docs"][d] = {"members": rec, "n_members": len(rec), "n_tenets": len(ids),
                          "n_roots": len(roots),
                          "codex_check": check or "pending",
                          "competing_pairs": ["|".join(p) for p in comp]}
        print(d, "members", len(rec), "of", len(ids), "roots", len(roots),
              "codex check", "pending" if not check else f"{sum(check.values())}/{len(check)}")
    (CST / "data" / "tconf_v1.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
