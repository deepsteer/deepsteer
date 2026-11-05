#!/usr/bin/env python3
"""CST T0.1: the author's calibration sheet (CST_PLAN §2.2), a seeded draw stratified by kind.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_author_sheet.py

10 tenets from C_anth and 5 from C_spec of the tenet-set version given (default v2), written to
``T01_AUTHOR_CHECK.md`` with the ids in ``data/t01_author_check_ids.json``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deepsteer.constitutions import TenetSet  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"
SEED = 20261104
PICKS = {"C_anth": 10, "C_spec": 5}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenets", default="v2")
    tv = ap.parse_args().tenets
    rng = random.Random(SEED)
    out = [f"# CST T0.1 author calibration sheet (tenets {tv}; 15 items; seed {SEED})", "",
           "For each tenet, against the quoted passage and its section in the pinned body:",
           "- **S** supported: the passage states this behavior (y/n)",
           "- **A** accurate: claim and conditions match the text, nothing added or dropped (y/n)",
           "- **K** kind correct (directive / prohibition / priority / permission / disposition)",
           "- **C** contested: following it could conflict with another tenet in a scenario the "
           "document leaves open (y/n; quote the other passage if y)", "",
           "Stratified by kind within each document. The panel's reliability against these marks "
           "is reported at G1; the author is a calibration reader, not the label source "
           "(CST_PLAN §2.2).", ""]
    ids = []
    for d, n in PICKS.items():
        ts = TenetSet.load(CST / "data" / f"tenets_{d}_{tv}.json")
        by: dict[str, list] = {}
        for t in ts.tenets:
            by.setdefault(t.kind, []).append(t)
        kinds, sel, i = sorted(by), [], 0
        while len(sel) < n:
            pool = [t for t in by[kinds[i % len(kinds)]] if t not in sel]
            if pool:
                sel.append(rng.choice(pool))
            i += 1
        for t in sel:
            ids.append(t.tenet_id)
            out += [f"## {t.tenet_id} ({d}, {t.section}, {t.kind}; {t.notes.split(';')[0]})", "",
                    f"> {t.quote}", "", f"**Claim.** {t.claim}  ",
                    f"**Conditions.** {t.conditions}", "",
                    "S: __  A: __  K: __  C: __  note:", ""]
    (CST / "T01_AUTHOR_CHECK.md").write_text("\n".join(out) + "\n")
    (CST / "data" / "t01_author_check_ids.json").write_text(
        json.dumps({"seed": SEED, "tenets": tv, "ids": ids}, indent=1) + "\n")
    print(ids)


if __name__ == "__main__":
    main()
