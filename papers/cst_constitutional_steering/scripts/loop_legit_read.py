#!/usr/bin/env python3
"""Amendment 44 item 4 (zero GPU): the held-out legitimate-order read of record on the loop pod.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/loop_legit_read.py [DIR]

On the 36 held-out legitimate-order scenarios labeled ``comply`` by the two-family rule
(``data/heldout_legit_labels_v1.json``, Amendment 39 item 3), each config's share following the
order at k 4 against the pod's session base, and config 2 minus config 1, paired by scenario
(scenario bootstrap, ``pctx_followups.paired``). Writes ``DIR/heldout_legit_comply.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import pctx_followups as F  # noqa: E402

CST = HERE.parent
TAGS = ("c1_repro", "c2_rule_right")


def read(out: Path) -> dict:
    rec = json.loads((CST / "data/heldout_legit_labels_v1.json").read_text())["record"]
    comply = sorted(s for s, v in rec.items() if v == "comply")
    lab = F.labels(out / "heldout_act.jsonl")
    u = {t: F.follow([json.loads(x) for x in open(out / "heldout" / t / "chunk_000.jsonl")], lab)
         for t in ("base",) + TAGS}
    res = {"rule": "Amendment 44 item 4 (the comply items, the read of record)",
           "comply_items": len(comply),
           "base_follow": F.mean(u["base"], [s for s in comply if u["base"].get(s)])}
    for t in TAGS:
        res[t] = {"follow": F.mean(u[t], [s for s in comply if u[t].get(s)]),
                  "vs_base": F.paired(u[t], u["base"], comply)}
    res["c2_minus_c1"] = F.paired(u["c2_rule_right"], u["c1_repro"], comply)
    (out / "heldout_legit_comply.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else CST / "outputs/loop_v1"
    print(json.dumps(read(d), indent=1))
