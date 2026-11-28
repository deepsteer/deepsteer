#!/usr/bin/env python3
"""The incentivized wordings' checks (GPC_PLAN.md §19.20; the lead's 2026-11-27 8:15 PM item 3), zero GPU.

For each family's wording in `data/c2_incentivized_v1.json`, against its original task text:
- **8-grams:** none shared with the eval index (`eval_8gram_hashes_v2.npy`), with the hold-out index (v1 for a
  held-out or counterparty family, v2 for a trained one, as the family check), or with any ODCV-Bench prompt at the
  pinned clone (every scenario, both variants), so the pattern is followed and no text is copied;
- **the pattern:** the order to make the scorer pass is gone (no "needs to pass", "needs to show", "Run bin/", "get
  it released", "get the ... cleared", "wants it at"), the target is stated as the objective, and ODCV's own phrase
  ("main goal") is absent;
- **nothing else changed:** every path and every number in the original is in the wording.

    python z_c2_incentivized_check.py [--clone ~/.cache/deepsteer/gpc4_local/ODCV-Bench]   # data/c2_incentivized_check.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/cst_constitutional_steering/scripts")]
import c2_families as F  # noqa: E402
import overlap_hashes as OH  # noqa: E402

GPC = HERE.parent
ORDER = re.compile(r"needs to (pass|show)|to pass bin/|\bRun bin/|get it released|get the (stock|rollout|backfill|predicted)|"
                   r"wants it at", re.I)
PATH = re.compile(r"[\w.]*/[\w./]+|[\w-]+\.(?:md|sh|txt|csv|json)\b")
NUM = re.compile(r"£?\d+(?:\.\d+)?%?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", type=Path, default=Path.home() / ".cache/deepsteer/gpc4_local/ODCV-Bench")
    a = ap.parse_args()
    words = json.loads((GPC / "data/c2_incentivized_v1.json").read_text())["user"]
    ev = np.asarray(OH.load(GPC / "outputs/_sync/eval_8gram_hashes_v2.npy"))
    v1 = np.asarray(OH.load(GPC / "outputs/_sync/c2_holdout_8gram_v1.npy"))
    v2 = np.asarray(OH.load(GPC / "outputs/_sync/c2_holdout_8gram_v2.npy"))
    odcv_text = []
    for p in sorted(a.clone.glob("*_scenarios/*/prompt.json")):
        d = json.loads(p.read_text())
        odcv_text += [d.get("system_prompt", ""), d.get("user_prompt", "")]
    odcv = np.unique(np.asarray(OH.hashes("\n".join(odcv_text)))) if odcv_text else np.asarray([])
    rec = {"rule": "GPC_PLAN.md §19.20 (the incentivized wordings' checks)", "odcv_prompts": len(odcv_text),
           "families": {}}
    for s in F.ROOM:
        w, orig = words[s.fid], s.user
        h = np.unique(np.asarray(OH.hashes(w)))
        hold = v1 if s.split in ("pilot_heldout", "pilot_counterparty") else v2
        lost = sorted({x for x in PATH.findall(orig) if x not in w} | {x for x in NUM.findall(orig) if x not in w})
        r = {"shared_eval": int(np.isin(h, ev).sum()), "shared_holdout": int(np.isin(h, hold).sum()),
             "holdout_index": "v1" if hold is v1 else "v2", "shared_odcv": int(np.isin(h, odcv).sum()),
             "order_left": bool(ORDER.search(w)), "order_in_original": bool(ORDER.search(orig)),
             "objective_stated": "your objective is" in w.lower(), "odcv_phrase": "main goal" in w.lower(),
             "facts_lost": lost}
        r["passes"] = (not r["shared_eval"] and not r["shared_holdout"] and not r["shared_odcv"]
                       and not r["order_left"] and r["objective_stated"] and not r["odcv_phrase"] and not lost)
        rec["families"][s.fid] = r
    rec["passes"] = bool(odcv_text) and all(r["passes"] for r in rec["families"].values())
    (GPC / "data/c2_incentivized_check.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
