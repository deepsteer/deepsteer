#!/usr/bin/env python3
"""Amendment 33 item 3: the panel's content reads with selection and comparison on stated cells
(zero GPU; S0a's and S0b's saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/content_null_reads.py

The directed share is the share of rollouts with an option (not capped) taking the option of
record the document directs (`directed_option_of_record`). Two reads, scenario bootstrap (2,000
draws, seeded):

- **the lead's items:** the 458 clear items where A0's directed share on the **user** cell is at
  most 0.5; on them A0s minus A0 on the **plain** cell (selection and comparison on different
  cells), with A0n minus A0 beside;
- **the 177-item read** (Amendment 32 item 9's note): the panel items with a directed option of
  record where A0 never takes it on the **plain** cell, compared on the same plain-cell rollouts:
  biased upward by construction (A0 is 0 on them), so A0n minus A0 on the same items is its null.

Writes `outputs/content_null_reads.json`.
"""

from __future__ import annotations

import glob
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import s0a_analysis as A  # noqa: E402

CST = HERE.parent
DIRS = {"a0": "outputs/s0a/gen/a0", "a0s": "outputs/s0a/gen/a0s", "a0n": "outputs/s0b/gen/a0n"}
B, SEED = 2000, "20261111|cst_content_null_reads"


def per_scenario(arm: str, cond: str, lab: dict) -> dict[str, list[int]]:
    u: dict[str, list[int]] = defaultdict(list)
    for f in sorted(glob.glob(str(CST / DIRS[arm] / f"panel_{cond}_[0-9]*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            rec = lab.get(r["scenario_id"])
            if (r["condition"] != cond or not rec or rec.get("set", "panel") != "panel"
                    or rec.get("directed_option_of_record") is None or r["hit_cap"]
                    or r["option"] is None):
                continue
            u[r["scenario_id"]].append(int(r["option"] == rec["directed_option_of_record"]))
    return dict(u)


def mean(u: dict, ids: list[str]) -> float:
    return statistics.mean(statistics.mean(u[s]) for s in ids)


def contrast(a: dict, b: dict, ids: list[str]) -> dict:
    ids = [s for s in ids if a.get(s) and b.get(s)]
    rng = random.Random(SEED)
    bs = sorted(mean(a, d) - mean(b, d) for d in ([rng.choice(ids) for _ in ids]
                                                  for _ in range(B)))
    return {"delta": mean(a, ids) - mean(b, ids), "ci95": [bs[int(0.025 * B)],
                                                           bs[int(0.975 * B) - 1]], "n": len(ids)}


def main() -> None:
    lab = A.labels()["record"]
    plain = {arm: per_scenario(arm, "plain", lab) for arm in DIRS}
    user = per_scenario("a0", "user", lab)
    clear = {s for s, r in lab.items() if r["stratum_of_record"] == "clear_refuse"
             and r.get("set", "panel") == "panel"}
    lead = sorted(s for s in clear if user.get(s) and statistics.mean(user[s]) <= 0.5)
    never = sorted(s for s in plain["a0"] if max(plain["a0"][s]) == 0)
    res = {"rule": "Amendment 33 item 3", "bootstrap": {"draws": B, "seed": SEED},
           "lead_items": {"selected_on": "A0, user cell, directed share <= 0.5, clear items",
                          "compared_on": "plain cell", "n_selected": len(lead),
                          "a0_plain": mean(plain["a0"], [s for s in lead if plain["a0"].get(s)]),
                          "a0s_minus_a0": contrast(plain["a0s"], plain["a0"], lead),
                          "a0n_minus_a0": contrast(plain["a0n"], plain["a0"], lead)},
           "never_items": {"selected_on": "A0, plain cell, directed share 0 (all panel items with "
                                         "a directed option of record)",
                           "compared_on": "the same plain-cell rollouts (biased upward)",
                           "n_selected": len(never),
                           "a0s_minus_a0": contrast(plain["a0s"], plain["a0"], never),
                           "a0n_minus_a0_null": contrast(plain["a0n"], plain["a0"], never)}}
    (CST / "outputs/content_null_reads.json").write_text(json.dumps(res, indent=1) + "\n")
    for name in ("lead_items", "never_items"):
        r = res[name]
        print(f"== {name}: n {r['n_selected']}, selected on {r['selected_on']}; compared on "
              f"{r['compared_on']}")
        for k, v in r.items():
            if isinstance(v, dict):
                print(f"   {k:18s} {v['delta']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
                      f" n {v['n']}")


if __name__ == "__main__":
    main()
