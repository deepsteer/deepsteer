#!/usr/bin/env python3
"""Amendment 26 item 2: the relaunched SDF pod's six reads, relabeled as reads of the base model
(CST-A22), as an empirical null on the operator cell (zero GPU, saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/sdf_base_null.py

Each read generated the operator cell at k 2 on the 458 clear items with its own seeds, so the six
are independent k-2 draws of the base model. On rollouts with an action (capped excluded), the
ordered-option share per scenario; per read the scenario-paired contrast against the saved A0
(S0a, k 4) with a scenario bootstrap; across reads the spread of those contrasts and of the 15
read-against-read contrasts (both at k 2, the rung-against-control comparison); beside them the
computed pilot MDEs (`t04d_ordered_power.json` `pilot_k2`: 0.042 against A0 at k 4, 0.049 rung
against control). Writes `outputs/sdf_base_null.json`.
"""

from __future__ import annotations

import glob
import itertools
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
READS = ("d_doc_1M_rec", "d_doc_1M", "r0", "d_chat_1M_rec", "d_doc_03M_rec", "d_doc_03M")
B, SEED = 2000, "20261110|cst_sdf_base_null"


def per_scenario(pattern: str, lab: dict) -> dict[str, list[int]]:
    u: dict[str, list[int]] = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        for x in open(f):
            r = json.loads(x)
            if r["condition"] != "operator" or r["hit_cap"] or r["option"] is None:
                continue
            rec = lab["record"][r["scenario_id"]]
            if rec["stratum_of_record"] != "clear_refuse" or rec.get("set", "panel") != "panel":
                continue
            u[r["scenario_id"]].append(int(r["option"] == rec["ordered_option"]))
    return dict(u)


def share(u: dict, ids: list[str]) -> float:
    return statistics.mean(statistics.mean(u[s]) for s in ids)


def contrast(a: dict, b: dict, ids: list[str]) -> dict:
    rng = random.Random(SEED)
    pt = share(a, ids) - share(b, ids)
    bs = sorted(share(a, d) - share(b, d) for d in ([rng.choice(ids) for _ in ids]
                                                    for _ in range(B)))
    return {"delta": pt, "ci95": [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]}


def main() -> None:
    lab = A.labels()
    a0 = per_scenario(str(CST / "outputs/s0a/gen/a0/panel_operator_[0-9]*.jsonl"), lab)
    reads = {t: per_scenario(str(CST / f"outputs/pilot_sdf/gen/{t}/panel_operator_*.jsonl"), lab)
             for t in READS}
    ids = sorted(set(a0).intersection(*reads.values()))
    vs_a0 = {t: contrast(u, a0, ids) for t, u in reads.items()}
    pairs = {f"{x}-{y}": contrast(reads[x], reads[y], ids)["delta"]
             for x, y in itertools.combinations(READS, 2)}
    mde = json.loads((CST / "outputs/t04d_ordered_power.json").read_text())["pilot_k2"]
    d0 = [v["delta"] for v in vs_a0.values()]
    dp = list(pairs.values())
    res = {"rule": "CST Amendment 26 item 2", "label": "reads of the base model (CST-A22)",
           "n_items": len(ids), "a0_share": share(a0, ids),
           "read_shares": {t: share(u, ids) for t, u in reads.items()},
           "vs_saved_a0": vs_a0,
           "vs_saved_a0_spread": {"min": min(d0), "max": max(d0), "sd": statistics.stdev(d0),
                                  "max_abs": max(map(abs, d0))},
           "read_vs_read": pairs,
           "read_vs_read_spread": {"sd": statistics.stdev(dp), "max_abs": max(map(abs, dp))},
           "pilot_mde": {"rung_vs_A0_k4": mde["rung_vs_A0_k4_null"],
                         "rung_vs_control_k2": mde["rung_vs_control_k2_null"]},
           "expected_sd_under_null": {"vs_a0": mde["rung_vs_A0_k4_null"] / 2.8,
                                      "read_vs_read": mde["rung_vs_control_k2_null"] / 2.8},
           "bootstrap": {"draws": B, "seed": SEED, "unit": "scenario"}}
    (CST / "outputs/sdf_base_null.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("read_vs_read",)}, indent=1))


if __name__ == "__main__":
    main()
