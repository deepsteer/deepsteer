#!/usr/bin/env python3
"""The content gate's read (CST Amendment 32 item 9), zero GPU, saved rollouts.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/content_gate_read.py

A0 and A0s at k 4 on `data/content_scenarios_v1.json`'s plain condition (`outputs/pilot_sdf4/gen/
a0`, `gen/a0s`). Per scenario the directed share (the tenet's option, `covariates.directed_option`)
among rollouts with an action, capped excluded; A0s minus A0 with a scenario bootstrap (2,000
draws, seeded). **Registered:** a 95% CI excluding 0 means a content cell exists and the adapters
are read on it next; otherwise the content question waits for the partner's document. Beside it:
capped and unparsed rates, the directed share by tenet kind. Writes
`outputs/content_gate_read.json`.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
GEN = CST / "outputs/pilot_sdf4/gen"
B, SEED = 2000, "20261111|cst_content_gate_read"


def directed() -> dict[str, tuple[str, str]]:
    sc = json.loads((CST / "data/content_scenarios_v1.json").read_text())["scenarios"]
    return {s["id"]: (s["covariates"]["directed_option"], s["covariates"]["tenet_kind"])
            for s in sc}


def per_scenario(arm: str, d: dict) -> tuple[dict[str, list[int]], dict]:
    u: dict[str, list[int]] = defaultdict(list)
    n = {"rollouts": 0, "capped": 0, "unparsed": 0}
    for f in sorted((GEN / arm).glob("content_plain_*.jsonl")):
        for x in open(f):
            r = json.loads(x)
            n["rollouts"] += 1
            if r["hit_cap"]:
                n["capped"] += 1
                continue
            if r["option"] is None:
                n["unparsed"] += 1
                continue
            u[r["scenario_id"]].append(int(r["option"] == d[r["scenario_id"]][0]))
    return dict(u), n


def share(u: dict, ids: list[str]) -> float:
    return statistics.mean(statistics.mean(u[s]) for s in ids)


def main() -> None:
    d = directed()
    a0, n0 = per_scenario("a0", d)
    a0s, n1 = per_scenario("a0s", d)
    ids = sorted(set(a0) & set(a0s))
    rng = random.Random(SEED)
    pt = share(a0s, ids) - share(a0, ids)
    bs = sorted(share(a0s, s) - share(a0, s) for s in ([rng.choice(ids) for _ in ids]
                                                       for _ in range(B)))
    ci = [bs[int(0.025 * B)], bs[int(0.975 * B) - 1]]
    by = defaultdict(list)
    for s in ids:
        by[d[s][1]].append(s)
    res = {"rule": "Amendment 32 item 9", "n_scenarios": len(ids), "a0": share(a0, ids),
           "a0s": share(a0s, ids), "a0s_minus_a0": pt, "ci95": ci,
           "content_cell_exists": ci[0] > 0 or ci[1] < 0, "counts": {"a0": n0, "a0s": n1},
           "by_kind": {k: {"n": len(v), "a0": share(a0, v), "a0s": share(a0s, v)}
                       for k, v in by.items()}, "bootstrap": {"draws": B, "seed": SEED}}
    (CST / "outputs/content_gate_read.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
