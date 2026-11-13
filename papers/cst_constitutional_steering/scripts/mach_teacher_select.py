#!/usr/bin/env python3
"""The teacher check's state selection and power at the size of record (CST Amendment 47 item 2).

    /opt/homebrew/bin/python3.13 mach_teacher_select.py STATES_JSONL

One seeded order per game over its eligible states (eligible by labels only); the first 60 are the
base and GOOD_BEHAVIOR set, the first 30 of those the constitution's set (nested). Writes
``data/mach_teacher_states_v1.json`` (ids, the two caps, the seed, a hash) and prints the power by
game bootstrap: each state's violating-choice probability its game's saved base rate, an effect
that does not vary by game (an assumption), 2,000 game resamples; MDE = 2.80 x SE; the detection
power at 0.042 (the ethics prompt's effect on other models at this base rate) beside.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

CST = Path(__file__).resolve().parent.parent
SEED, CAP_WIDE, CAP_NESTED, K = "20261113|cst_teacher_states", 60, 30, 2
RATE = {"base": 2100.0, "good_behavior": 2100.0, "constitution": 840.0}
OVERHEAD_H = 0.6


def sid(s: dict) -> str:
    return f"{s['game']}|{s['episode']}|{s['step']}"


def power(sel: dict, p_game: dict, k: int, rng, reps: int = 200) -> float:
    games = sorted(sel)
    ses = []
    for _ in range(reps):
        d = {g: (rng.binomial(k, p_game[g], len(v)) - rng.binomial(k, p_game[g], len(v))) / k
             for g, v in sel.items()}
        bs = [np.mean(np.concatenate([d[g] for g in rng.choice(games, len(games))]))
              for _ in range(100)]
        ses.append(np.std(bs))
    return float(np.mean(ses))


def main() -> None:
    st = [json.loads(x) for x in open(sys.argv[1])]
    st = [s for s in st if s["eligible"]]
    by = defaultdict(list)
    for s in st:
        by[s["game"]].append(s)
    order = {g: sorted(v, key=sid) for g, v in by.items()}
    for g in order:
        random.Random(f"{SEED}|{g}").shuffle(order[g])
    wide = {g: v[:CAP_WIDE] for g, v in order.items()}
    nested = {g: v[:CAP_NESTED] for g, v in order.items()}
    ids_w = sorted(sid(s) for v in wide.values() for s in v)
    ids_n = sorted(sid(s) for v in nested.values() for s in v)
    assert set(ids_n) <= set(ids_w)
    blob = json.dumps({"wide": ids_w, "nested": ids_n}, sort_keys=True)
    rec = {"rule": "Amendment 47 item 2 (the teacher check's states, nested)", "seed": SEED,
           "cap_wide": CAP_WIDE, "cap_nested": CAP_NESTED, "k": K, "games": len(by),
           "n_wide": len(ids_w), "n_nested": len(ids_n),
           "ids_sha256": hashlib.sha256(blob.encode()).hexdigest(),
           "source": "the base arm of the finalist MACHIAVELLI run, eligible by labels only",
           "wide": ids_w, "nested": ids_n}
    out = CST / "data/mach_teacher_states_v1.json"
    out.write_text(json.dumps(rec, indent=0) + "\n")
    p_game = {g: float(np.mean([s["saved_violating"] for s in v])) for g, v in by.items()}
    rng = np.random.default_rng(0)
    se_w, se_n = power(wide, p_game, K, rng), power(nested, p_game, K, rng)
    hours = OVERHEAD_H + K * (len(ids_w) * (1 / RATE["base"] + 1 / RATE["good_behavior"])
                              + len(ids_n) / RATE["constitution"])

    def pw(se: float, eff: float = 0.042) -> float:
        return 0.5 * (1 + math.erf((eff / se - 1.959964) / math.sqrt(2)))

    print(f"{len(by)} games; wide N {len(ids_w)}, nested N {len(ids_n)}; "
          f"sha {rec['ids_sha256'][:16]}")
    print(f"base vs GOOD_BEHAVIOR (wide, k {K}): MDE {2.80 * se_w:.3f}, "
          f"power at 0.042 {pw(se_w):.2f}")
    print(f"constitution vs base (nested, k {K}): MDE {2.80 * se_n:.3f}, "
          f"power at 0.042 {pw(se_n):.2f}")
    print(f"price about {hours:.2f} A100-h (ceiling 6)")


if __name__ == "__main__":
    main()
