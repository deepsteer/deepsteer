#!/usr/bin/env python3
"""The single-decision teacher check's power table (CST Amendment 46 item C4), zero GPU.

    /opt/homebrew/bin/python3.13 mach_state_power.py STATES_JSONL

From the extracted eligible states: each state's violating-choice probability is taken as its game's
saved base rate (the measured variance; the saved decision is not the base read). For a cap of m
states per game (a seeded draw) and k samples per arm per state, two arms are simulated at the same
states under no difference, the paired mean difference is bootstrapped **by game** (2,000 draws),
and the MDE is 2.80 x its SE (80% power, two-sided 5%). Price per state-sample from this run's
rates: base and GOOD_BEHAVIOR at about 2,100 decisions/h (the MACHIAVELLI engine's plateau), the
constitution in context at about 840/h (A0s's 300 tokens/s at an assumed 1,290 tokens per
decision), plus a 0.6 h pod overhead.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

import numpy as np

RATE = {"base": 2100.0, "good_behavior": 2100.0, "constitution": 840.0}
OVERHEAD_H, CEILING = 0.6, 5.0


def main() -> None:
    st = [json.loads(x) for x in open(sys.argv[1]) if json.loads(x)["eligible"]]
    by = defaultdict(list)
    for s in st:
        by[s["game"]].append(s)
    p_game = {g: float(np.mean([s["saved_violating"] for s in v])) for g, v in by.items()}
    games = sorted(by)
    print(f"{len(st)} eligible states in {len(games)} games; saved rate by game "
          f"{min(p_game.values()):.2f}-{max(p_game.values()):.2f}")
    rng = np.random.default_rng(0)
    for m in (10, 20, 30, 50, 80):
        sel = {g: random.Random(f"cap|{g}|{m}").sample(by[g], min(m, len(by[g]))) for g in games}
        n = sum(len(v) for v in sel.values())
        for k in (1, 2, 4):
            h = OVERHEAD_H + n * k * sum(1 / r for r in RATE.values())
            ses = []
            for _ in range(200):  # simulated arms under no difference, then a game bootstrap
                d = {g: (rng.binomial(k, p_game[g], len(v))
                         - rng.binomial(k, p_game[g], len(v))) / k for g, v in sel.items()}
                bs = [np.mean(np.concatenate([d[g] for g in rng.choice(games, len(games))]))
                      for _ in range(100)]
                ses.append(np.std(bs))
            mde = 2.80 * float(np.mean(ses))
            print(f"cap {m:3d}/game  N {n:5d}  k {k}  MDE {mde:.3f}"
                  f"  price {h:5.1f} A100-h{'  over the ceiling' if h > CEILING else ''}")


if __name__ == "__main__":
    main()
