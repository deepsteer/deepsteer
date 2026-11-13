#!/usr/bin/env python3
"""MACH_THROUGHPUT_NOTE.md's screen check (CST Amendment 45 item 3), zero GPU.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/mach_screen_check.py

On MCH's ethics-prompt cells, the ethics-prompt agent minus the plain agent paired by game and seed:
the 17 shortest games (the note's screen) against the full read and the long games, with MCH's
estimator of record (the mean over games of per-game means, episodes resampled within game, games
fixed) and a bootstrap over games beside it. Run from the repository root.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "papers/mch_machiavelli/scripts")
import mch_thinking_analyze as MA  # noqa: E402
from mch_newmodel_analyze import stat  # noqa: E402

OUTPUTS = Path("papers/mch_machiavelli/outputs")
SCREEN = ["avatar-of-the-wolf", "pendragon-rising", "ratings-war", "t-rex-time-machine",
          "death-collector", "versus-the-lost-ones", "hero-project-redemption-season",
          "undercover-agent", "hero-of-kendrickstone", "kung-fu", "midsummer-nights-choice",
          "tower-behind-the-moon", "battlemage", "versus-deathscapes", "mysteries-of-baroque",
          "trials-of-the-thief-taker", "seventh-sea-a-pirates-pact"]
THINK = ("ThinkCoTAgent", "ThinkGoodCoTAgent")
PLAIN = ("CoTAgent", "GoodCoTAgent")
CELLS = {
    "qwen3_30b_instruct (cell, 30 games)": (["newmodel/cell_qwen3_30b_instruct"], PLAIN),
    "gpt_oss_20b thinking (cell, 20 games)": (["thinking/cell_gpt_oss_20b"], THINK),
    "gpt_oss_20b thinking (pilot+cell, 30 games)":
        (["thinking/pilot_gpt_oss_20b", "thinking/cell_gpt_oss_20b"], THINK),
    "gpt_oss_20b 1,024 (cell, 20 games)": (["newmodel/cell_gpt_oss_20b"], PLAIN),
    "gpt_oss_20b 1,024 (pilot+cell, 30 games)":
        (["newmodel/pilot_1024", "newmodel/cell_gpt_oss_20b"], PLAIN),
}


def ci(x) -> list[float]:
    return [round(float(v), 1) for v in np.percentile(x, [2.5, 97.5])]


def game_boot(groups: dict, games: list[str], rng, n: int = 10000) -> np.ndarray:
    m = np.array([np.mean(groups[g]) for g in games])
    return np.array([m[rng.integers(0, len(m), len(m))].mean() for _ in range(n)])


def main() -> None:
    for name, (dirs, (none, good)) in CELLS.items():
        eps = [e for d in dirs for e in MA.jl(OUTPUTS / d / "episodes.jsonl")
               if e["status"] == "done"]
        a = [e for e in eps if e["agent"] == good]
        b = [e for e in eps if e["agent"] == none]
        print(f"\n{name}")
        for key in ("violations", "reward"):
            d = MA.paired(a, b, MA.H[key])  # the ethics-prompt agent minus the plain agent
            games = sorted(d)
            short = [g for g in games if g in SCREEN]
            long_ = [g for g in games if g not in SCREEN]
            rng = np.random.default_rng(0)
            full = stat(d, rng)
            s = stat({g: d[g] for g in short}, rng)
            lg = stat({g: d[g] for g in long_}, rng)
            gap = s["draws"] - lg["draws"]  # games fixed (MCH's of record); disjoint games
            rg = np.random.default_rng(0)
            gb_s, gb_l = game_boot(d, short, rg), game_boot(d, long_, rg)
            print(f"  {key:10s} full {full['point']:+6.1f} {ci(full['draws'])} | "
                  f"screen ({len(short)}) {s['point']:+6.1f} {ci(s['draws'])} | "
                  f"long ({len(long_)}) {lg['point']:+6.1f} {ci(lg['draws'])} | "
                  f"short-long {s['point'] - lg['point']:+6.1f} {ci(gap)} "
                  f"[by game: {ci(gb_s - gb_l)}; screen by game {ci(gb_s)}]")


if __name__ == "__main__":
    main()
