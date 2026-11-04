"""MCH-A11's trace-length read (zero GPU, descriptive; MCH_PLAN §5 Amendment 11, ANOMALIES MCH-A11).

With thinking on, ThinkGoodCoTAgent's traces ran about half as long as ThinkCoTAgent's in the pilot.
This read separates the registered rivals on GPT-OSS-20B's full thinking-on cell (120 episodes):
- R_shortcut (the ethics paragraph shortens deliberation over the same choice) against R_content
  (the agents reach different scenes, and the prompted agent's scenes need less reasoning): trace
  length at scenes BOTH agents reached in the same game, against all decisions;
- R_cap (capped traces inflate the gap): the same on attempts that did not hit the 4,096 cap.
Each episode is replayed from its seed and saved actions on the hash-verified engine and must
reproduce its saved harm counts. A decision's trace length is its first attempt's (retries repeat
the prompt). Beside it, by outcome: trace length at decisions whose next scene carries a labelled
violation, against those whose next scene does not.

Tracked output (``outputs/mch_a11_trace_read.json``): aggregates only. Per-scene detail goes to the
gitignored ``outputs_raw/`` (no scene id with a per-scene value in tracked files; MCH_PLAN §9).
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

MCH = Path(__file__).resolve().parents[1]
REPO = MCH.parents[1]
sys.path.insert(0, str(REPO))
TH = MCH / "outputs" / "thinking"
DIRS = ("pilot_gpt_oss_20b", "cell_gpt_oss_20b")
OUT = MCH / "outputs" / "mch_a11_trace_read.json"
RAW = MCH / "outputs_raw" / "mch_a11_scenes.json"
AGENTS = ("ThinkCoTAgent", "ThinkGoodCoTAgent")
N_BOOT = 10_000


def ratio_ci(per_game: dict[str, dict[str, list[float]]], rng) -> dict:
    """Good / CoT ratio of mean trace length, games resampled (10,000 draws, seed 0)."""
    games = sorted(g for g, v in per_game.items() if all(v[a] for a in AGENTS))
    sums = np.array([[sum(per_game[g][a]), len(per_game[g][a])] for g in games for a in AGENTS])
    sums = sums.reshape(len(games), 2, 2)

    def r(idx):
        s = sums[idx].sum(0)
        return (s[1, 0] / s[1, 1]) / (s[0, 0] / s[0, 1])

    draws = [r(rng.integers(0, len(games), len(games))) for _ in range(N_BOOT)]
    s = sums.sum(0)
    return {"games": len(games), "mean_chars": {a: float(s[i, 0] / s[i, 1])
                                                for i, a in enumerate(AGENTS)},
            "decisions": {a: int(s[i, 1]) for i, a in enumerate(AGENTS)},
            "ratio_good_over_cot": float(r(np.arange(len(games)))),
            "ratio_ci95": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def main() -> None:
    from deepsteer.datasets.machiavelli import fetch_machiavelli

    src = fetch_machiavelli(verify_bundle=False)
    decs = collections.defaultdict(list)
    for d in DIRS:
        for line in gzip.open(TH / d / "decisions.jsonl.gz", "rt"):
            r = json.loads(line)
            decs[(r["agent"], r["game"], r["episode"])].append(r)
    saved = {}
    for d in DIRS:
        for c in map(json.loads, (TH / d / "_counts.jsonl").read_text().splitlines()):
            saved[(c["agent"], c["game"], c["episode"])] = c["counts"]
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.config import MachiavelliConfig
    from machiavelli.game.machiavelli_env import MachiavelliEnv, Trajectory
    from machiavelli.utils import random_seed

    thr = MachiavelliConfig().confidence_threshold
    rows = []  # (agent, game, node, first-attempt chars, first attempt capped, violation next)
    replays_ok = 0
    for game in sorted({g for _, g, _ in decs}):
        env = MachiavelliEnv(game=game)
        for (a, g, e), ds in sorted(decs.items()):
            if g != game:
                continue
            ds = sorted(ds, key=lambda r: r["step"])
            seed = e + 2417
            random_seed(seed)
            obs, info = env.reset(seed=seed)
            traj, rew, nodes = Trajectory(g, agent_name=a), 0, []
            for r in ds:
                traj.add(obs, r["action"], rew, info)
                nodes.append("|".join(info["game_state"]["current_nodes"]))
                obs, rew, done, info = env.step(r["action"])
                if done:
                    break
            traj.add(obs, -1, rew, info)
            got = {k: float(v) for k, v in traj.get_mach_counts().items()}
            want = saved[(a, g, e)]
            assert all(abs(got.get(k, 0) - v) < 1e-9 for k, v in want.items()), (a, g, e)
            replays_ok += 1
            for i, r in enumerate(ds[: len(nodes)]):
                ann = traj.info_list[i + 1]["annotations"]
                viol = ann is not None and any(v > thr for v in ann["morality"].values())
                rows.append((a, g, nodes[i], r["trace_chars"][0], r["capped"][0], viol))
    rng = np.random.default_rng(0)
    reached = collections.defaultdict(set)
    for a, g, n, *_ in rows:
        reached[(a, g)].add(n)

    def table(keep) -> dict:
        t: dict = collections.defaultdict(lambda: {a: [] for a in AGENTS})
        for a, g, n, c, cap, v in rows:
            if keep(a, g, n, cap, v):
                t[g][a].append(c)
        return ratio_ci(t, rng)

    def shared(a, g, n):
        return all(n in reached[(b, g)] for b in AGENTS)

    out = {"amendment": "MCH_PLAN.md §5 Amendment 11 (MCH-A11, descriptive)",
           "episodes_replayed": replays_ok, "replays_match_saved_counts": True,
           "trace_length": "first attempt's characters per decision",
           "all_decisions": table(lambda a, g, n, cap, v: True),
           "shared_scenes": table(lambda a, g, n, cap, v: shared(a, g, n)),
           "shared_scenes_uncapped": table(lambda a, g, n, cap, v: shared(a, g, n) and not cap),
           "unshared_scenes": table(lambda a, g, n, cap, v: not shared(a, g, n)),
           "share_of_decisions_at_shared_scenes": {
               a: float(np.mean([shared(b, g, n) for b, g, n, *_ in rows if b == a]))
               for a in AGENTS},
           "by_outcome": {a: {
               "median_chars_next_scene_violation": float(np.median(
                   [c for b, _, _, c, _, v in rows if b == a and v])),
               "median_chars_next_scene_clean": float(np.median(
                   [c for b, _, _, c, _, v in rows if b == a and not v])),
               "decisions_next_scene_violation": int(sum(v for b, *_, v in rows if b == a))}
               for a in AGENTS}}
    per_scene = collections.defaultdict(dict)
    for a, g, n, c, *_ in rows:
        per_scene[f"{g}::{n}"].setdefault(a, []).append(c)
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps({k: {a: [len(v), float(np.mean(v))] for a, v in d.items()}
                               for k, d in per_scene.items() if len(d) == 2}, indent=1) + "\n")
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
