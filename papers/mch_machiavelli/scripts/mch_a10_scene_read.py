"""MCH-A10's scene-level read (zero GPU): where Qwen's ethics-prompt disutility rise comes from.

Replays the Qwen3-30B-A3B-Instruct-2507 cell's episodes in the three games that carry 81% of the
disutility difference, from each episode's seed and saved actions, on the hash-verified engine.
Each replay must reproduce the saved per-episode disutility count. Every disutility event
(upstream's ``utility.suffering``: others at -3 with confidence above 0.5) is attributed to the
scene that produced it; GoodCoTAgent's excess over CoTAgent is split into scenes only it reached
and scenes both agents reached, and the first step where the two agents' paths part is recorded.

Tracked output (``outputs/mch_a10_scene_read.json``): aggregates only. Scene-level detail goes to
the gitignored ``outputs_raw/`` (per-scene labels would rebuild the authors' table; MCH_PLAN §9).
"""

from __future__ import annotations

import collections
import gzip
import json
import os
import sys
from pathlib import Path

MCH = Path(__file__).resolve().parents[1]
REPO = MCH.parents[1]
sys.path.insert(0, str(REPO))
CELL = MCH / "outputs" / "newmodel" / "cell_qwen3_30b_instruct"
OUT = MCH / "outputs" / "mch_a10_scene_read.json"
RAW = MCH / "outputs_raw" / "mch_a10_scenes.json"
GAMES = ("kung-fu", "versus-the-lost-ones", "versus-deathscapes")
AGENTS = ("CoTAgent", "GoodCoTAgent")


def main() -> None:
    from deepsteer.datasets.machiavelli import fetch_machiavelli

    src = fetch_machiavelli(verify_bundle=False)
    decs = collections.defaultdict(list)
    for line in gzip.open(CELL / "decisions.jsonl.gz", "rt"):
        r = json.loads(line)
        if r["game"] in GAMES:
            decs[(r["agent"], r["game"], r["episode"])].append(r)
    saved = {(c["agent"], c["game"], c["episode"]): c["counts"]
             for c in map(json.loads, (CELL / "_counts.jsonl").read_text().splitlines())}
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.config import MachiavelliConfig
    from machiavelli.game.machiavelli_env import MachiavelliEnv, Trajectory
    from machiavelli.utils import random_seed

    thr = MachiavelliConfig().confidence_threshold
    out, raw = {"games": {}}, {}
    for g in GAMES:
        env = MachiavelliEnv(game=g)
        paths, events, n_ep = {}, {}, {}
        for a in AGENTS:
            ev = collections.Counter()
            for e in (0, 1):
                rows = sorted(decs[(a, g, e)], key=lambda r: r["step"])
                seed = e + 2417
                random_seed(seed)
                obs, info = env.reset(seed=seed)
                traj = Trajectory(g, agent_name=a)
                path, rew = [], 0
                for r in rows:
                    traj.add(obs, r["action"], rew, info)
                    path.append("|".join(info["game_state"]["current_nodes"]))
                    obs, rew, done, info = env.step(r["action"])
                    if done:
                        break
                traj.add(obs, -1, rew, info)
                path.append("|".join(info["game_state"]["current_nodes"]))
                got = traj.get_mach_counts().get("utility.suffering", 0)
                want = saved[(a, g, e)].get("utility.suffering", 0)
                assert got == want, (a, g, e, got, want)  # the replay is the saved episode
                for i, inf in enumerate(traj.info_list):
                    ann = inf["annotations"]
                    if ann is not None and ann["utility"]["others"]["-3"] > thr:
                        ev[path[i]] += 1
                paths[(a, e)] = path
            events[a] = ev
            n_ep[a] = sum(events[a].values())
        only_good = {s: c for s, c in events["GoodCoTAgent"].items() if s not in events["CoTAgent"]}
        only_cot = {s: c for s, c in events["CoTAgent"].items() if s not in events["GoodCoTAgent"]}
        shared = set(events["GoodCoTAgent"]) & set(events["CoTAgent"])
        shared_delta = sum(events["GoodCoTAgent"][s] - events["CoTAgent"][s] for s in shared)
        diverge = {}
        for e in (0, 1):
            p, q = paths[("CoTAgent", e)], paths[("GoodCoTAgent", e)]
            diverge[e] = next((i for i, (x, y) in enumerate(zip(p, q)) if x != y), None)
        out["games"][g] = {
            "suffering_events": n_ep,
            "excess_good_minus_cot": n_ep["GoodCoTAgent"] - n_ep["CoTAgent"],
            "from_scenes_only_good_reached": sum(only_good.values()),
            "from_scenes_only_cot_reached": -sum(only_cot.values()),
            "from_scenes_both_reached": shared_delta,
            "distinct_suffering_scenes": {a: len(events[a]) for a in AGENTS},
            "shared_suffering_scenes": len(shared),
            "first_divergence_step": diverge,
            "top3_only_good_share": (sum(sorted(only_good.values(), reverse=True)[:3])
                                     / max(1, sum(only_good.values()))),
        }
        raw[g] = {"only_good": dict(collections.Counter(only_good).most_common(10)),
                  "only_cot": dict(collections.Counter(only_cot).most_common(10)),
                  "shared": {s: [events["CoTAgent"][s], events["GoodCoTAgent"][s]]
                             for s in shared}}
    tot = {k: sum(v[k] for v in out["games"].values())
           for k in ("excess_good_minus_cot", "from_scenes_only_good_reached",
                     "from_scenes_only_cot_reached", "from_scenes_both_reached")}
    out["total"] = tot
    out["replays_match_saved_counts"] = True
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_text(json.dumps(raw, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
