"""MCH-G0 random baseline on CPU (MCH_PLAN.md §5, MCH-G0).

Runs the upstream MACHIAVELLI engine, unmodified, from the pinned checkout and reproduces the
episode loop of upstream ``experiments/run_trajectories.py`` (the path that
``baselines_random.sh`` calls): per episode ``random_seed(idx + 2417)`` and
``env.reset(seed=idx + 2417)``, a uniform random choice per step, the 1,000-entry trajectory cap,
and the final ``add(obs, -1, rew, info)``. Episodes of one game run in order in one environment, as
upstream did; games run in parallel. An engine failure ends that game's run, as upstream's had no
handler (MCH-A5).

The agent is the ``RandomAgent`` of the pin (``np.random.choice(num_choices)``). The paper-era
commit ``9f82117`` ships an empty ``random_agent.py``, so this is the 2024 reconstruction; it
reproduces the paper's random run exactly (``MCH_G0_REPORT.md`` §2).

Writes one JSONL per game: game, episode index, seed, steps, normalized and raw points,
achievement count, the per-episode harm counts of upstream ``Trajectory.get_mach_counts``, the
action indices and rewards, the number of zero-choice states met, and a sha256 of the observation
sequence (a determinism check). No game text and no per-scene labels (MCH_PLAN §9).

Must run under the engine's environment (Python <= 3.11, gym 0.25.1, asteval 0.9.29, numpy < 2).

Usage:
    python mch_g0_random.py --src ~/.cache/deepsteer/mch/src --episodes 10 --out-dir DIR
    python mch_g0_random.py --src ... --games hollywood-visionary --indices 583 584 --out-dir DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

SEED_OFFSET = 2417  # upstream experiments/run_trajectories.py
MAX_TRAJ_LENGTH = 1000  # upstream run_trajectories default

GAMES_TEST = [
    "avatar-of-the-wolf", "battlemage", "cliffhanger", "death-collector",
    "fate-of-the-storm-gods", "hero-of-kendrickstone", "hero-project-open-season",
    "hero-project-redemption-season", "heroes-of-myth", "hollywood-visionary", "i-cyborg",
    "kidnapped", "kung-fu", "luminous-underground", "metahuman-inc", "midsummer-nights-choice",
    "mysteries-of-baroque", "pendragon-rising", "ratings-war", "sea-eternal",
    "seventh-sea-a-pirates-pact", "slammed", "t-rex-time-machine", "tower-behind-the-moon",
    "trials-of-the-thief-taker", "undercover-agent", "vampire-the-masquerade_parliament-of-knives",
    "versus-deathscapes", "versus-the-lost-ones", "werewolves-haven-rising",
]


def _done_indices(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {json.loads(line)["episode"] for line in path.read_text().splitlines() if line}


def run_game(job: tuple[str, str, list[int], str]) -> tuple[str, int, float]:
    """Run the given episode indices of one game; append one JSON line per episode."""
    src, game, indices, out_dir = job
    os.chdir(src)  # upstream config resolves ./game_data relative to the checkout
    sys.path.insert(0, src)
    from machiavelli.agent.random_agent import RandomAgent  # noqa: PLC0415
    from machiavelli.game.machiavelli_env import MachiavelliEnv, Trajectory  # noqa: PLC0415
    from machiavelli.utils import random_seed  # noqa: PLC0415

    out = Path(out_dir) / f"{game}.jsonl"
    crash_file = Path(out_dir) / f"{game}.crash.json"
    if crash_file.exists():  # upstream's run ended at the crash; so does ours
        return game, 0, 0.0
    todo = [i for i in indices if i not in _done_indices(out)]
    if not todo:
        return game, 0, 0.0
    t_game = time.time()
    env = MachiavelliEnv(game=game)
    agent = RandomAgent()
    for idx in todo:
        t0 = time.time()
        seed = idx + SEED_OFFSET
        random_seed(seed)
        if agent.should_reset_each_episode:
            agent.reset()
        obs, info = env.reset(seed=seed)
        traj = Trajectory(env.game, agent_name=str(agent))
        agent.test()
        rew, done, zero_choice = 0, False, 0
        while not done:
            if info["game_state"]["num_choices"] == 0:
                zero_choice += 1
            action = agent.get_action(obs, rew, done, info)
            traj.add(obs, action, rew, info)
            try:
                obs, rew, done, info = env.step(action)
            except AssertionError:
                # The engine asserts a non-empty choice list (player.state). Upstream's loop has
                # no handler, so its run for this game ends here; record where and stop the game.
                crash_file.write_text(json.dumps({
                    "game": game, "episode": idx, "seed": seed, "step": len(traj),
                    "actions": [int(a) for a in traj.action_list],
                    "error": "AssertionError in player.state: len(choice_texts) > 0"}) + "\n")
                return game, todo.index(idx), time.time() - t_game
            if len(traj) >= MAX_TRAJ_LENGTH:
                done = True
        traj.add(obs, -1, rew, info)
        counts = traj.get_mach_counts()
        obs_hash = hashlib.sha256("\x1e".join(traj.obs_list).encode()).hexdigest()
        rec = {
            "game": game,
            "episode": idx,
            "seed": seed,
            "steps": len(traj),
            "capped": len(traj) - 1 >= MAX_TRAJ_LENGTH,
            "points_norm": traj.get_points(normalize=True),
            "points_raw": traj.get_points(),
            "achievements": len(traj.get_achievements()),
            "counts": {k: float(v) for k, v in counts.items()},
            "actions": [int(a) for a in traj.action_list[:-1]],
            "rewards": [float(r) for r in traj.reward_list],
            "zero_choice_states": zero_choice,
            "obs_sha256": obs_hash,
            "wall_s": round(time.time() - t0, 3),
        }
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
    return game, len(todo), time.time() - t_game


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=10, help="indices 0..N-1 per game")
    p.add_argument("--indices", type=int, nargs="*", help="explicit episode indices")
    p.add_argument("--games", nargs="*", default=GAMES_TEST)
    p.add_argument("--workers", type=int, default=12)
    args = p.parse_args()

    src = args.src.expanduser().resolve()
    assert (src / "game_data" / "game_metadata.json").exists(), f"no game_data under {src}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    indices = args.indices if args.indices else list(range(args.episodes))
    jobs = [(str(src), g, indices, str(args.out_dir.resolve())) for g in args.games]
    t0 = time.time()
    with Pool(min(args.workers, len(jobs))) as pool:
        for game, n, secs in pool.imap_unordered(run_game, jobs):
            print(f"{game}: {n} episodes in {secs:.1f} s", flush=True)
    print(f"total wall {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
