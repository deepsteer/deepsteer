"""MCH-G0 prompt parity at zero GPU (MCH_PLAN §5): our prompts against upstream's, byte for byte.

Game states come from replaying saved random episodes (seed and action indices from the G0 pilot)
on the pinned engine. Upstream's prompts come from the paper-era agents (``lm_agents.py`` at
``9f82117``) with ``machiavelli.openai_helpers`` replaced by a stub that records each call and
returns a fixed reply; no API is called. Ours come from ``papers/mch_machiavelli/mch_prompts.py``.
Every agent is rendered on both paths: non-chat (``davinci``; for the CoT agents both calls, the
second embedding a fixed reasoning string) and chat (``gpt-4``).

Writes ``outputs/mch_g0_parity.json``: per game, state and prompt, whether the bytes match and the
sha256 of ours. Rendered text is not written (MCH_PLAN §9).

Must run under the engine's environment (Python <= 3.11, gym 0.25.1, asteval 0.9.29).

Usage:
    python mch_g0_parity.py --src ~/.cache/deepsteer/mch/src \
        --paper-era-agents ~/.cache/deepsteer/mch/paper_era/lm_agents_9f82117.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve()
MCH = HERE.parents[1]
sys.path.insert(0, str(MCH))
import mch_prompts as ours  # noqa: E402

PILOT = MCH / "outputs" / "g0_random_pilot"
OUT = MCH / "outputs" / "mch_g0_parity.json"
REASONING = "Step one: SENTINEL reasoning text for the parity check."
SEED_OFFSET = 2417


class Recorder:
    """Stands in for upstream's OpenAI helpers; records prompts, returns fixed replies."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def completion_with_retries(self, model: str, prompt: str, **kwargs) -> dict:
        self.calls.append({"kind": "completion", "prompt": prompt})
        return {"choices": [{"text": REASONING}]}

    def chat_completion_with_retries(self, model: str, sys_prompt: str, prompt: str,
                                     **kwargs) -> dict:
        self.calls.append({"kind": "chat", "sys": sys_prompt, "user": prompt})
        return {"choices": [{"message": {"content": '{"reasoning": "x", "action": 0}'}}]}

    @staticmethod
    def get_class_probs(res: dict, labels: list[str]) -> list[float]:
        return [1 / len(labels)] * len(labels)

    @staticmethod
    def logit_whitelist(labels: list[str]) -> dict:
        return {}


def load_upstream_agents(path: Path, rec: Recorder) -> types.ModuleType:
    stub = types.ModuleType("machiavelli.openai_helpers")
    for name in ("completion_with_retries", "chat_completion_with_retries", "get_class_probs",
                 "logit_whitelist"):
        setattr(stub, name, getattr(rec, name))
    sys.modules["machiavelli.openai_helpers"] = stub
    spec = importlib.util.spec_from_file_location("lm_agents_paper_era", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", required=True, type=Path)
    p.add_argument("--paper-era-agents", required=True, type=Path)
    args = p.parse_args()
    src = args.src.expanduser().resolve()
    agents_path = args.paper_era_agents.expanduser().resolve()
    os.chdir(src)
    sys.path.insert(0, str(src))
    from machiavelli.game.machiavelli_env import MachiavelliEnv  # noqa: PLC0415

    rec = Recorder()
    up = load_upstream_agents(agents_path, rec)

    results, n, n_eq = [], 0, 0
    # The first pilot episode of each game (episode 0), from the pilot files or their gzip.
    if any(PILOT.glob("*.jsonl")):
        lines = [f.read_text().splitlines()[0] for f in sorted(PILOT.glob("*.jsonl"))]
    else:
        lines = gzip.open(PILOT.with_suffix(".jsonl.gz"), "rt").read().splitlines()
    firsts = sorted((json.loads(x) for x in lines if json.loads(x)["episode"] == 0),
                    key=lambda r: r["game"])
    for ep in firsts:
        game, actions = ep["game"], ep["actions"]
        env = MachiavelliEnv(game=game)
        obs, info = env.reset(seed=ep["seed"])
        picks = {0, len(actions) // 2, len(actions) - 1}
        for t, a in enumerate(actions):
            if t in picks:
                s = ours.SceneState.from_info(obs, info)
                for agent in ours.AGENTS:
                    # Non-chat path.
                    rec.calls.clear()
                    getattr(up, agent)(model="davinci").get_action_probabilities(
                        obs, 0, False, info)
                    mine = [ours.completion_prompt(agent, s)]
                    if agent in ("CoTAgent", "GoodCoTAgent"):
                        mine.append(ours.cot_answer_prompt(agent, s, REASONING))
                    theirs = [c["prompt"] for c in rec.calls]
                    # Chat path.
                    rec.calls.clear()
                    getattr(up, agent)(model="gpt-4").get_action_completion(obs, 0, False, info)
                    sys_, user = ours.chat_prompts(agent, s)
                    mine += [sys_, user]
                    theirs += [rec.calls[0]["sys"], rec.calls[0]["user"]]
                    names = (["completion", "cot_answer"] if len(mine) == 4 else
                             ["completion"]) + ["chat_system", "chat_user"]
                    assert len(theirs) == len(mine), (game, agent, len(theirs), len(mine))
                    for name, m, th in zip(names, mine, theirs):
                        eq = m == th
                        n += 1
                        n_eq += eq
                        results.append({"game": game, "step": t, "agent": agent, "prompt": name,
                                        "equal": eq, "sha256_ours": sha(m),
                                        "sha256_upstream": sha(th)})
            obs, _, done, info = env.step(a)
            if done:
                break
    OUT.write_text(json.dumps({
        "upstream_agents": "lm_agents.py at 9f82117 (prompts identical at ebb44e1)",
        "states": len({(r["game"], r["step"]) for r in results}),
        "prompts_compared": n, "byte_equal": n_eq,
        "mismatches": [r for r in results if not r["equal"]],
        "rows": results}, indent=1) + "\n")
    print(f"{n_eq} of {n} prompts byte-equal over "
          f"{len({(r['game'], r['step']) for r in results})} states; wrote {OUT}")


if __name__ == "__main__":
    main()
