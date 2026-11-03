"""MACHIAVELLI prompts against the paper-era agents (no game data: synthetic scene states).

The reference is upstream's ``lm_agents.py`` at the paper-era commit ``9f82117``, read from the
fetched engine clone with its OpenAI helpers stubbed to record prompts. Skips when the engine has
not been fetched (``deepsteer.datasets.machiavelli.fetch_engine``).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types

import pytest

from deepsteer.benchmarks.agentic.machiavelli import prompts as P


def _upstream(tmp_path):
    from deepsteer.datasets import machiavelli as mch

    src = mch.default_cache_dir() / "src"
    if not (src / ".git").exists():
        pytest.skip("MACHIAVELLI engine not fetched")
    code = subprocess.run(["git", "show", f"{P.UPSTREAM_COMMIT}:machiavelli/agent/lm_agents.py"],
                          cwd=src, check=True, capture_output=True, text=True).stdout
    path = tmp_path / "lm_agents_paper_era.py"
    path.write_text(code)
    calls: list[dict] = []
    stub = types.ModuleType("machiavelli.openai_helpers")
    stub.completion_with_retries = lambda model, prompt, **kw: (
        calls.append({"prompt": prompt}) or {"choices": [{"text": "REASONING"}]})
    stub.chat_completion_with_retries = lambda model, sys_prompt, prompt, **kw: (
        calls.append({"sys": sys_prompt, "user": prompt})
        or {"choices": [{"message": {"content": '{"reasoning": "x", "action": 0}'}}]})
    stub.get_class_probs = lambda res, labels: [1 / len(labels)] * len(labels)
    stub.logit_whitelist = lambda labels: {}
    sys.path.insert(0, str(src))
    sys.modules["machiavelli.openai_helpers"] = stub
    spec = importlib.util.spec_from_file_location("lm_agents_paper_era", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, calls


def _info(n: int, unlocked: list[str]) -> tuple[str, dict]:
    achievements = {f"a{i}": {"description": f"Invented {i}: do thing {i}", "points": 5 * i}
                    for i in range(1, 5)}
    info = {"title": "An Invented Game", "short_summary": "A summary.\nOn two lines.",
            "player_character": "You",
            "game_achievements": achievements,
            "game_state": {"num_choices": n, "choice_texts": [f"Do {i}" for i in range(n)],
                           "unlocked_achievements": unlocked}}
    obs = "  Scene opens.\n\n" + "\n".join(f"{i}: Do {i}" for i in range(n)) + "\n  "
    return obs, info


@pytest.mark.parametrize("n,unlocked", [(2, []), (4, ["a2"]), (7, ["a1", "a3"])])
def test_every_prompt_is_byte_equal_to_the_paper_era_agent(tmp_path, n, unlocked):
    # most probable failure: a one-character drift in a template (a space, a newline, a word)
    up, calls = _upstream(tmp_path)
    obs, info = _info(n, unlocked)
    s = P.SceneState.from_info(obs, info)
    for agent in P.AGENTS:
        calls.clear()
        getattr(up, agent)(model="davinci").get_action_probabilities(obs, 0, False, info)
        assert calls[0]["prompt"] == P.completion_prompt(agent, s)
        if agent in ("CoTAgent", "GoodCoTAgent"):
            assert calls[1]["prompt"] == P.cot_answer_prompt(agent, s, "REASONING")
        calls.clear()
        getattr(up, agent)(model="gpt-4").get_action_completion(obs, 0, False, info)
        assert (calls[0]["sys"], calls[0]["user"]) == P.chat_prompts(agent, s)


def test_good_agent_has_no_goals_and_good_cot_agent_keeps_them():
    obs, info = _info(3, [])
    s = P.SceneState.from_info(obs, info)
    assert "TARGET ACHIEVEMENTS" not in P.chat_prompts("GoodAgent", s)[0]
    sys_ = P.chat_prompts("GoodCoTAgent", s)[0]
    assert "TARGET ACHIEVEMENTS" in sys_ and "TARGET BEHAVIOR" in sys_
