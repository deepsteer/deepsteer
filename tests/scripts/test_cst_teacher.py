"""CST's single-decision teacher check (Amendments 46 item C, 47 item 2):
papers/cst_constitutional_steering/scripts/cst_teacher_pod.py and teacher_read.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts"), str(ROOT / "tests/scripts")]

import cst_teacher_pod as P  # noqa: E402

CST = ROOT / "papers/cst_constitutional_steering"


def test_the_constitutions_states_are_nested_in_the_bases():
    # most probable failure (Amendment 47 item 2): the constitution is read on states the base and
    # GOOD_BEHAVIOR never decided, so its contrast against the base has no paired states
    rec = json.loads(P.STATES.read_text())
    wide, nested = set(rec["wide"]), set(rec["nested"])
    assert nested <= wide and rec["cap_wide"] == 60 and rec["cap_nested"] == 30
    per_game = {}
    for s in nested:
        per_game[s.split("|")[0]] = per_game.get(s.split("|")[0], 0) + 1
    assert max(per_game.values()) <= 30 and len(rec["wide"]) == rec["n_wide"]


class _Eng:
    """Replies by request: the first attempt hits the cap, the second answers in range."""

    def __init__(self, answer_ids, cap_ids):
        self.calls, self.answer, self.cap = 0, answer_ids, cap_ids

    def generate(self, prompts, max_tokens):
        self.calls += 1
        return [{"ids": self.cap if self.calls == 1 else self.answer} for _ in prompts]


class _Tok:
    def decode(self, ids):
        return "2" if ids == [7] else ""


def test_attempt_rule_allows_one_more_after_a_cap_then_falls_back_paired(monkeypatch):
    # most probable failures: a capped decision is retried five times (MCH's rule allows one more),
    # or the arms' fallbacks draw different random actions so a fallback reads as an arm effect
    monkeypatch.setattr(P.F, "MACH_TOKENS", 3)
    eng = _Eng([7], [0, 0, 0])
    jobs = [{"arm": a, "sid": "g|0|1", "k": 0, "n": 4, "ids": [1]}
            for a in ("base", "constitution")]
    out = P.decide(eng, _Tok(), None, jobs, True)
    assert all(j["attempts"] == 2 and j["action"] == 2 and not j["fallback"] for j in out)
    eng = _Eng([0, 0, 0], [0, 0, 0])  # every attempt capped: one more, then MCH's fallback
    out = P.decide(eng, _Tok(), None, [dict(j) for j in jobs], True)
    assert all(j["attempts"] == 2 and j["fallback"] for j in out)
    assert out[0]["action"] == out[1]["action"]  # the same draw on both arms


def test_branches_follow_the_registered_rule(tmp_path, monkeypatch):
    # most probable failure (Amendment 46 item C6): the branch is read from the wrong contrast, or
    # a constitution that lowers violations is reported as "only GOOD_BEHAVIOR"
    import teacher_read as R

    games = [f"g{i}" for i in range(6)]
    lab = [{"game": g, "episode": 0, "step": s, "violating": [1, 0], "labels": [
        {c: 1 for c in R.CATS}, {c: 0 for c in R.CATS}], "saved_violating": 1, "saved_action": 0}
        for g in games for s in range(3)]
    lf = tmp_path / "labels.jsonl"
    lf.write_text("".join(json.dumps(x) + "\n" for x in lab))
    monkeypatch.setattr(R, "LABELS", lf)
    (tmp_path / "teacher").mkdir()

    def rows(arm, action):
        return [{"arm": arm, "sid": f"{g}|0|{s}", "k": k, "action": action, "fallback": False}
                for g in games for s in range(3) for k in (0, 1)]

    (tmp_path / "teacher" / "base_good_behavior.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("base", 0)[:18] + rows("base", 1)[18:]
        + rows("good_behavior", 1)))
    (tmp_path / "teacher" / "constitution.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("constitution", 1)))
    res = R.read(tmp_path)
    assert res["constitution_minus_base"]["below_0"] and res["branch"].startswith(
        "the constitution lowers")


@pytest.mark.slow
def test_dry_arms_rebuild_states_through_the_replay_gate(tmp_path):
    # most probable failure (Amendment 46 item C1): a rebuilt state's prompt differs from the saved
    # one, so the arms decide other scenes than the base reached; dry, four states
    from deepsteer.datasets import machiavelli as mch

    if not (P.RUN / "decisions.jsonl").exists() or not (
            mch.default_cache_dir() / "src" / "game_data").exists():
        pytest.skip("the finalist run's decisions or the MACHIAVELLI bundle are not here")
    rec = P.stage_arms(tmp_path, ("base", "good_behavior"), "wide", True)
    assert rec["states"] == 4 and rec["dropped_by_the_replay_gate"] == 0
    rows = [json.loads(x) for x in open(tmp_path / "teacher" / "base_good_behavior.jsonl")]
    assert len(rows) == 4 * 2 * P.K and {r["arm"] for r in rows} == {"base", "good_behavior"}
