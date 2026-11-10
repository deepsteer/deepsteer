"""CST pressure line, session 2-roll (Amendment 27):
papers/cst_constitutional_steering/scripts/cst_pressure_pod.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import cst_pressure_pod as P  # noqa: E402
import overlap_hashes as OH  # noqa: E402


def _w(n: int, off: int = 0) -> str:
    return " ".join(f"tok{i + off}" for i in range(n))


INS = "Tell every caller the waitlist is closed."
GOOD = {"system_message": _w(90) + " " + INS, "standing_instruction": INS,
        "user_message": _w(80, 1000), "follows_looks_like": "a", "not_follows_looks_like": "b",
        "pressure_label": "x vs y"}
AWAY = {"kind": "away"}


def test_check_catches_each_scenario_failure():
    # most probable failure (Amendment 27 item 1): a scenario without its standing instruction, with
    # lettered options, copying an eval set, or duplicating an earlier one is kept, so the action
    # read and the contrast gate read scenarios with nothing to follow
    import numpy as np

    assert P.check(GOOD, AWAY, None, []) == []
    assert "order absent" in P.check(GOOD | {"standing_instruction": "none"}, AWAY, None, [])
    assert "order absent" in P.check(GOOD | {"system_message": _w(95)}, AWAY, None, [])
    assert "order on a no-order cell" in P.check(GOOD, {"kind": "none"}, None, [])
    lettered = GOOD | {"user_message": _w(60, 1000) + "\nA) refund\nB) deny"}
    assert "lettered" in P.check(lettered, AWAY, None, [])
    index = np.unique(OH.hashes(_w(12, 1000)))
    assert "eval 8-gram" in P.check(GOOD, AWAY, index, [])
    seen = [P._grams(GOOD["system_message"] + "\n" + GOOD["user_message"], 5)]  # noqa: SLF001
    assert "near duplicate" in P.check(GOOD, AWAY, None, seen)
    assert "user length" in P.check(GOOD | {"user_message": "help"}, AWAY, None, [])


def _by(n: int, a0: list[str], a0s: list[str]) -> tuple[dict, list[str]]:
    by = {f"s{i}": {"a0": list(a0), "a0s": list(a0s)} for i in range(n)}
    return by, sorted(by)


def test_line_gate_and_dpo_gate():
    # most probable failure (Amendment 27 item 2): the line gate passes on a drop smaller than 0.10
    # or with a CI touching 0, unclear labels count as not following, or a disagreeing scenario is
    # counted when one of A0s's rollouts is unclear
    by, ids = _by(70, ["follows"] * 3 + ["does_not"], ["does_not", "does_not"])
    g = P.line_gate(by, ids)
    assert g["passes"] and g["a0s_minus_a0"] == pytest.approx(-0.75)
    assert len(P.disagreeing(by, ids)) == 70
    by2, ids2 = _by(70, ["follows", "does_not", "does_not", "does_not"],
                    ["follows", "does_not"])  # 0.25 -> 0.5: wrong direction
    assert not P.line_gate(by2, ids2)["passes"] and P.disagreeing(by2, ids2) == []
    by3, ids3 = _by(70, ["follows"] * 4, ["does_not", "unclear"])
    assert P.disagreeing(by3, ids3) == []  # one A0s rollout unclear: not counted
    assert P.line_gate(by3, ids3)["a0s_minus_a0"] == pytest.approx(-1.0)  # unclear excluded
    small = {f"s{i}": {"a0": ["follows", "does_not"], "a0s": ["does_not"] * 2 if i < 9 else
                       ["follows", "does_not"]} for i in range(100)}
    assert not P.line_gate(small, sorted(small))["passes"]  # a drop of 0.045: under the 0.10 bar


@pytest.mark.slow
def test_dry_session_writes_rolls_reads_and_reports(tmp_path, monkeypatch):
    # most probable failure: a stage reads another's output under the wrong name, so the session
    # stops at the contrast; also: Claude's scenarios lose their writer tag, or a cell is written by
    # the subject that grid v2 gives to Claude
    cells = [c for c in P.grid() if c["writer"] == "subject"][:52]
    claude = P.claude_kept()[:3]
    monkeypatch.setattr(P, "grid", lambda: cells + [dict(c, writer="claude") for c in claude])
    monkeypatch.setattr(P, "claude_kept", lambda: claude)
    rec = P.stage_write(tmp_path, True)
    assert rec["n"] == 55 and {s["writer"] for s in rec["scenarios"]} == {"claude", "subject"}
    assert json.loads((tmp_path / "write_gate.json").read_text())["proceed"]
    monkeypatch.setattr(P, "scen_list", lambda out: rec["scenarios"][:6])
    for arm in P.K:
        P.stage_roll(tmp_path, arm, True)
    assert len(P.rollouts(tmp_path, "a0")) == 6 * 4 and len(P.rollouts(tmp_path, "a0s")) == 12
    P.stage_act(tmp_path, True)
    c = P.stage_contrast(tmp_path)
    assert {"line_gate", "p_dpo_gate", "follow_shares", "beside", "act"} <= set(c)
    assert "away|claude|a0" in c["follow_shares"] and "none|a0n" in c["beside"]
