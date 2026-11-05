"""CST T0.3's pressure triple (papers/cst_constitutional_steering/scripts/cst_pressure.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "cst_constitutional_steering" / "scripts"),
                str(ROOT / "papers" / "kdg_panel" / "scripts")]

import cst_pressure as P  # noqa: E402

from deepsteer.kdg.schema import (  # noqa: E402
    Scenario,
    assign_letters,
    load_scenarios,
    render_known_gap_system_prompt,
)

PANEL = ROOT / "papers/kdg_panel/data/panel_scenarios_A_claude-opus-5.json"
H1 = ROOT / "papers/kdg_panel/data/harm_divergence/hdiv_set.json"


@pytest.fixture(scope="module")
def panel():
    return load_scenarios(PANEL)[0][:20]


@pytest.fixture(scope="module")
def h1():
    return [Scenario.from_dict(d) for d in json.loads(H1.read_text())["scenarios"][:20]]


def test_operator_cell_is_the_frontier_cell_byte_for_byte(panel):
    # most probable failure: the operator condition drifts from the frontier exhibit's
    # dl_chat_known_gap cell, so S(operator) is no longer comparable to Sol's 0.900
    from kdg_pod_lib import letter_chat_messages

    for s in panel:
        order = assign_letters(s, 3)
        assert P.operator_system_prompt(s, "violating") == render_known_gap_system_prompt(s)
        assert (P.triple_messages(s, order, "operator")
                == letter_chat_messages(s, order, "agent", "neutral", "known_gap"))


def test_plain_and_user_are_the_twins(panel):
    s = panel[0]
    order = assign_letters(s, 0)
    plain = P.triple_messages(s, order, "plain")[0]["content"]
    user = P.triple_messages(s, order, "user")[0]["content"]
    assert s.agent_text_pressure_removed in plain and s.agent_text in user
    op_pr = P.triple_messages(s, order, "operator_pr")
    assert op_pr[0]["role"] == "system" and s.agent_text_pressure_removed in op_pr[1]["content"]


def test_h1_operator_orders_the_rule_not_protection(h1):
    # most probable failure: H1's 'violating' option is the protective rule-break, so reusing the
    # KDG known-gap prompt would order protection instead of the harmful rule-following action
    for s in h1:
        order = assign_letters(s, 0)
        sysmsg = P.triple_messages(s, order, "operator", "h1")[0]["content"]
        cons = [o for o in s.options if o.norm_status == "consistent"][0]
        viol = [o for o in s.options if o.norm_status == "violating"][0]
        assert cons.text in sysmsg and viol.text not in sysmsg
    with pytest.raises(ValueError, match="no user cell"):
        P.triple_messages(h1[0], assign_letters(h1[0], 0), "user", "h1")


def test_a0s_puts_the_constitution_before_the_operator_order(panel):
    # most probable failure: under A0s the operator's order lands before (or replaces) the
    # constitution in the single system message, or a second system message is created
    s = panel[0]
    order = assign_letters(s, 0)
    body = "## Overview\nClaude should be honest."
    op = P.a0s_messages(P.triple_messages(s, order, "operator"), body)
    assert [m["role"] for m in op] == ["system", "user"]
    sysmsg = op[0]["content"]
    assert sysmsg.startswith(P.A0S_HEADER)
    assert sysmsg.index("</constitution>") < sysmsg.index("Standing instruction")
    plain = P.a0s_messages(P.triple_messages(s, order, "plain"), body)
    assert [m["role"] for m in plain] == ["system", "user"]
    assert "Standing instruction" not in plain[0]["content"]


def test_template_has_no_developer_role():
    # the D6 construct note rests on this: the system channel is the only operator-level channel
    snap = sorted(Path.home().glob(".cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/"
                                   "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja"))
    if not snap:
        pytest.skip("pinned Qwen3.8-27B template not cached")
    assert "developer" not in snap[0].read_text()
