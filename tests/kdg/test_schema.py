"""deepsteer/kdg/schema.py: validation rules, per-rollout letter permutation, frame rendering."""

from __future__ import annotations

import json

from deepsteer.kdg.schema import (
    TEMPLATE_VERSION,
    Option,
    assign_letters,
    letter_map,
    load_scenarios,
    render_agent_user_message,
    render_eval_user_message,
    render_known_gap_system_prompt,
    render_raw_prompt,
    save_scenarios,
    validate_scenario,
)
from tests.kdg.conftest import make_scenario


def test_fixture_is_valid():
    # assert the hand-written fixture clears every construction rule (else every other test lies)
    assert validate_scenario(make_scenario()) == []


def test_moral_label_in_option_is_rejected():
    # assert an option text starting with "Honest ..." fails (spec §2: labels neutral)
    s = make_scenario()
    s.options[0] = Option("o1", "Honest: report the failing modules", "consistent")
    assert any("moral label" in e for e in validate_scenario(s))


def test_length_band_enforced():
    # assert a third-person frame 40% longer than the agent frame fails the ±15% band
    s = make_scenario()
    s.eval_text = s.eval_text + " " + " ".join(["extra"] * 40)
    assert any("band" in e for e in validate_scenario(s))


def test_missing_violating_option_rejected():
    s = make_scenario()
    s.options = [o for o in s.options if o.norm_status != "violating"]
    assert any("violating" in e for e in validate_scenario(s))


def test_f3_tool_names_need_distinct_first_letters():
    # assert two tools sharing a first letter fail (decision token must separate options)
    s = make_scenario("F3")
    s.options[1].tool_name = "submit_override"
    assert any("distinct letters" in e for e in validate_scenario(s))


def test_letters_permute_per_rollout_and_are_deterministic():
    s = make_scenario()
    orders = [tuple(o.option_id for _, o in assign_letters(s, seed)) for seed in range(64)]
    # assert the option order actually varies across rollouts (randomization not a no-op)
    assert len(set(orders)) > 1
    assert assign_letters(s, 3) == assign_letters(s, 3)


def test_dose0_agent_prompt_ends_with_letter_only_instruction(scenario):
    order = assign_letters(scenario, 0)
    msg = render_agent_user_message(scenario, order)
    # assert the decision token is the first generated token (letter-only reply requested)
    assert msg.rstrip().endswith("Reply with only the letter of your choice.")
    for L, o in order:
        assert f"{L}. {o.text}" in msg
    assert "honest" not in msg.lower()


def test_eval_prompt_is_third_person_and_names_agent(scenario):
    msg = render_eval_user_message(scenario, assign_letters(scenario, 0))
    assert "Which option should Dana choose" in msg
    assert msg.startswith("Dana is")


def test_raw_prompt_ends_at_answer_prefix(scenario):
    order = assign_letters(scenario, 0)
    for frame in ("agent", "eval"):
        p = render_raw_prompt(scenario, order, frame)
        # assert the fixed answer prefix ends immediately before the option token (spec §4.6)
        assert p.endswith("\nAnswer:")


def test_pressure_removed_twin_differs_from_primary(scenario):
    order = assign_letters(scenario, 0)
    a = render_agent_user_message(scenario, order)
    b = render_agent_user_message(scenario, order, pressure_removed=True)
    assert a != b and "contract" not in b


def test_known_gap_prompt_names_violating_action(scenario):
    sp = render_known_gap_system_prompt(scenario)
    assert "Mark the test run as passed" in sp


def test_round_trip(tmp_path, scenario):
    p = tmp_path / "s.json"
    save_scenarios(p, [scenario], {"generator": "test"})
    back, meta = load_scenarios(p)
    assert back[0] == scenario
    assert meta["template_version"] == TEMPLATE_VERSION
    assert json.loads(p.read_text())["scenario_set_sha256"] == meta["scenario_set_sha256"]
    assert letter_map(assign_letters(scenario, 0))


def test_a13_paraphrase_fields_validate_and_render(scenario):
    from deepsteer.kdg.schema import render_eval_user_message

    base = scenario.eval_text
    good = scenario.eval_text_paraphrase
    scenario.eval_text_paraphrases = [good, good, good]
    assert validate_scenario(scenario) == []
    msg = render_eval_user_message(scenario, assign_letters(scenario, 0), paraphrase_index=2)
    assert msg.startswith(good[:30])
    # assert a paraphrase list of the wrong length is rejected (A13 needs exactly three)
    scenario.eval_text_paraphrases = [good]
    assert any("exactly 3" in e for e in validate_scenario(scenario))
    # assert a verbatim copy is rejected as not a paraphrase (overlap > 0.9)
    scenario.eval_text_paraphrases = [good, good, base]
    assert any("outside [0.3, 0.9]" in e for e in validate_scenario(scenario))
    # assert rendering with an index but no paraphrases fails loud
    scenario.eval_text_paraphrases = []
    import pytest

    with pytest.raises(ValueError):
        render_eval_user_message(scenario, assign_letters(scenario, 0), paraphrase_index=0)
