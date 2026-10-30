"""MB-G0 audit functions on synthetic items (papers/mb_moralbench/scripts/mb_g0_audit.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.moralbench import MoralBenchItem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mb_moralbench" / "scripts"))

import mb_g0_audit as A  # noqa: E402


def _it(item_id: str, key: dict[str, float], prompt: str = "") -> MoralBenchItem:
    return MoralBenchItem("s", item_id, item_id.rsplit("_", 1)[0],
                          MoralFoundation.CARE_HARM, prompt, key)


def test_ladder_bounds_and_uniform():
    # most probable failure: oracle/worst computed per letter column instead of per item, which
    # makes the band too narrow whenever the best letter differs across items
    items = [_it("harm_1", {"A": 3.0, "B": 2.0}), _it("harm_2", {"A": 1.0, "B": 4.0})]
    lad = A.ladder(items)
    assert lad == {"n": 2, "worst": 3.0, "always_A": 4.0, "always_B": 6.0, "uniform": 5.0,
                   "oracle": 7.0}


def test_tied_item_raises_the_floor():
    # most probable failure: a tied key (both letters score 1) counted as 0 in the worst policy,
    # hiding that any A/B answer earns the point (MFQ_30_compare ingroup_2 upstream)
    items = [_it("ingroup_1", {"A": 0.0, "B": 1.0}), _it("ingroup_2", {"A": 1.0, "B": 1.0})]
    assert A.ladder(items)["worst"] == 1.0
    assert A.worst_case_mde([items[1]], 5, 5) == 0.0


def test_formula_check_flags_off_scale_items():
    items = [_it("harm_1", {"A": 2.2, "B": 2.8}), _it("harm_2", {"A": 2.0, "B": 2.0})]
    assert A.formula_check(items, 5.0) == ["harm_2"]


def test_compare_consistency_detects_an_impossible_group():
    # most probable failure: checking each item against "higher H wins" separately, which needs
    # the anchor's H (unrecoverable); the group test needs only the binary H of option A
    binary = [
        _it(f"harm_{i}", {"A": h, "B": 5 - h}, f"Stem\n\nStatement number {i} here\n\nA. Agree")
        for i, h in [(1, 3.0), (2, 2.0), (3, 1.0)]
    ]
    anchor = "B. The shared anchor statement"
    comp = [
        _it("harm_1", {"A": 0.0, "B": 1.0}, f"Pick\n\nA. Statement number 1 here\n{anchor}"),
        _it("harm_2", {"A": 1.0, "B": 0.0}, f"Pick\n\nA. Statement number 2 here\n{anchor}"),
        _it("harm_3", {"A": 0.0, "B": 1.0}, f"Pick\n\nA. Statement number 3 here\n{anchor}"),
    ]
    # A wins at H=2, loses at H=3 and H=1: no single anchor H fits either orientation
    g = A.compare_consistency(comp, binary)["groups"][0]
    assert g["consistent_higher_h_wins"] is False and g["consistent_lower_h_wins"] is False
    dup = A.compare_consistency(comp + [_it("harm_4", {"A": 0.0, "B": 1.0}, comp[1].prompt)],
                                binary)["duplicate_items"]
    assert dup == [{"items": ["harm_2", "harm_4"], "keys_agree": False}]


def test_assert_no_text_blocks_a_leak():
    # most probable failure: an option string reaching the tracked JSON
    it = _it("harm_1", {"A": 1.0, "B": 0.0}, "Pick\n\nA. A long invented option sentence\nB. x")
    A.assert_no_text('{"ok": 1}', [it], "system")
    with pytest.raises(AssertionError):
        A.assert_no_text('{"leak": "a long invented option sentence"}', [it], "system")
