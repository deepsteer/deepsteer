"""deepsteer.judges.panel: test-retest, pooled labels, the T_conf disagreement rule."""

from __future__ import annotations

from deepsteer.judges import (
    conflict_threshold,
    cross_family,
    disagreement,
    is_conflict,
    pooled_label,
    retest_agreement,
)


def test_pooled_label_tie_is_none():
    # most probable failure: a 1-1-1 split pooled to whichever label sorts first
    assert pooled_label(["t1", "t2", "t3"])[0] is None
    assert pooled_label(["t1", "t1", None]) == ("t1", 2 / 3)


def test_conflict_rule_admits_only_beyond_calibration_spread():
    # with three families disagreement is discrete (0, 1/3, 2/3): the quantile decides whether a
    # 2-1 split is a conflict, which is the author's G0 choice
    assert conflict_threshold([0.0] * 9 + [1 / 3], 0.9) == 0.0
    thr = conflict_threshold([0.0] * 9 + [1 / 3], 0.95)
    assert thr == 1 / 3
    assert not is_conflict(["a", "a", "b"], thr)  # 1/3 is within the settled-item spread
    assert is_conflict(["a", "b", "c"], thr)
    assert disagreement(["a", "a", "a"]) == 0.0


def test_retest_and_cross_family():
    r = retest_agreement(["a", "b", "a", "b"], ["a", "b", "a", "a"])
    assert r["n"] == 4 and not r["passes"]
    cf = cross_family({"x": ["a", "b"], "y": ["a", "b"], "z": ["b", "a"]})
    assert cf["pairwise_kappa"]["x|y"] == 1.0
