"""T1.5's local read (scripts/analyze_t15.py): Amendment 21 item 3's truncation columns and the
non-carrier verdict ladder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_t15 as A  # noqa: E402


def _r(sid, ns, status="completed"):
    return {"scenario_id": sid, "norm_status": ns, "status": status}


def test_column_of_record_counts_truncated_as_non_protect():
    # most probable failure: a truncated rollout drops out of the denominator in the column of
    # record, which would credit L-beh's long traces with its finished ones only
    rows = [_r("S", "violating"), _r("S", None, "truncated"), _r("S", "consistent"),
            _r("S", None, "other")]
    assert A.p_record(rows)["S"] == 1 / 3
    assert A.T13.p_by_scenario(rows)["S"] == 1 / 2


def _x(cr, ce, spec=True, cap=True, guard=None):
    x = {"category_record": cr, "category_excluded": ce, "specificity": {"passes": spec},
         "capability": {"passes": cap}}
    if guard is not None:
        x["guard"] = guard
    return x


def test_verdict_ladder():
    # most probable failure: a failed guard or an empty guard reads as passed, a comparator whose
    # columns disagree gets the record column's verdict, or L-self gets a verdict at all
    ok = {"fails": False, "n": 200}
    assert A.verdict("anchor", _x("H1 criterion met", "H1 criterion met", guard=ok), "beh") \
        .startswith("H1 criterion met, guard passed")
    assert A.verdict("anchor", _x("H1 criterion met", "H1 criterion met",
                                  guard={"fails": True, "n": 200}), "beh") == \
        "non-specific (panel gap)"
    assert A.verdict("anchor", _x("null", "null", guard={"fails": False, "n": 0}), "beh") \
        .startswith("uninformative")
    assert A.verdict("beh", _x("null", "partial", guard=ok), "beh").startswith(
        "unresolved by truncation")
    assert A.verdict("beh_epoch3", _x("null", "partial"), "beh") == "null"
    assert A.verdict("anchor", _x("partial", "partial", spec=False, guard=ok), "beh") == \
        "non-specific"
    assert A.verdict("self", _x("H1 criterion met", "H1 criterion met"), "beh").startswith(
        "descriptive")


def test_category_bars():
    assert A.category({"mean": 0.16, "ci95": [0.01, 0.3]}) == "H1 criterion met"
    assert A.category({"mean": 0.10, "ci95": [0.01, 0.2]}) == "partial"
    assert A.category({"mean": -0.1, "ci95": [-0.2, -0.01]}) == "rule-ward (CI below 0)"
    assert A.category({"mean": 0.0, "ci95": [-0.1, 0.1]}) == "null"
