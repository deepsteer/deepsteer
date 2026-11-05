"""CST Amendment 2's item-label rule (cst_constitutional_steering/scripts/scenario_labels.py)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "cst_constitutional_steering" / "scripts"),
                str(ROOT / "papers" / "kdg_panel" / "scripts"),
                str(ROOT / "papers" / "st_live_reference" / "scripts")]

import scenario_labels as L  # noqa: E402

OPTS = ["o1", "o2", "o3"]


def _row(directions, ordered="o2"):
    return {"directions": directions, "operator": {"ordered_option": ordered}}


def test_split_is_unsettled_without_an_explicit_ranking():
    # most probable failure: a split between tenets is resolved by the document's global priority
    # ordering (safe > ethical > guidelines > helpful), which the narrow rule does not count
    r = L.item_rule(_row({"A1": "o1", "A2": "o3"}), ranks=set(), options=OPTS)
    assert r["directed_option"] == "unsettled" and r["direction"] == "unsettled"
    assert not r["ordered_option_directed_by_any"]


def test_explicit_ranking_resolves_and_agreement_settles():
    ranked = L.item_rule(_row({"A1": "o1", "A2": "o3"}), ranks={("A1", "A2")}, options=OPTS)
    assert ranked["directed_option"] == "o1" and ranked["direction"] == "refuse"
    comply = L.item_rule(_row({"A1": "o2", "A2": "o2", "A3": "none"}), ranks=set(), options=OPTS)
    assert comply["direction"] == "comply"
    none = L.item_rule(_row({"A1": "none"}), ranks=set(), options=OPTS)
    assert none["direction"] == "unsettled"
