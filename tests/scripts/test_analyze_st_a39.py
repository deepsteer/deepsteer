"""ST-A39's position split (papers/st_live_reference/scripts/analyze_st_a39.py)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "st_live_reference" / "scripts"))

import analyze_st_a39 as Z  # noqa: E402


def _r(m, lo, hi):
    return {"mean": m, "ci95": [lo, hi], "n": 50}


def test_branches_follow_the_registration():
    # most probable failure: "mixed" (holds everywhere, larger at A) reported as B_A, or a
    # protection that vanishes off A reported as holding
    base = {"A": _r(0.5, 0.4, 0.6), "B": _r(0.25, 0.15, 0.35), "C": _r(0.27, 0.18, 0.36)}
    assert Z.branch({**base, "A_minus_mean_BC": _r(0.25, 0.14, 0.37)}).startswith("mixed")
    assert Z.branch({**base, "B": _r(0.05, -0.05, 0.15),
                     "A_minus_mean_BC": _r(0.3, 0.2, 0.4)}).startswith("B_A")
    assert Z.branch({**base, "A_minus_mean_BC": _r(0.02, -0.08, 0.12)}).startswith("B_hold")


def test_gen_rows_are_split_by_where_protect_sits_not_by_the_chosen_letter():
    # most probable failure: rows grouped by the letter the model chose (which tracks the outcome)
    # instead of the protect option's position in that row's order
    o = lambda a, b, c: {"A": a, "B": b, "C": c}  # noqa: E731
    rows = [{"scenario_id": "s", "order": o("o1", "o2", "o3"), "norm_status": "violating"},
            {"scenario_id": "s", "order": o("o2", "o1", "o3"), "norm_status": "consistent"},
            {"scenario_id": "s", "order": {"A": "o2", "B": "o3", "C": "o1"}, "status": "truncated",
             "norm_status": None}]
    out = Z.gen_by_position(rows, {"s": "o1"}, {"s"})
    assert out["A"] == {"s": 1.0} and out["B"] == {"s": 0.0} and out["C"] == {"s": 0.0}
