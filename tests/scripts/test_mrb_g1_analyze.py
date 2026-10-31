"""MRB-G1 analysis (papers/mrb_morebench/scripts/mrb_g1_analyze.py): scoring, match rule, J1."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_g1_analyze as A  # noqa: E402


def _rows(unit, credits):
    return [{"unit": unit, "criterion_id": cid, "verdict": "yes", "credit_substring": sub,
             "credit_exact": ex} for cid, (sub, ex) in credits.items()]


def test_unit_scores_credit_negative_criteria_in_their_intended_direction():
    # most probable failure: a credited negative criterion (the response avoided it) scored as a
    # miss, because the rendered judgement for code_score is inverted for negative weights
    weights = {"mrb-000": {"a": 2, "b": -3}}
    u = "m/default/mrb-000/k0/trace"
    us = A.unit_scores(_rows(u, {"a": (True, True), "b": (True, False)}), weights)[u]
    assert us["credit_substring"] == 100.0 and us["credit_exact"] == 40.0
    assert abs(us["eq1_paper_scale"] - us["credit_substring"]) < 1e-9  # Eq. 1 == code on 0-100
    us = A.unit_scores(_rows(u, {"a": (False, False), "b": (True, True)}), weights)[u]
    assert us["credit_substring"] == 60.0 and us["complete"]


def test_match_rule_widens_by_the_rounding_and_no_further():
    # most probable failure: the 0.05 rounding allowance applied twice, or not at all, so a cell
    # just outside the PI is called a replication (or one inside it a miss)
    rng = np.random.default_rng(0)
    score = np.full((500, 2), 70.0)
    length = np.full((500, 2), 1000.0)
    cell = [0] * 7 + [70.05, 1000, 70.0]
    assert A.match(score, length, cell, rng)["regular"]["verdict"] == "replicates"
    cell[7] = 70.06
    assert A.match(score, length, cell, rng)["regular"]["verdict"] == "misses"
    assert A.match(score, length, cell, rng)["hard"]["verdict"] == "replicates"


def test_criterion_agreement_pairs_each_criterion_with_its_own_twin():
    # most probable failure: a retest row paired with the judge fork's (or with another unit's)
    rows = (_rows("u", {"a": (True, True), "b": (False, False)})
            + _rows("u/retest", {"a": (True, True), "b": (True, True)})
            + _rows("u/judgefork", {"a": (False, False), "b": (False, False)}))
    r = A.criterion_agreement(rows, "/retest")
    assert r["n"] == 2 and r["agreement"] == 0.5
    assert A.criterion_agreement(rows, "/judgefork")["agreement"] == 0.5
