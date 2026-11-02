"""MRB Phase 2 length-matched read and DPO pair letters (mrb_p2_length.py, mrb_p2_pairs.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_p2_length as L  # noqa: E402
import mrb_p2_pairs as P  # noqa: E402


def _arms(beta: float, length_shift_k: float, reasoning_cost: float, n=500, seed=0):
    """Scores built as 70 + beta x length + noise: the adapter shortens answers by length_shift_k
    (thousands of characters) and costs reasoning_cost on top."""
    rng = np.random.default_rng(seed)
    base_len = rng.normal(6.0, 0.8, n)
    out = {}
    for arm, shift, cost in (("none", 0.0, 0.0), ("adapter", length_shift_k, reasoning_cost)):
        ln = base_len[:, None] + shift + rng.normal(0, 0.6, (n, 2))
        sc = 70 + beta * ln + cost + rng.normal(0, 2.0, (n, 2))
        out[arm] = (sc, ln)
    return out


def test_a_pure_length_cost_reads_as_length_and_a_reasoning_cost_remains():
    # most probable failure: the adjustment carrying the wrong sign (adding the length term back
    # instead of removing it), so a pure length effect reads as a reasoning cost
    d = _arms(beta=3.0, length_shift_k=-1.5, reasoning_cost=0.0)
    e = L.estimates(d)
    assert abs(e["beta"] - 3.0) < 0.5 and e["delta"] < -3.5 and abs(e["delta_adj"]) < 0.6
    assert 0.8 < e["share_explained"] < 1.2
    d = _arms(beta=3.0, length_shift_k=-1.5, reasoning_cost=-2.0)
    e = L.estimates(d)
    assert -2.6 < e["delta_adj"] < -1.4 and -2.6 < e["cross_intercept"] < -1.4
    assert L.reading([-2.5, -1.5], 3.0, [0.3, 0.8]).startswith("part length, part reasoning")
    assert L.reading([-0.5, 0.4], 3.0, [0.8, 1.1]).startswith("R_length")
    assert L.reading([-2.5, -1.5], 0.1, [-0.1, 0.2]).startswith("R_quality")


def test_exact_binomial_interval_matches_clopper_pearson():
    # most probable failure: the bisection solving the wrong tail, giving an interval that misses
    # the point estimate or is far too narrow at the ends
    assert P.exact_ci(5, 10) == [0.187, 0.813]  # Clopper-Pearson for 5 of 10
    assert P.exact_ci(0, 40)[0] == 0.0 and abs(P.exact_ci(0, 40)[1] - 0.088) < 0.002
    lo, hi = P.exact_ci(11, 40)
    assert lo < 11 / 40 < hi
