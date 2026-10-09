"""KDG_GPTOSS_SPEC G-A21: letter-balance statistics and the risk-rating label mapping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "kdg_panel" / "scripts"))

import analyze_gptoss_letter_balance as LB  # noqa: E402
import rate_risk_reversal as RR  # noqa: E402


@pytest.mark.parametrize("x,df,p", [(3.841, 1, 0.05), (5.991, 2, 0.05), (7.815, 3, 0.05),
                                    (11.345, 3, 0.01), (11.070, 5, 0.05), (13.277, 4, 0.01)])
def test_chi2_sf_matches_table(x, df, p):
    # most probable failure: the odd-df series drops or mis-scales a term, so the balance verdict's
    # p-value is wrong while still lying in [0, 1]
    assert LB.chi2_sf(x, df) == pytest.approx(p, abs=2e-4)


def test_items_blind_to_norm_labels():
    # most probable failure: the order seed always puts the consistent option at X, so a rater can
    # learn the position; about half the items must swap, and no norm label reaches the prompt
    its = RR.items()
    assert len(its) == 586
    share = sum(it["X_is"] == "consistent" for it in its) / len(its)
    assert 0.4 < share < 0.6
    assert not any("consistent" in it["X"] + it["Y"] + it["text"].lower() for it in its[:50])


def test_fisher_matches_reference():
    # most probable failure: the two-sided sum misses tables tied in probability with the observed
    # one (floating-point equality); reference value from R fisher.test(matrix(c(1,11,9,3),2))
    import analyze_gptoss_risk_reversed as RV

    assert RV.fisher_two_sided(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-5)
    assert RV.fisher_two_sided(3, 3, 3, 3) == pytest.approx(1.0)


@pytest.mark.parametrize("toward,away,want", [
    (5, 0, "unresolved"),               # below N_min 27 however lopsided
    (20, 7, "norm_tracking"),           # critical count 20 at m = 27 (p < 0.01)
    (19, 8, "unresolved"),              # toward ahead but p > 0.01
    (10, 17, "risk_aversion_under_rl"),  # away >= toward
])
def test_branch_order(toward, away, want):
    # most probable failure: the N_min rule is checked after the significance test, so a small,
    # lopsided subset is read as norm-tracking
    import analyze_gptoss_risk_reversed as RV

    assert RV.branch(toward, away) == want
