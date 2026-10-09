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
