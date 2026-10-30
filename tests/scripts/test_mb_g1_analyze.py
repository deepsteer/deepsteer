"""MB-G1 analysis on synthetic responders (papers/mb_moralbench/scripts/mb_g1_analyze.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.moralbench import MoralBenchItem

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mb_moralbench" / "scripts"))

import mb_g1_analyze as A  # noqa: E402


def _items(n=6):
    keys = [{"A": 3.0, "B": 2.0}, {"A": 1.0, "B": 4.0}] * (n // 2)
    return [MoralBenchItem("MFQ_30", f"harm_{i}", "harm", MoralFoundation.CARE_HARM, "", k)
            for i, k in enumerate(keys)]


def _rows(items, choose, cell="t07", n=8):
    return [{"set": it.set_name, "item_id": it.item_id, "rollout": k, "r0": choose(it, k),
             "r0_unstripped": choose(it, k), "r1": choose(it, k)}
            for it in items for k in range(n)]


def _a5(items, rows_o, rows_c):
    # restrict the analysis to the one synthetic set
    orig = A.MORALBENCH_SETS
    A.MORALBENCH_SETS = ("MFQ_30",)
    try:
        return A.acquiescence(rows_o, rows_c, items)["MFQ_30"]
    finally:
        A.MORALBENCH_SETS = orig


def test_constant_letter_responder_reads_as_acquiescence():
    # most probable failure: S computed against the item-level key instead of the constant-rate
    # prediction, so an always-A responder looks item-selective
    its = _items()
    r = _a5(its, _rows(its, lambda it, k: "A"), _rows(its, lambda it, k: "A"))
    assert r["verdict"] == "R_acquiescence"
    assert r["L"][0] == 0.5 and r["K"][0] == 0.0 and r["S"][0] == 0.0


def test_content_follower_reads_as_content():
    # picks the higher-scoring option in both orders: letter bias 0, selectivity > 0
    its = _items()
    best = lambda it, k: max(it.key, key=it.key.get)  # noqa: E731
    flipped = lambda it, k: "B" if best(it, k) == "A" else "A"  # noqa: E731
    r = _a5(its, _rows(its, best), _rows(its, flipped))
    assert r["L"][0] == 0.0 and r["S"][0] > 0 and r["verdict"] == "R_content"


def test_miss_scores_zero_in_the_paper_aggregate():
    # most probable failure: misses dropped from the paper-comparison aggregate (the parsed-only
    # mean) instead of scoring 0, which inflates totals against the paper's
    its = _items(2)
    lab = A.letters(_rows(its, lambda it, k: None if k % 2 else "A", n=4), its, "r0")
    sc = A.scores(lab, its)
    assert np.nan_to_num(sc, nan=0.0).mean(axis=1).sum() == (3.0 + 1.0) / 2
    assert np.nansum(np.nanmean(sc, axis=1)) == 4.0
