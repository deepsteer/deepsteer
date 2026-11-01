"""MRB Phase 2 overlap check (papers/mrb_morebench/scripts/mrb_p2_overlap.py)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mrb_morebench" / "scripts"))

import mrb_p2_overlap as O  # noqa: E402

DILEMMA = ("You manage a small clinic and must decide whether to share a patient's records with "
           "their employer, who has offered to fund a new wing if the records show the patient is "
           "fit to return to work after the injury last spring.")


def test_check_catches_a_planted_dilemma_and_a_partial_copy():
    # most probable failure (a null with no positive control): the check reads 0 because it never
    # compares the right strings, e.g. case or punctuation differences hiding an exact copy
    texts = ["An unrelated scenario about a delivery robot choosing a route through a park.",
             DILEMMA.upper().replace(",", " ;")]
    r = O.check({"d1": DILEMMA, "d2": "A different dilemma entirely about a farmer and a well "
                                      "that the village depends on during a long dry summer."},
                texts)
    assert r["overlapping"] == ["d1"] and r["n_exact"] == 1
    half = " ".join(O.words(DILEMMA)[:20])  # a copied fragment: well over 10% of d1's 8-grams
    r2 = O.check({"d1": DILEMMA}, ["Prefix words here " + half + " and a different ending."])
    assert r2["overlapping"] == ["d1"] and r2["n_exact"] == 0
    assert 0.10 <= r2["top_shares"][0] < 1.0


def test_strings_reach_nested_texts():
    # most probable failure: paraphrase lists or option texts skipped, so overlap there is missed
    obj = {"eval_text": "a", "eval_text_paraphrases": ["b", "c"], "options": [{"text": "d"}],
           "level": 3}
    assert sorted(O.strings(obj)) == ["a", "b", "c", "d"]
