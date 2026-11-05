"""deepsteer.constitutions: body cuts, quote verification, the T_core and T_conf rules."""

from __future__ import annotations

import json

import pytest

from deepsteer.constitutions import (
    Tenet,
    TenetSet,
    check_ids,
    core_intersection,
    cut_body,
    document_conflicts,
    majority_match,
    split_sections,
    verify_quotes,
)

PAGE = "\n".join([
    "Site nav",
    "The preface and the acknowledgements are not part of the official constitution.",
    "## Chapter 01 : Overview",
    "01 Overview",
    "## Overview",
    "Claude should be honest.",
    "## Being honest",
    "Claude never tries to create false impressions of itself or the world.",
    "## Acknowledgements",
    "Thanks to everyone.",
    "## Research",
])


def _ts(*tenets: Tenet) -> TenetSet:
    return TenetSet("C_anth", "x", list(tenets))


def test_body_excludes_preface_and_acknowledgements():
    # most probable failure: the body keeps the page's acknowledgements/nav or the preface
    body = cut_body("C_anth", PAGE)
    assert body.startswith("## Overview")
    assert "Thanks to everyone" not in body and "preface" not in body and "Research" not in body
    assert [s.heading for s in split_sections(body)] == ["Overview", "Being honest"]


def test_body_rule_raises_when_boundary_missing():
    with pytest.raises(ValueError, match="body end"):
        cut_body("C_anth", PAGE.replace("## Acknowledgements", "## Thanks"))


def test_verify_quotes_catches_paraphrase():
    # most probable failure: a model-written "quote" that paraphrases the document
    body = cut_body("C_anth", PAGE)
    ok = Tenet("t1", "C_anth", "s001", "Claude never tries to create false impressions of "
               "itself or   the world.", "Do not deceive.")
    para = Tenet("t2", "C_anth", "s001", "Claude never creates false impressions.", "x")
    curly = Tenet("t3", "C_anth", "s001", "Claude should be honest.", "x")
    assert verify_quotes(_ts(ok, para, curly), body) == ["t2"]


def test_majority_rule_two_abstentions_never_match():
    # most probable failure: a lone yes with two abstaining families counted as a match
    assert not majority_match({"gpt_oss_120b": True})
    assert majority_match({"a": True, "b": True, "c": False})
    core = core_intersection({("a1", "s1"): {"a": True, "b": True, "c": False},
                              ("a2", "s2"): {"a": True, "b": False}})
    assert core["pairs"] == [["a1", "s1"]] and core["dissent"] == {"a1|s1": ["c"]}


def test_document_conflicts_and_id_checks():
    a = Tenet("a", "C_anth", "s1", "q", "c", conditioned_on=["b"])
    b = Tenet("b", "C_anth", "s1", "q", "c")
    c = Tenet("c", "C_anth", "s1", "q", "c")
    ts = _ts(a, b, c)
    check_ids(ts)
    conf = document_conflicts(ts, [("b", "c")])
    assert conf == {"a": ["conditioned_on:b"], "b": ["competes:c"], "c": ["competes:b"]}
    with pytest.raises(ValueError, match="unknown"):
        check_ids(_ts(Tenet("x", "C_anth", "s", "q", "c", conditioned_on=["nope"])))
    with pytest.raises(ValueError, match="kind"):
        Tenet("y", "C_anth", "s", "q", "c", kind="vibe")


def test_tenet_set_round_trip_detects_hand_edit(tmp_path):
    ts = _ts(Tenet("a", "C_anth", "s1", "q", "c"))
    p = tmp_path / "t.json"
    ts.save(p)
    assert TenetSet.load(p).ids() == ["a"]
    d = json.loads(p.read_text())
    d["tenets"][0]["claim"] = "edited"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="sha256"):
        TenetSet.load(p)
