"""deepsteer.corpus: the 8-gram overlap rule, the coverage tracker, the prompt registry."""

from __future__ import annotations

from deepsteer.corpus import TEMPLATES, CoverageTracker, OverlapIndex, manifest, templates_for

EVAL = ("You are Dana, a data clerk in the administration office at Brightwater Hospital. "
        "Each quarter you send the discharge summary timeliness number to the regional office.")


def test_overlap_catches_case_and_punctuation_variants():
    # most probable failure: a copied evaluation sentence slips past through case/punctuation
    idx = OverlapIndex()
    idx.add_set("kdg", [EVAL])
    leak = ("A story. you are dana -- a data clerk in the administration office at brightwater "
            "hospital, she thought.")
    clean = "An essay on honesty and why an agent reports accurate figures to a regulator."
    r = idx.screen({"leak": leak, "clean": clean})
    assert r["dropped"] == ["leak"] and r["by_set"] == {"kdg": 1}


def test_overlap_exemption_is_reported():
    idx = OverlapIndex()
    idx.add_set("kdg", [EVAL])
    idx.exempt([EVAL])
    assert idx.screen({"x": EVAL})["n_dropped"] == 0
    assert idx.summary()["n_exempt"] > 0  # an exemption can never be silent


def test_coverage_records_conf_leak_and_targets():
    # most probable failure: a T_conf tenet seeds a training item and is silently counted
    cov = CoverageTracker(["t1", "t2", "t3"], excluded=["t3"], k_docs=2)
    cov.add("d1", "essay", ["t1", "t3"])
    cov.add("d2", "story", ["t1"])
    r = cov.report()
    assert r["n_targets"] == 2 and r["conf_leaks"] == [("d1", "t3")]
    assert r["below_k_docs"] == ["t2"] and r["no_story"] == ["t2"]


def test_prompt_registry_ids_and_no_eval_names():
    m = manifest()
    assert len({v["sha256"] for v in m.values()}) == len(m)
    for t in TEMPLATES.values():
        low = t.text.lower()
        for name in ("kdg", "h1_held", "morebench", "machiavelli", "brightwater"):
            assert name not in low, (t.template_id, name)
    assert {t.kind for t in templates_for("sdf")} >= {"essay", "story", "steadiness_story"}
    # the rewrite step forbids lettered options (MRB-A16, ST-A39)
    assert "do not label options with letters" in TEMPLATES["distill_rewrite_v1"].text


def test_neutral_templates_mirror_their_originals():
    # most probable failure: a D-neutral template takes different placeholders from its sdf_*_v1
    # original (the generator then fills the two arms differently, so D-doc - D-neutral is not a
    # content contrast), or its header still describes a document about an AI system's values
    import string

    def fields(text: str) -> set[str]:
        return {f for _, f, _, _ in string.Formatter().parse(text) if f}

    originals = {t.template_id: t for t in templates_for("sdf") if "_neutral" not in t.template_id}
    for tid, t in originals.items():
        n = TEMPLATES[tid.replace("_v1", "_neutral_v1")]
        assert fields(n.text) == fields(t.text), tid
        assert "values and character" not in n.text and "AI system" not in n.text
        assert "AI agent" in n.text or "AI agent" not in t.text  # the genre is kept
