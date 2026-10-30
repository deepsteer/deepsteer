"""MoralBench loader on synthetic MoralBench-shaped items (no upstream text; MB_PLAN §9.2)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets import moralbench as mb


def _tree(root: Path) -> dict:
    """A two-set checkout with invented items, plus its manifest."""
    (root / "template").mkdir(parents=True)
    (root / "template" / "moral_system.txt").write_text("Pick one letter.")
    files = {}
    for s, items in {
        "MFQ_30": {"harm_1": "Stem line\n\nSynthetic statement one\n\nA. Agree B. Disagree",
                   "ingroup_1": "Stem line\n\nSynthetic statement two\n\nA. Agree B. Disagree"},
        "MFQ_30_compare": {"harm_1": "Pick the more moral.\n\nA. First invented\nB. Second"},
    }.items():
        (root / "questions" / s).mkdir(parents=True)
        for stem, text in items.items():
            (root / "questions" / s / f"{stem}.txt").write_text(text)
        keys = {stem: {"A": 2.0, "B": 3.0} for stem in items}
        keys["orphan_9"] = {"A": 1.0, "B": 0.0}
        (root / "answers").mkdir(exist_ok=True)
        (root / "answers" / f"{s}.json").write_text(json.dumps(keys))
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(root))] = mb.sha256_file(p)
    return {"repo": "synthetic", "commit": "0" * 40, "files": files}


def test_readlines_join_doubles_newlines():
    # most probable failure: rendering with str.splitlines() or plain read(), which drops the
    # doubling the upstream harness applies (every "\n" becomes "\n\n"; no newline added at EOF)
    assert mb.readlines_join("a\nb\n\nc") == "a\n\nb\n\n\n\nc"
    assert mb.readlines_join("a\n") == "a\n"
    assert mb.readlines_join("x") == "x"
    # U+2028 is not a line break for readlines() on a text stream; splitlines() would split it
    assert mb.readlines_join("a b\nc") == "a b\n\nc"


def test_load_items_keys_and_foundations(tmp_path, monkeypatch):
    # most probable failure: items driven by answer keys instead of question files, so orphan
    # keys (present upstream: life_boat, trolley_tracks, MFQ liberty_*) become phantom items
    manifest = _tree(tmp_path)
    monkeypatch.setattr(mb, "load_manifest", lambda path=mb.MANIFEST_PATH: manifest)
    system, items = mb.load_moralbench(tmp_path, sets=("MFQ_30", "MFQ_30_compare"))
    assert system == "Pick one letter."
    assert [it.qualified_id for it in items] == [
        "MFQ_30/harm_1", "MFQ_30/ingroup_1", "MFQ_30_compare/harm_1"]
    assert items[1].foundation is MoralFoundation.LOYALTY_BETRAYAL
    assert items[0].foundation is MoralFoundation.CARE_HARM
    assert items[0].key == {"A": 2.0, "B": 3.0}
    assert items[0].prompt.startswith("Stem line\n\n\n\nSynthetic")


def test_verify_rejects_a_modified_file(tmp_path):
    # most probable failure: verification checks presence only, so a changed key file passes
    manifest = _tree(tmp_path)
    (tmp_path / "answers" / "MFQ_30.json").write_text("{}")
    with pytest.raises(ValueError, match="sha256 mismatch: answers/MFQ_30.json"):
        mb.verify_moralbench(tmp_path, manifest)


def test_question_without_key_fails_loudly(tmp_path, monkeypatch):
    # most probable failure: a missing key silently skipped, shrinking the item set
    _tree(tmp_path)
    (tmp_path / "questions" / "MFQ_30" / "purity_1.txt").write_text("Unkeyed")
    with pytest.raises(KeyError):
        mb.load_moralbench(tmp_path, sets=("MFQ_30",), verify=False)


def test_foundation_map_covers_six_foundations():
    assert set(mb.FOUNDATION_MAP.values()) == set(MoralFoundation)


def test_manifest_holds_hashes_only():
    # most probable failure: a manifest that embeds file contents (§9.2: paths and hashes only)
    m = mb.load_manifest()
    assert m["commit"] == mb.MORALBENCH_COMMIT
    assert len(m["files"]) == 93
    assert all(len(v) == 64 and set(v) <= set("0123456789abcdef") for v in m["files"].values())


@pytest.mark.network
@pytest.mark.skipif(os.environ.get("DEEPSTEER_NETWORK_TESTS") != "1",
                    reason="opt-in: fetches MoralBench (no upstream license; never in CI)")
def test_real_checkout_counts():
    # most probable failure: the pinned commit drifts from the manifest (upstream force-push)
    _, items = mb.load_moralbench()
    counts = {s: sum(it.set_name == s for it in items) for s in mb.MORALBENCH_SETS}
    assert counts == {"MFQ_30": 20, "6_concepts": 24, "MFQ_30_compare": 20,
                      "6_concepts_compare": 24}
