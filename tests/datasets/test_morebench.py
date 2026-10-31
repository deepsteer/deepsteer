"""MoReBench loader on a synthetic MoReBench-shaped CSV (no upstream text)."""

from __future__ import annotations

import csv

import pytest

from deepsteer.datasets import morebench as mb


def _csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["DILEMMA", "DILEMMA_SOURCE", "DILEMMA_TYPE", "THEORY",
                                          "RUBRIC", "ROLE_DOMAIN", "CONTEXT"])
        w.writeheader()
        w.writerows(rows)


def _row(i, crit):
    return {"DILEMMA": f"Invented dilemma {i}.", "DILEMMA_SOURCE": "daily_dilemmas",
            "DILEMMA_TYPE": "short_case", "THEORY": "neutral", "RUBRIC": repr(crit),
            "ROLE_DOMAIN": "ai_advisor", "CONTEXT": "ctx"}


def test_rubric_is_a_python_literal_and_ids_follow_row_order(tmp_path):
    # most probable failure: parsing RUBRIC as JSON (it is a single-quoted Python literal, which
    # upstream reads with ast.literal_eval) or ids that drift from upstream's row index `idx`
    crit = [{"annotations": {"rubric_dimension": "identifying"}, "id": "c1",
             "title": "Names the 'conflict'", "weight": 3},
            {"annotations": {"rubric_dimension": "harmless outcome"}, "id": "c2",
             "title": "Recommends harm", "weight": -2}]
    for name in mb.MOREBENCH_FILES:
        (tmp_path / name).write_text("")
    _csv(tmp_path / "morebench_public.csv", [_row(0, crit), _row(1, crit[:1])])
    items = mb.load_morebench(tmp_path, "public", verify=False)
    assert [it.scenario_id for it in items] == ["mrb-000", "mrb-001"]
    assert items[0].criteria[1] == mb.MoReBenchCriterion("c2", "Recommends harm", -2,
                                                          "harmless outcome")


def test_verify_rejects_a_changed_file(tmp_path):
    # most probable failure: verification by presence only, so an edited CSV passes
    for name in mb.MOREBENCH_FILES:
        (tmp_path / name).write_text("x")
    manifest = {"revision": "r", "files": {n: mb.sha256_file(tmp_path / n)
                                            for n in mb.MOREBENCH_FILES}}
    (tmp_path / "morebench_public.csv").write_text("y")
    with pytest.raises(ValueError, match="sha256 mismatch: morebench_public.csv"):
        mb.verify_morebench(tmp_path, manifest)


def test_manifest_pins_the_revision_with_hashes_only():
    m = mb.load_manifest()
    assert m["revision"] == mb.MOREBENCH_REVISION
    assert set(m["files"]) == set(mb.MOREBENCH_FILES)
    assert all(len(v) == 64 for v in m["files"].values())
