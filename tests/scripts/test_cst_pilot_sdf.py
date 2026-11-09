"""CST pilot SDF pod (papers/cst_constitutional_steering/scripts/cst_pilot_sdf_pod.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"), str(ROOT / "tests/scripts")]

import cst_pilot_sdf_pod as P  # noqa: E402
import overlap_hashes as OH  # noqa: E402

PROSE = " ".join(["The flame draws the melted wax up the wick by capillary action"] * 70)


def _rows(final: str, n: int = 50) -> list[dict]:
    return [{"final": final} for _ in range(n)]


def test_gate_passes_prose_and_stops_on_each_failure():
    # most probable failure: the pilot gate lets through chat-formatted or think-tagged documents,
    # documents outside the word range, or documents copying an eval set, so SDF trains on them
    import numpy as np

    eval_text = "a completely separate sentence that sits in an evaluation set and nowhere else"
    index = np.unique(OH.hashes(eval_text))
    assert P.gate(_rows(PROSE), index)["proceed"]
    assert not P.gate(_rows("User: hi\nAssistant: " + PROSE), index)["proceed"]
    assert not P.gate(_rows("<think>\n\n</think>" + PROSE), index)["proceed"]
    assert not P.gate(_rows("too short to count"), index)["proceed"]
    assert not P.gate(_rows(PROSE + " " + eval_text), index)["proceed"]
    assert not P.gate(_rows(PROSE, 49), index)["proceed"]  # fewer than the gate's 50


def test_items_block_names_each_item_with_its_claim():
    # failure: the template's tenet block carries ids without claims (or D-neutral's statements are
    # looked up in the tenet file), so the corpus is written about the wrong items
    plan = json.loads((P.CST / "data/pilot_corpus_plan_v1.json").read_text())["arms"]
    for arm in ("d_doc", "d_neutral"):
        block = P.items_text(arm, plan[arm][0]["items"])
        assert block.count("\n") == 2 and all(f"[{i}]" in block for i in plan[arm][0]["items"])


@pytest.mark.slow
def test_corpus_stops_at_a_failed_gate(tmp_path):
    # failure: a failed pilot gate does not stop generation, so the pod spends its hours on a corpus
    # that will not be trained (the dry stand-in writes random text, which fails the gate)
    P.stage_corpus(tmp_path, "d_doc", True, target=10**9)
    g = json.loads((tmp_path / "corpus_gate_d_doc.json").read_text())
    chunks = sorted((tmp_path / "corpus" / "d_doc").glob("chunk_*.jsonl"))
    assert not g["proceed"] and len(chunks) == 1
