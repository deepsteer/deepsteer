"""MB-G1 driver (papers/mb_moralbench/scripts/mb_g1_run.py): counterbalancing and the dry run."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.moralbench import MoralBenchItem, default_cache_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "papers" / "mb_moralbench" / "scripts"))

import mb_g1_run as G  # noqa: E402


def _item(set_name: str, prompt: str, key: dict[str, float]) -> MoralBenchItem:
    return MoralBenchItem(set_name, "harm_1", "harm", MoralFoundation.CARE_HARM, prompt, key)


def test_binary_flip_moves_the_score_with_the_label():
    # most probable failure: letters flipped in the text but the key left on the letter, so the
    # label "Agree" (now at B) keeps scoring as if it were still A
    it = _item("MFQ_30", "Stem?\n\nInvented statement.\n\nA. Agree B. Disagree",
               {"A": 3.2, "B": 1.8})
    cb = G.counterbalance(it)
    assert cb.prompt.endswith("A. Disagree B. Agree")
    assert cb.key == {"A": 1.8, "B": 3.2}  # B now reads "Agree" and carries Agree's score
    it2 = _item("MFQ_30", "Stem?\n\nX.\n\nA. Correct B. Not Correct", {"A": 2.0, "B": 3.0})
    assert G.counterbalance(it2).prompt.endswith("A. Not Correct B. Correct")


def test_comparative_swap_keeps_each_line_prefix():
    # most probable failure: the swap rebuilds lines as "A. text" and drops an item's own
    # formatting (one upstream item has no space after "A.")
    it = _item("6_concepts_compare", "Pick one.\n\nA.first statement\n\nB. second statement",
               {"A": 1.0, "B": 0.0})
    cb = G.counterbalance(it)
    assert cb.prompt == "Pick one.\n\nA.second statement\n\nB. first statement"
    assert cb.key == {"A": 0.0, "B": 1.0}
    assert G.counterbalance(cb).prompt == it.prompt


def _zephyr_cached() -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(G.MODELS["zephyr"]["repo"],
                                      revision=G.MODELS["zephyr"]["revision"],
                                      local_files_only=True)
        return default_cache_dir().exists()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _zephyr_cached(), reason="needs the Zephyr tokenizer and MoralBench cached")
def test_dry_run_rows_hold_letters_not_scores(tmp_path):
    # most probable failure: a tracked row carries a per-item score (which, with the letter,
    # rebuilds the answer key; MB_PLAN v0.5 §9.2), or the cells draw from different seeds
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    rc = subprocess.run([sys.executable, str(REPO / "papers/mb_moralbench/scripts/mb_g1_run.py"),
                         "--model", "zephyr", "--dry-run", "--device", "cpu", "--out",
                         str(tmp_path)], capture_output=True, text=True, env=env, cwd=REPO)
    assert rc.returncode == 0, rc.stderr[-2000:]
    d = tmp_path / "outputs" / "g1" / "zephyr"
    rows = {c: [json.loads(x) for x in (d / f"{c}.jsonl").read_text().splitlines()]
            for c in G.CELLS}
    for cell_rows in rows.values():
        assert len(cell_rows) == 8 * 2
        assert not any("score" in k for r in cell_rows for k in r)
    seeds = {c: [(r["set"], r["item_id"], r["seed"]) for r in rows[c]] for c in G.CELLS}
    assert seeds["t07"] == seeds["t10"] == seeds["cb07"]
    r2 = [json.loads(x) for x in (d / "r2.jsonl").read_text().splitlines()]
    assert {r["order"] for r in r2} == {"original", "counterbalanced"} and len(r2) == 16
    assert (tmp_path / "outputs_raw" / "g1" / "zephyr" / "t07.jsonl").exists()
