"""think_io against goldens written from the code before the MRB-A8 fix (goldens/think_io_v1.json).

Local-only goldens (the banked W4 rollouts and MRB-G1 VALIDATE's raw judge replies, both
gitignored) skip where their inputs are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import think_io_goldens as G  # noqa: E402

GOLD = json.loads(G.GOLDEN.read_text())


def test_split_rollout_on_every_caller_shape():
    # most probable failure: a split change that moves a final answer, or the reasoning span of a
    # think-tag or prompt-opened harmony rollout (D3/W4 with a prefill)
    assert G.synthetic_split() == GOLD["synthetic_split"]


def test_token_index_functions_unchanged():
    # most probable failure: the fix leaking into cot_token_boundary or cot_content_start, which
    # would move every banked Paper 7 / D1 position
    assert G.fake_token_fns() == GOLD["fake_token_fns"]


def test_banked_w4_rollouts():
    # most probable failure: the same, on 40 real GPT-OSS-20B rollouts through D3's path
    got = G.w4_banked()
    if got is None or GOLD["w4_banked"] is None:
        pytest.skip("needs the banked W4 npz and the GPT-OSS-20B tokenizer")
    assert got == GOLD["w4_banked"]


def test_banked_mrb_judge_replies():
    # most probable failure: a moved final answer on MRB's judge path (2,236 real replies)
    got = G.mrb_judge_banked()
    if got is None or GOLD["mrb_judge_banked"] is None:
        pytest.skip("needs MRB-G1 VALIDATE's raw judge replies")
    assert got == GOLD["mrb_judge_banked"]
