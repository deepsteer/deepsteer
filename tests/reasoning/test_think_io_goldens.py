"""think_io against goldens written from the code before the MRB-A8 fix (goldens/think_io_v1.json).

The fix (author-approved) changes one thing: the reasoning span of a harmony rollout whose analysis
channel the model opened itself, which was empty and is now that analysis message. Every final
answer, every prompt-opened or think-tag span, and both token-index functions match the goldens.
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


# the one intended change: generated-opener harmony rollouts get their analysis message
FIXED_REASONING = {
    "harmony/gen_open/closed": "The user asks X. Policy says Y.",
    "harmony/gen_open/unclosed": "The user asks X. Policy says Y and then",
    "harmony/gen_open/analysis_only_ended": "Short thought.",
    "harmony/gen_open/empty_analysis": "",
    "harmony/gen_open/final_only": "",
    "harmony/gen_open/final_unterminated": "Thought.",
    "harmony/gen_open/with_prompt": "Thought about Q.",
}


def test_split_rollout_on_every_caller_shape():
    # most probable failure: a split change that moves a final answer, or the reasoning span of a
    # think-tag or prompt-opened harmony rollout (D3/W4 with a prefill)
    got, gold = G.synthetic_split(), GOLD["synthetic_split"]
    assert got.keys() == gold.keys()
    for name, (reasoning, final) in got.items():
        assert final == gold[name][1], name
        want = FIXED_REASONING[name] if name in FIXED_REASONING else gold[name][0]
        assert reasoning == want, name
    # most probable failure of the old split (MRB-A8): a generated opener gave an empty span
    assert all(gold[n][0] == "" for n in FIXED_REASONING)


def test_token_index_functions_unchanged():
    # most probable failure: the fix leaking into cot_token_boundary or cot_content_start, which
    # would move every banked Paper 7 / D1 position
    assert G.fake_token_fns() == GOLD["fake_token_fns"]


def test_banked_w4_rollouts():
    # most probable failure: the same, on 40 real GPT-OSS-20B rollouts through D3's path
    got = G.w4_banked()
    if got is None or GOLD["w4_banked"] is None:
        pytest.skip("needs the banked W4 npz and the GPT-OSS-20B tokenizer")
    for row, gold in zip(got, GOLD["w4_banked"], strict=True):
        moved = {"reasoning", "reasoning_empty"} if row["gen_open"] else set()
        assert {k: v for k, v in row.items() if k not in moved} == \
            {k: v for k, v in gold.items() if k not in moved}, row["row"]
        if row["gen_open"]:  # all 10 baselines reached the final channel after real analysis
            assert gold["reasoning_empty"] and not row["reasoning_empty"], row["row"]


def test_banked_mrb_judge_replies():
    # most probable failure: a moved final answer on MRB's judge path (2,236 real replies)
    got = G.mrb_judge_banked()
    if got is None or GOLD["mrb_judge_banked"] is None:
        pytest.skip("needs MRB-G1 VALIDATE's raw judge replies")
    assert (got["n"], got["finals"]) == (GOLD["mrb_judge_banked"]["n"],
                                         GOLD["mrb_judge_banked"]["finals"])


def test_content_window_drops_the_harmony_control_tokens():
    # most probable failure (MRB-A9): the window still holds the generated header or the
    # <|end|><|start|>assistant tail, or a think-tag window that moved
    from deepsteer.reasoning import think_io as ti
    tok = G.CharTok()
    for name, (fmt, text) in G.SYNTHETIC.items():
        ids = tok.encode(text)
        start, end = ti.cot_content_window(tok, ids, fmt)
        span = tok.decode(ids[start:end])
        assert not any(m in span for m in G.CharTok.SPECIAL), name
        if fmt == ti.CoTFormat.THINK_TAGS:
            assert (start, end) == (0, ti.cot_token_boundary(tok, ids, fmt)), name
        else:
            assert span.strip() == ti.split_rollout(text, fmt)[0], name


def test_content_window_on_banked_gpt_oss_rollouts():
    # most probable failure: the same on the real tokenizer, where the six control tokens are
    # single special tokens: a closed analysis window is boundary - start - 3 tokens long
    rows = G.w4_windows()
    if rows is None:
        pytest.skip("needs the banked W4 npz and the GPT-OSS-20B tokenizer")
    for r in rows:
        assert not r["has_marker"], r["row"]
        assert r["span"] == r["split_reasoning"], r["row"]
        if r["gen_open"] and r["final_reached"]:
            assert (r["start"], r["end"]) == (3, r["boundary"] - 3), r["row"]
    assert sum(r["gen_open"] for r in rows) == 10
