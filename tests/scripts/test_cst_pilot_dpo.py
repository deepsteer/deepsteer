"""CST pilot DPO pod (papers/cst_constitutional_steering/scripts/cst_pilot_dpo_pod.py)."""

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

import cst_pilot_dpo_pod as D  # noqa: E402


def _candidates(tmp: Path, tok, pids: list[str]) -> None:
    c = tmp / "candidates"
    c.mkdir(parents=True)
    rows = []
    for pid in pids:
        p = tok.encode("Should I tell my friend?", add_special_tokens=False)
        for k in range(4):
            text = f"<think>{'weigh ' * (3 + 5 * k)}</think>\n\nanswer {k}"
            rows.append({"prompt_id": pid, "k": k, "prompt_ids": p,
                         "gen_ids": tok.encode(text, add_special_tokens=False), "text": text,
                         "hit_cap": False})
    (c / "none.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    doc = [{"prompt_id": pid, "text": "<think>the constitution says</think>\n\nkind answer",
            "gen_ids": tok.encode("<think>the constitution says</think>\n\nkind answer",
                                  add_special_tokens=False), "hit_cap": False} for pid in pids]
    (c / "with_doc.jsonl").write_text("".join(json.dumps(r) + "\n" for r in doc))
    rw = [{"prompt_id": pid, "text": "a kinder rewrite", "hit_cap": False} for pid in pids]
    (c / "rewrite.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rw))


def _tok():
    import cst_s0a_pod as S

    return S.V._tok(S.V._model_path(True))  # noqa: SLF001


def test_filter_keeps_the_chosen_side_under_the_flip(tmp_path):
    # most probable failure: the filter keeps pairs where the judge picked a fixed letter, not the
    # chosen reply (the A/B order is flipped per pair), so it keeps the first answer as often as
    # the rewrite; and the 250-pair gate must stop a run with too few
    pids = [p["prompt_id"] for p in D.prompts(False)[:40]]
    _candidates(tmp_path, _tok(), pids)
    rec = D.stage_filter(tmp_path, True)  # the dry judge always answers "A"
    rows = [json.loads(x) for x in open(tmp_path / "filter.jsonl")]
    assert all(r["keep"] == (r["chosen_is"] == "A") for r in rows)
    assert 0 < rec["kept"]["p_answer"] < 40 and not rec["proceed"]


def test_pairs_differ_in_the_answer_only_for_p_answer(tmp_path):
    # failure: P-answer's chosen and rejected carry different traces (the answer-only DPO then sees
    # a trace difference through the unmasked answer's context), or the mask leaks the trace; and
    # P-long pairs the longest against the shortest rollout without the document
    tok = _tok()
    pids = [p["prompt_id"] for p in D.prompts(False)[:6]]
    _candidates(tmp_path, tok, pids)
    (tmp_path / "filter.json").write_text(json.dumps(
        {"kept_ids": {"p_answer": pids, "p_trace": pids}}))
    rec = D.stage_pairs(tmp_path, True)
    pa = [json.loads(x) for x in open(tmp_path / "pairs_p_answer.jsonl")]
    end = D.think_end(tok)
    for r in pa:
        cut_c, cut_r = r["chosen"].index(end) + 1, r["rejected"].index(end) + 1
        assert r["chosen"][:cut_c] == r["rejected"][:cut_r]  # the same trace
        assert sum(r["chosen_mask"][:cut_c]) == 0 and all(r["chosen_mask"][cut_c:])
    pl = [json.loads(x) for x in open(tmp_path / "pairs_p_long.jsonl")]
    assert all(len(r["chosen"]) > len(r["rejected"]) for r in pl) and rec["n"]["p_long"] == 6
    assert rec["p_trace_chosen_citing_share"] == 1.0 and rec["length_ratio"]["p_long"] > 1


@pytest.mark.slow
def test_candidates_stage_writes_every_rollout(tmp_path):
    # failure: a candidate file is missing a prompt or the four rollouts without the document
    D.stage_candidates(tmp_path, True)
    none = [json.loads(x) for x in open(tmp_path / "candidates/none.jsonl")]
    assert len(none) == 3 * D.K_NONE
    assert sum(1 for _ in open(tmp_path / "candidates/with_doc.jsonl")) == 3
    assert sum(1 for _ in open(tmp_path / "candidates/rewrite.jsonl")) == 3
