"""CST S0a pod driver (papers/cst_constitutional_steering/scripts/cst_s0a_pod.py)."""

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

import cst_s0a_pod as S  # noqa: E402

CST = ROOT / "papers/cst_constitutional_steering"


def test_samples_are_drawn_from_ids_the_driver_produces():
    # most probable failure: the committed samples (retest, Olmo's 10%, frontier 200) name rollout
    # ids the generation never writes, so those reads come back empty on the pod
    sc = S.scenarios(False)
    made = set()
    for arm in ("a0", "a0s"):
        for part in S.PARTS:
            for set_, cond, sid, k in S.part_units(part, sc):
                made.add(f"{arm}|{set_}|{cond}|{sid}|{k}")
    smp = json.loads((CST / "data/s0a_samples_v1.json").read_text())
    for ids in (smp["retest_a0_operator_clear"], smp["frontier_200"], *smp["olmo_10pct"].values()):
        assert set(ids) <= made
    per_arm = sum(len(S.part_units(p, sc)) for p in S.PARTS)
    assert per_arm == 586 * 16 + 132 * 16 + 102 * 8  # the registered design


def test_a0s_carries_the_constitution_in_every_condition():
    # most probable failure: one condition (operator_pr builds its own system message) renders
    # A0s without the constitution, or A0 with it
    from deepsteer.kdg.schema import assign_letters

    sc = S.scenarios(True)
    for part in S.PARTS:
        set_, cond = part.split("_", 1)
        s = sc["objs"][sc["ids"][set_][0]]
        o = assign_letters(s, 0)
        a0 = S.messages(s, o, set_, cond, "a0", None)
        a0s = S.messages(s, o, set_, cond, "a0s", "## Overview\nBODY-MARKER")
        assert "BODY-MARKER" in a0s[0]["content"] and a0s[0]["role"] == "system"
        assert not any("BODY-MARKER" in m["content"] for m in a0)
        if cond.startswith("operator"):  # the order follows the constitution, same message
            assert a0s[0]["content"].index("BODY-MARKER") < len(a0s[0]["content"]) - 10


def test_teacher_forced_sum_aligns_positions(tmp_path):
    # most probable failure: an off-by-one between a token and the logit that predicts it, which
    # scores the paraphrase on the wrong positions and makes the recall margin noise
    import torch

    path, eng = S.subject(True)
    ids = list(range(10, 30))
    got = S.forced_logprobs(eng, [ids], [12], True)[0]
    with torch.no_grad():
        lp = torch.log_softmax(eng.m(input_ids=torch.tensor([ids])).logits[0].float(), -1)
    want = sum(float(lp[i - 1, ids[i]]) for i in range(12, 20))
    assert got == pytest.approx(want, rel=1e-5)


@pytest.mark.slow
def test_gen_resumes_and_scores_end_to_end(tmp_path):
    # most probable failure: a resumed session regenerates (or skips) chunks, or a half-written
    # chunk is taken as done; then the scorer misses rollouts the generation wrote
    S.stage_gen(tmp_path, "a0", ["panel_operator", "h1_operator"], True)
    files = sorted((tmp_path / "gen" / "a0").glob("*.jsonl"))
    stamp = {f.name: f.stat().st_mtime_ns for f in files}
    (tmp_path / "gen" / "a0" / "h1_plain_000.tmp").write_text("partial")
    S.stage_gen(tmp_path, "a0", ["panel_operator", "h1_operator", "h1_plain"], True)
    after = {f.name: f.stat().st_mtime_ns for f in (tmp_path / "gen" / "a0").glob("*.jsonl")}
    assert all(after[n] == t for n, t in stamp.items())  # nothing regenerated
    assert "h1_plain_000.jsonl" in after  # the partial chunk was redone, not trusted
    rows = S.scored_rows(tmp_path, "a0")
    assert len(rows) == 2 * 4 + 2 * 8 + 2 * 8
    assert all(r["seed"] == S.seed_of(r["rollout_id"]) for r in rows)
    S.stage_score(tmp_path, "a0", "gpt_oss_120b", True)
    blind = [json.loads(x) for f in (tmp_path / "score/a0/gpt_oss_120b").glob("blind_*.jsonl")
             for x in open(f)]
    assert {b["rollout_id"] for b in blind} == {r["rollout_id"] for r in rows}
    assert all(b["ok"] and not b["permuted"] for b in blind)


@pytest.mark.slow
def test_dose0_recall_and_capability_dry(tmp_path):
    # most probable failure: a stage reads inputs the pod does not have (the capability set, the
    # recall windows) or writes nothing per unit
    S.stage_dose0(tmp_path, "a0s", True)
    d0 = [json.loads(x) for x in open(tmp_path / "dose0/a0s.jsonl")]
    assert len(d0) == sum(2 for _ in S.PARTS) and all(len(r["logprobs"]) == len(r["letter_ids"])
                                                       for r in d0)
    S.stage_recall(tmp_path, True)
    rc = [json.loads(x) for x in open(tmp_path / "recall/a0.jsonl")]
    assert len(rc) == 9 and {r["variant"] for r in rc} == {"true", "p1", "p2"}
    S.stage_capability(tmp_path, "a0", True)
    cap = [json.loads(x) for x in open(tmp_path / "cap/a0.jsonl")]
    assert len(cap) == 2 and all("correct" in r for r in cap)
