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
    # every scenario and condition at k 0, plus ST's four E cells on the panel
    assert len(d0) == 2 * len(S.PARTS) + 2 * len(S.E_CELLS)
    assert all(len(r["logprobs"]) == len(r["letter_ids"]) for r in d0)
    assert {r["condition"] for r in d0} >= set(S.E_CELLS)
    S.stage_recall(tmp_path, True)
    rc = [json.loads(x) for x in open(tmp_path / "recall/a0.jsonl")]
    assert len(rc) == 9 and {r["variant"] for r in rc} == {"true", "p1", "p2"}
    S.stage_capability(tmp_path, "a0", True)
    cap = [json.loads(x) for x in open(tmp_path / "cap/a0.jsonl")]
    assert len(cap) == 2 and all("correct" in r for r in cap)


def test_long_tasks_vote_on_the_record_splits_and_score_only_the_overlap():
    # most probable failure: the third family votes on the wrong pair set (the 207 one-sided count,
    # or matched pairs), or Layer A scores pairs outside the 51 where the direction of record
    # coincides with the human verdict
    olmo = S.long_tasks("olmo31_32b_instruct", False)
    pairs = [p for _, _, m in olmo["tcore_votes"] for p in m["pairs"]]
    tv = json.loads((CST / "outputs/t01_tcore_votes_v2.json").read_text())
    assert sorted(pairs) == sorted(tv["pending_split"]) and len(pairs) == 214
    assert all(len(set(tv["votes"][p].values())) == 2 for p in pairs)  # every one a split
    assert len(olmo["tconf_votes"][0][2]["pairs"]) == 13
    gpt = S.long_tasks("gpt_oss_120b", False)
    assert len(gpt["relabel"]) == 72
    la = json.loads((CST / "data/layer_a_scruples_v1.json").read_text())["items"]
    ids = [i for _, _, m in gpt["layer_a"] for i in m["ids"]]
    assert sorted(ids) == sorted(i for i, r in la.items() if r["coincides"]) and len(ids) == 51


@pytest.mark.slow
def test_long_stage_runs_olmo_twice(tmp_path):
    # most probable failure: the self-agreement read has one repetition, so it reads 1.0
    S.stage_long(tmp_path, "olmo31_32b_instruct", True)
    files = sorted(f.name for f in (tmp_path / "long/olmo31_32b_instruct").glob("*.jsonl"))
    assert files == ["tconf_votes_r0.jsonl", "tconf_votes_r1.jsonl", "tcore_votes_r0.jsonl",
                     "tcore_votes_r1.jsonl"]
    S.stage_long(tmp_path, "gpt_oss_120b", True)
    assert {f.name for f in (tmp_path / "long/gpt_oss_120b").glob("*.jsonl")} == {
        "relabel_r0.jsonl", "layer_a_r0.jsonl"}


def test_claude_reader_reports_unscored_after_three_attempts(tmp_path, monkeypatch):
    # most probable failure (Amendment 11 addendum item 3): a stopped rollout is dropped instead of
    # reported unscored, or a resumed run grants it three fresh attempts (more than three in all),
    # or re-calls a rollout already scored
    import cst_claude_reader as R

    r = {"rollout_id": "a0|panel|operator|F1-A-00|0", "scenario_id": "F1-A-00", "k": 0,
         "condition": "operator", "order": [["A", "o1"], ["B", "o2"]],
         "text": "<think>I'll pick A.</think>Answer: A"}
    calls = []

    def stop(sys_, text):
        calls.append(1)
        return {"returncode": 0, "is_error": True, "result": "API Error: safeguards flagged",
                "safety_stops": 1}

    monkeypatch.setattr(R, "prompt", lambda r: ("p", {"options": {"o1", "o2"},
                                                      "shortlist": ["A036"],
                                                      "letters": {"o1": "A", "o2": "B"}}))
    assert R.read_one(r, tmp_path, stop) == "unscored" and len(calls) == 3
    assert R.read_one(r, tmp_path, stop) == "unscored" and len(calls) == 3  # no fresh attempts
    r2 = {**r, "rollout_id": "a0|panel|operator|F1-A-00|1", "k": 1}
    seq = iter([stop, lambda s, t: {"returncode": 0, "is_error": False, "safety_stops": 0,
                                     "result": '{"concluded_option": "A", '
                                               '"applied_tenets": ["A036"]}'}])
    assert R.read_one(r2, tmp_path, lambda s, t: next(seq)(s, t)) == "scored"
    assert R.read_one(r2, tmp_path, stop) == "cached"
    rec = json.loads((tmp_path / "a0__panel__operator__F1-A-00__1.json").read_text())
    assert rec["attempt"] == 2 and rec["concluded_option"] == "o1" and rec["letter_mapped"]


def test_a0s_runs_at_its_capped_concurrency_and_small_chunks(tmp_path):
    # most probable failure (CST-A13, the first S0a-2): A0s inherits the engine of record's 128
    # sequences, overcommits the KV cache with its 35k-token prefix and thrashes; or a stop loses a
    # 1,024-rollout chunk
    assert S.engine_kwargs("a0s")["max_num_seqs"] == S.A0S_MAX_SEQS == 32
    assert S.engine_kwargs("a0")["max_num_seqs"] == 128
    a0s, a0 = S.engine_kwargs("a0s"), S.engine_kwargs("a0")
    assert {k: v for k, v in a0s.items() if k != "max_num_seqs"} == {
        k: v for k, v in a0.items() if k != "max_num_seqs"}  # nothing else differs
    assert S.CHUNK_BY_ARM == {"a0": 1024, "a0s": 256}


def test_claude_reader_does_not_count_a_usage_limit_as_an_attempt(tmp_path, monkeypatch):
    # most probable failure (the first A0 read): the account's usage limit is counted as three
    # failed attempts, so hundreds of rollouts are reported unscored for a reason that is not theirs
    import cst_claude_reader as R

    r = {"rollout_id": "a0|panel|operator|F5-A-00|0", "scenario_id": "F5-A-00", "k": 0,
         "condition": "operator", "order": [["A", "o1"]], "text": "<think>x</think>A"}
    monkeypatch.setattr(R, "prompt", lambda r: ("p", {"options": {"o1"}, "shortlist": [],
                                                      "letters": {"o1": "A"}}))
    lim = {"returncode": 1, "is_error": True, "safety_stops": 0,
           "result": "You've hit your 5-hour limit · resets 7:10pm (Pacific/Honolulu)"}
    with pytest.raises(R.UsageLimit):
        R.read_one(r, tmp_path, lambda s, t: lim)
    assert not list(tmp_path.glob("*.json"))  # no attempt recorded
