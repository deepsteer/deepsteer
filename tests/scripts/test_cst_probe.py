"""CST probe pod (Amendment 9 item 4): papers/cst_constitutional_steering/scripts/
cst_probe_pod.py."""

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

import cst_probe_pod as P  # noqa: E402


def test_engine_configs_reach_the_engine_kwargs():
    # most probable failure: a configuration's override never reaches vLLM, so every config runs
    # ST's engine and the comparison measures nothing
    st = P.engine_kwargs("st")
    assert "speculative_config" in st and st["enable_prefix_caching"] is True
    assert st["max_model_len"] == 44032 and st["max_num_seqs"] == 64
    nm = P.engine_kwargs("no_mtp")
    assert "speculative_config" not in nm and nm["enable_prefix_caching"] is True
    assert P.engine_kwargs("no_mtp_seqs128")["max_num_seqs"] == 128
    npf = P.engine_kwargs("no_prefix_no_mtp")
    assert npf["enable_prefix_caching"] is False and "speculative_config" not in npf


def test_blind_pass_never_shows_the_final_answer():
    # most probable failure (Amendment 9 addendum item 2): the read of record still shows the
    # subject's final answer, so S_concl is not blind and the judge can copy the action
    import cst_judge as J

    p = J.judge_prompt("s", [("o1", "x")], "trace text", None, [], letters={"o1": "B"})
    assert "<final_answer>" not in p and "final answer is not shown" in p
    shown = J.judge_prompt("s", [("o1", "x")], "trace text", "Answer: B", [], letters={"o1": "B"})
    assert "<final_answer>\nAnswer: B\n</final_answer>" in shown


def test_permuted_retest_relabels_and_maps_back():
    # most probable failure: the permuted pass maps a reply back through the wrong table, so its
    # kappa measures the mapping, not the judge's position bias; or it never moves an option
    order = [("A", "o2"), ("B", "o1"), ("C", "o3")]
    same = P.presentation(order, False, 0)
    assert [(j, o, L) for j, o, L in same] == [("o2", "o2", "A"), ("o1", "o1", "B"),
                                                ("o3", "o3", "C")]
    for i in range(20):
        perm = P.presentation(order, True, i)
        assert [o for _, o, _ in perm] != [o for _, o, _ in same]  # the order moves
        assert [j for j, _, _ in perm] == ["o1", "o2", "o3"]  # relabelled by position
        assert {(o, L) for _, o, L in perm} == {("o2", "A"), ("o1", "B"), ("o3", "C")}


def test_tenet_kappa_counts_every_shortlist_cell():
    # most probable failure: kappa over the named tenets only, which drops the agreement on
    # tenets both passes leave out and is undefined when a pass names nothing
    a = [{"ok": True, "applied_tenets": ["A1"], "shortlist": ["A1", "A2", "A3"],
          "concluded_option": "o1"}] * 6
    b = [{"ok": True, "applied_tenets": ["A1", "A2"], "shortlist": ["A1", "A2", "A3"],
          "concluded_option": "o1"}] * 6
    assert len(P.tenet_cells(a)) == 18
    same = P.kappa_ci(a, a, "tenets")
    assert same["kappa"] == 1.0 and same["ci95"] == [1.0, 1.0] and not same["flagged"]
    diff = P.kappa_ci(a, b, "tenets")
    assert diff["kappa"] < 1.0 and diff["flagged"]


@pytest.mark.slow
def test_dry_run_end_to_end(tmp_path):
    # most probable failure: the judges read rollouts the engine stage did not write, or the
    # summary misses the validity and retest reads the probe exists for
    P.stage_engine(tmp_path, "st", True)
    # the tiny model emits 8 tokens and no letter; give each rollout a parsed answer (its first
    # option's letter) so the validity read has subject answers to compare with
    f = tmp_path / "gen_a0.jsonl"
    rows = [json.loads(x) for x in open(f)]
    for r in rows:
        first = sorted(r["order"], key=lambda lo: lo[1])[0][0]
        r["parse"] = {**(r["parse"] or {}), "letter": first}
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    for j in P.PASSES:
        P.stage_judge(tmp_path, j, True)
    P.stage_summary(tmp_path)
    s = json.loads((tmp_path / "summary.json").read_text())
    g = s["gpt_oss_120b"]
    assert g["match_subject_answer_blind"]["n"] == 4 and g["shown_minus_blind"] == 0.0
    for tag in ("retest_same_order", "retest_permuted"):
        assert g[tag]["concluded_option"]["kappa"] == 1.0  # the canned replies map back exactly
    assert g["gate"]["task"] == "applied tenets, permuted retest"
    assert g["letter_mapped"]["shown_s0"] > 0 and g["letter_mapped"]["blind_perm"] == 0
    assert "disagreement_10pct" in s["olmo31_32b_instruct"]
    assert "st" in s["engine"]


def test_effort_high_reaches_the_harmony_render():
    # most probable failure (CST-A12, the probe): the judge's reasoning effort is hardcoded, so the
    # effort-high passes score at low again
    import mrb_g1_pod as M
    from huggingface_hub import try_to_load_from_cache
    from transformers import AutoTokenizer

    rev = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
    if not try_to_load_from_cache("openai/gpt-oss-120b", "tokenizer.json", revision=rev):
        pytest.skip("gpt-oss-120b tokenizer not cached")
    from deepsteer.reasoning.think_io import CoTFormat

    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", revision=rev)
    cfg = M.harmony_cfg()
    hi = tok.decode(P.judge_ids(tok, "x", "harmony", CoTFormat.HARMONY_ANALYSIS, cfg, "high"))
    lo = tok.decode(P.judge_ids(tok, "x", "harmony", CoTFormat.HARMONY_ANALYSIS, cfg, "low"))
    assert "Reasoning: high" in hi and "Reasoning: low" in lo
    assert P.EFFORT["gpt_oss_120b_high"] == "high"
    assert P.MODEL["gpt_oss_120b_high"] == "gpt_oss_120b"


def test_majority_of_three_keeps_tenets_named_twice():
    # most probable failure: the majority keeps a tenet named once (a union) or drops one named
    # twice (an intersection), which moves its kappa for reasons that are not the judge's
    r = lambda ts, co: {"ok": True, "shortlist": ["A1", "A2", "A3"], "applied_tenets": ts,  # noqa
                        "concluded_option": co}
    m = P.majority([[r(["A1", "A2"], "o1")], [r(["A1"], "o1")], [r(["A2", "A3"], "o2")]])[0]
    assert m["applied_tenets"] == ["A1", "A2"] and m["concluded_option"] == "o1"
    split = P.majority([[r([], "o1")], [r([], "o2")], [r([], "o3")]])[0]
    assert split["concluded_option"] == "invalid"


@pytest.mark.slow
def test_dry_run_effort_high_summary(tmp_path):
    # most probable failure: the summary pairs the wrong passes (a triple against itself), which
    # reads kappa 1 whatever the judge does
    test_dry_run_end_to_end(tmp_path)
    P.stage_judge(tmp_path, "gpt_oss_120b_high", True)
    P.stage_summary_high(tmp_path)
    s = json.loads((tmp_path / "summary_high.json").read_text())
    for k in ("single_same_order", "single_permuted", "majority3_same_order",
              "majority3_permuted"):
        assert {"tenets_all", "tenets_deference", "tenets_other", "concluded_option"} <= set(s[k])
    pr = s["price_per_arm_read_h"]
    assert pr["majority_on_operator_and_operator_pr"] > pr["single_sample"]
    rows = [json.loads(x) for x in open(tmp_path / "judge_gpt_oss_120b_high_h_s3.jsonl")]
    assert len(rows) == 4


def test_per_tenet_kappa_reads_only_rollouts_that_list_the_tenet():
    # most probable failure: a tenet's kappa counted over rollouts whose shortlist does not carry
    # it (all agreeing zeros), which inflates every rare tenet past the bar
    def r(short, named):
        return {"ok": True, "shortlist": short, "applied_tenets": named, "concluded_option": "o1"}
    p1 = [r(["A1", "A2"], ["A1"]), r(["A1", "A2"], []), r(["A3"], [])]
    p2 = [r(["A1", "A2"], ["A1"]), r(["A1", "A2"], ["A2"]), r(["A3"], [])]
    res = P.per_tenet_kappa([p1, p2])
    assert res["tenets"]["A2"]["items"] == 2 and res["tenets"]["A3"]["items"] == 1
    # never named in any pass: no kappa, not passing (agreeing absences are not reliability)
    assert res["tenets"]["A3"]["kappa"] is None and "A3" not in res["passing"]
    assert res["tenets"]["A1"]["kappa"] == 1.0 and "A1" in res["passing"]
    assert "A2" not in res["passing"]
    assert res["coverage_of_named_cells"] == 0.5  # A1 named on 1 rollout, A2 on 1
