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


def test_withheld_pass_hides_the_final_answer():
    # most probable failure: the copy check still shows the subject's final answer, so agreement
    # with it cannot separate reading the reasoning from copying the answer
    import cst_judge as J

    p = J.judge_prompt("s", [("o1", "x")], "trace text", P.WITHHELD, [], letters={"o1": "B"})
    assert "<final_answer>\n(withheld)\n</final_answer>" in p
    assert "Answer: B" not in p


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
    assert g["validity_answer_shown"]["n"] == 4 and g["validity_answer_shown"]["rate"] == 1.0
    assert g["validity_answer_withheld"]["n"] == 4
    assert {"concluded_option", "applied_tenets"} <= set(g["retest_t1"])
    assert g["letter_mapped"]["t0"] > 0  # the canned replies name letters
    assert "disagreement_10pct" in s["olmo31_32b_instruct"]
    assert "st" in s["engine"]
