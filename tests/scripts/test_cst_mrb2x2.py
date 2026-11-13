"""CST's MoReBench pod, stage 1 (Amendment 46 item B):
papers/cst_constitutional_steering/scripts/cst_mrb2x2_pod.py and mrb2x2_read.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts"), str(ROOT / "tests/scripts")]

import cst_mrb2x2_pod as B  # noqa: E402


def test_a0s_rows_pair_with_mrbs_base_rows():
    # most probable failure (Amendment 46 item B1): A0s's rollouts carry other seeds or job ids
    # than MRB's saved base rows, so the paired read pairs nothing or pairs the wrong rollouts
    import mrb_p2_pod as M

    jobs = B.a0s_jobs(None, None, True)
    assert {j["job_id"].split("/")[1] for j in jobs} == {"a0s"}
    assert all(j["seed"] == M.seed(B.MODEL, j["sid"], j["k"]) for j in jobs)
    assert {j["k"] for j in jobs} == {0, 1}


def test_a0s_render_puts_the_constitution_first_in_a_system_message():
    # most probable failure: A0s's document lands after MoReBench's prompt or in the user turn, so
    # the arm is not the in-context constitution of record (A0s's render)
    import cst_pressure as CP

    msgs = CP.a0s_messages([{"role": "user", "content": "DILEMMA"}], "BODY")
    assert msgs[0]["role"] == "system" and "BODY" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "DILEMMA"}


def test_a0s_engine_fits_the_longest_prompt_with_mrbs_cap():
    # most probable failure: the longest A0s prompt plus MRB's 10,500 cap exceeds max_model_len
    import mrb_g1_pod as P

    assert B.A0S_MAX_LEN >= 34_722 + P.MAX_TOKENS


def _adapter(out: Path, tag: str) -> None:
    import pilot_pod_lib as PL

    _, wb, _ = PL.load_model(True)
    adir = out / "adapters" / tag / "1"
    wb.model.save_pretrained(str(adir))
    (out / f"train_{tag}.json").write_text(json.dumps({"trained": True,
                                                       "adapters": {"1": str(adir)}}))


@pytest.mark.slow
def test_dry_session_runs_every_stage_in_order(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage breaks on the pod after hours; the
    # stages run on the tiny model in the runner's order, the gate is written, each arm's rows land
    # under its own job ids and the session check stays out of the judge's inputs
    import cst_sdf_norec_pod as Q
    import mrb_g1_pod as P
    import mrb_p2_pod as M

    for t in B.ADAPTERS:
        _adapter(tmp_path, t)
    F = B.F
    path, wb, _ = F.PL.load_model(True)
    rec = F.PL.smoke_hf_side(wb, path, tmp_path, True, [Path(F.cand_dir(tmp_path, t))
                                                        for t in B.ADAPTERS])
    assert all(s["passes"] for s in rec["structural"])
    assert F.PL.smoke_serve(tmp_path, True)["parity_passes"]
    F.PL.hf_letters(tmp_path, [(t, F.cand_dir(tmp_path, t)) for t in B.ADAPTERS], True)
    F.PL.stage_read(tmp_path, [(t, F.cand_dir(tmp_path, t)) for t in B.ADAPTERS], True, parts=(),
                    base=False)
    B.stage_a0s(tmp_path, True)
    gate = json.loads((tmp_path / "gate.json").read_text())
    assert gate["p_ctx_n"] in ("this pod", "a later pod (the card says so)")
    for t in B.ADAPTERS:
        B.stage_adapter(tmp_path, t, True)
    gens = sorted(p.name for p in (tmp_path / "mrb" / "gen").glob("*.jsonl"))
    assert gens == [f"{B.MODEL}.a0s.jsonl", f"{B.MODEL}.p_ctx.jsonl", f"{B.MODEL}.p_ctx_n.jsonl"]
    assert (tmp_path / "mrb_session" / "gen.jsonl").exists()
    monkeypatch.setattr(M, "judge_units", lambda items, g: [
        {"unit": f"{u}/response", "sid": u.split("/")[2], "text": t, "k": 0}
        for u, t in list(g.items())[:1]])
    monkeypatch.setitem(P.LLM_KWARGS, "judge", dict(P.LLM_KWARGS["judge"]))
    monkeypatch.setattr(Q, "CAP_N", Q.CAP_N)
    monkeypatch.setattr(Q, "items", Q.items)
    F.configure()
    assert "calls_planned" in F.stage_mrb_judge(tmp_path, True)
