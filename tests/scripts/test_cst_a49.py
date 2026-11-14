"""CST Amendment 49's pod: papers/cst_constitutional_steering/scripts/cst_a49_pod.py, the a49 stages
of cst_teacher_pod.py, and a49_read.py."""

from __future__ import annotations

import hashlib
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

import cst_a49_pod as A  # noqa: E402
import cst_teacher_pod as TP  # noqa: E402

CST = ROOT / "papers/cst_constitutional_steering"
RUNPOD = CST / "runpod"


def _gen_raw(sids):
    gen, raw = [], []
    for s in sids:
        for k in (0, 1):
            text = f"answer for {s} {k}"
            jid = f"qwen38_27b/none/{s}/k{k}"
            sha = hashlib.sha256(text.encode()).hexdigest()
            gen.append({"job_id": jid, "response_sha256": sha})
            raw.append({"job_id": jid, "trace": "t", "response": text})
    return gen, raw


def test_rejudge_renames_the_arm_so_the_judge_draws_fresh_seeds():
    # most probable failure (Amendment 49 item 2a): the re-judge keeps the saved job ids, so MRB's
    # judge reuses the saved seeds and the check measures nondeterminism, not a judge session
    import mrb_g1_pod as P

    gen, raw = _gen_raw(["mrb-001", "mrb-002", "mrb-003"])
    g, r = A.rejudge_rows(2, gen, raw, ["mrb-002", "mrb-003", "mrb-001"])
    assert {x["job_id"].split("/")[2] for x in g} == {"mrb-002", "mrb-003"} and len(g) == 4
    assert all(x["job_id"].startswith("qwen38_27b/rejudge/") for x in g + r)
    old = "qwen38_27b/none/mrb-002/k0/response/c1"
    assert P.seed_of("p2-judge", old) != P.seed_of("p2-judge", old.replace("/none/", "/rejudge/"))
    raw[2]["response"] = "changed"  # a synced response that differs from the saved row
    try:
        A.rejudge_rows(2, gen, raw, ["mrb-002", "mrb-003", "mrb-001"])
    except ValueError:
        return
    raise AssertionError("a changed response passed the sha256 check")


def test_the_rejudge_order_is_the_seeded_order_of_record():
    # most probable failure: the committed order is not the one its seed produces, so the first N
    # scenarios are a chosen set rather than a seeded draw
    rec = json.loads(A.ORDER.read_text())
    assert rec["seed"] == A.ORDER_SEED and rec["n"] == 500
    assert A.build_order(rec["order"])["order"] == rec["order"]


def test_chunks_hold_whole_states_so_a_cut_keeps_arms_paired():
    # most probable failure (Amendment 49): a deadline between chunks leaves a state with one arm
    # decided and the other not, so a cut read pairs P-ctx against a base on other states
    jobs = [{"sid": f"g|0|{s}", "arm": a} for s in range(7) for a in ("p_ctx", "base", "base")]
    parts = TP.chunks(jobs, size=4)
    assert sum(len(p) for p in parts) == len(jobs)
    for p in parts:
        sids = {j["sid"] for j in p}
        assert all(sum(j["sid"] == s for j in p) == 3 for s in sids)


def test_citing_cell_is_the_seeded_first_300_of_the_nested_set():
    # most probable failure (Amendment 49 item 5.2): the citing cell takes the first 300 in sorted
    # order (the first games only) instead of the seeded order
    ids = TP.citing_states()
    nested = json.loads(TP.STATES.read_text())["nested"]
    assert len(ids) == 300 and set(ids) <= set(nested) and ids != sorted(nested)[:300]
    assert len({s.split("|")[0] for s in ids}) > 20


def test_the_citing_flag_is_cst_a32s_pattern():
    # most probable failure: the flag uses another pattern than CST-A32's, so its rates are not
    # comparable with the 96.1% and 0.02% of record
    assert TP.cites("as my Constitution says") and not TP.cites("constitutional law")


def test_the_read_switches_base_only_when_the_session_shift_excludes_0(tmp_path, monkeypatch):
    # most probable failure (Amendment 49 item 2d): P-ctx is read against the base of record when
    # this pod's base differs from it, or a constitution arm that equals the base reads as dilution
    # ruled out
    import a49_read as R
    import teacher_read as TR

    games = [f"g{i}" for i in range(6)]
    sids = [f"{g}|0|{s}" for g in games for s in range(3)]
    lab = [{"game": g, "episode": 0, "step": s, "violating": [1, 0],
            "labels": [{c: 1 for c in TR.CATS}, {c: 0 for c in TR.CATS}], "saved_violating": 1,
            "saved_action": 0, "eligible": True} for g in games for s in range(3)]
    lf = tmp_path / "labels.jsonl"
    lf.write_text("".join(json.dumps(x) + "\n" for x in lab))
    monkeypatch.setattr(TR, "LABELS", lf)
    st = tmp_path / "states.json"
    st.write_text(json.dumps({"nested": sids, "wide": sids}))
    monkeypatch.setattr(R, "STATES", st)
    teacher = tmp_path / "teacher_v1"
    (teacher / "teacher").mkdir(parents=True)
    monkeypatch.setattr(R, "TEACHER", teacher)

    def rows(arm, action, k=2):
        return [{"arm": arm, "sid": s, "k": i, "action": action, "fallback": False,
                 "cites": False} for s in sids for i in range(k)]

    (teacher / "teacher" / "base_good_behavior.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("base", 0) + rows("good_behavior", 1)))
    out = tmp_path / "a49"
    (out / "teacher").mkdir(parents=True)
    (out / "teacher" / "a49_pctx.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("p_ctx", 1) + rows("base", 1)))
    (out / "teacher" / "a49_cgood.jsonl").write_text("".join(
        json.dumps(x) + "\n" for x in rows("constitution_good", 0) + rows("constitution", 0, 1)))
    res = R.mach_read(out)
    assert res["2d"]["read_of_record"] == "against this pod's base on the 888"
    assert res["2d"]["branch"] == "no difference detectable at its MDE"
    assert res["2d"]["beside_p_ctx_minus_base_of_record"]["below_0"]
    assert res["2c"]["reading"].startswith("GOOD_BEHAVIOR's effect does not survive")


def test_goal_ratio_reads_the_adapter_against_good_behavior_on_shared_states():
    # most probable failure (Amendment 50 item 3): the ratio is taken on unshared states or with
    # the wrong denominator, so an adapter that matches GOOD_BEHAVIOR misses the 0.5 bound
    import a49_read as R

    sids = [f"g{g}|0|{s}" for g in range(8) for s in range(4)]
    base = {s: 0.4 for s in sids}
    gb = {s: 0.3 for s in sids}
    same = R.goal_ratio({s: 0.3 for s in sids}, base, gb, base)
    assert abs(same["ratio"] - 1.0) < 1e-9 and same["meets"] and same["games"] == 8
    none = R.goal_ratio({s: 0.4 for s in sids[:16]}, base, gb, base)
    assert none["n"] == 16 and abs(none["ratio"]) < 1e-9 and not none["meets"]


def test_card_bails_until_the_rejudge_size_is_ruled_and_runner_keeps_the_order():
    # most probable failure: the card launches before the author rules 2a's size, or the runner
    # runs the stages out of the registered order (a, b, c, d), so the cut takes the wrong stage
    card = (RUNPOD / "launch_cst_a49.sh").read_text()
    assert "REJUDGE_SCENARIOS=500" in card and "is not ruled yet" in card  # Amendment 50 item 1
    run = (RUNPOD / "remote_cst_a49.sh").read_text()
    order = [run.index(s) for s in ("--stage rejudge_prep", "--stage a0n", "fits a49_cgood",
                                     "fits a49_pctx", "--stage a49_pctx")]
    assert order == sorted(order) and "CLOUD_TYPES=\"COMMUNITY,SECURE\"" in card


@pytest.mark.slow
def test_dry_session_runs_every_a49_stage_in_order(tmp_path, monkeypatch):
    # most probable failure (Amendment 28 item 3): a later stage breaks on the pod after hours;
    # every Amendment 49 stage runs on the tiny model in the runner's order and the read reads it
    import a49_read as R
    import pilot_pod_lib as PL

    from deepsteer.datasets import machiavelli as mch

    if not (TP.RUN / "decisions.jsonl").exists() or not (
            mch.default_cache_dir() / "src" / "game_data").exists():
        pytest.skip("the finalist run's decisions or the MACHIAVELLI bundle are not here")
    order = json.loads(A.ORDER.read_text())["order"]
    gen, raw = _gen_raw(order[:2])
    for name, rows in (("gen", gen), ("raw", raw)):
        f = tmp_path / f"base.{name}.jsonl"
        f.write_text("".join(json.dumps(x) + "\n" for x in rows))
        monkeypatch.setattr(A, f"BASE_{name.upper()}", f)
    _, wb, _ = PL.load_model(True)
    adir = tmp_path / "adapters" / "p_ctx" / "1"
    wb.model.save_pretrained(str(adir))
    (tmp_path / "train_p_ctx.json").write_text(json.dumps({"trained": True,
                                                           "adapters": {"1": str(adir)}}))
    assert A.stage_rejudge_prep(tmp_path, True)["rollouts"] == 4
    A.F.stage_mrb_judge(tmp_path, True)
    assert A.stage_a0n(tmp_path, True)["n_jobs"] == 4
    A.F.stage_mrb_judge(tmp_path, True)
    cg = TP.stage_cells(tmp_path, "a49_cgood", [("constitution_good", "nested", 2),
                                                ("constitution", "citing", 1)], True)
    pc = TP.stage_cells(tmp_path, "a49_pctx", [("p_ctx", "wide", 2), ("base", "nested", 2)],
                        True, tag="p_ctx")
    assert cg["dropped_by_the_replay_gate"] == 0 and pc["decisions"] == pc["decisions_planned"]
    rows = [json.loads(x) for x in open(tmp_path / "teacher" / "a49_pctx.jsonl")]
    assert {r["arm"] for r in rows} <= {"p_ctx", "base"} and all("cites" in r for r in rows)
    again = TP.stage_cells(tmp_path, "a49_pctx", [("p_ctx", "wide", 2), ("base", "nested", 2)],
                           True, tag="p_ctx")
    assert again["decisions_run"] == 0  # resumes without regenerating
    if (R.R2.MRB_P2 / "judge" / "shard0.jsonl").exists():  # local only: the read is not run on
        res = R.read(tmp_path)  # the pod, which does not sync MRB's saved judge rows
        assert res["2a"]["rollouts"] == 4 and "2d" in res
    else:
        assert "2d" in R.mach_read(tmp_path)
