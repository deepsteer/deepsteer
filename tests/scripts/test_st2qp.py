"""Pod 2 on Qwen3.8-27B (ST_PLAN Amendment 57 and its additions; scripts/st2qp_driver.py,
scripts/analyze_st2qp.py): the teacher's scenarios, the step rule and checkpoints, the selection
rule, the selection set, the forced read, the branch, the runner and a dry run."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2qp as A  # noqa: E402
import st2qp_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
POD1 = ST / "outputs" / "st2q" / "qwen38_27b"


def test_teacher_scenarios_are_the_registered_35():
    # most probable failure: the teacher reads held scenarios, or scenarios with only one of the
    # two outcomes, so the pair ceiling differs from the one registered
    if not (POD1 / "gen" / "h1_rollouts.jsonl").exists():
        pytest.skip("pod 1's rows are not local")
    ids = D.teacher_scenarios({"pod1": POD1})
    assert len(ids) == D.N_TEACHER == 35
    split = json.loads((ST / "split_v1.json").read_text())
    assert set(ids) <= set(split["H1_train"])


def test_steps_and_checkpoints():
    # most probable failure (the third addition, item 1): the steps are scaled to 80 pairs, so the
    # adapters see half T1.6's epochs; or a third rounds to 0 and an untrained adapter is saved
    assert D.steps_for(40) == 243  # T1.6: 243 steps over 40 pairs, about 24 epochs at 4 micro
    assert D.steps_for(90) == 547 and 547 * 4 / 90 == pytest.approx(24.3, abs=0.1)
    assert D.checkpoints(547) == [182, 365, 547]
    assert D.checkpoints(2) == [1, 2]


def test_bail_keeps_the_pod_inside_the_envelope():
    # most probable failure: the bail is a multiple of the smoke rate, which at 24 epochs lets
    # two adapters' training carry the pod past the 125 envelope
    assert D.SPENT_BEFORE + D.NON_TRAIN_H + 2 * D.bail_hours() == pytest.approx(D.ENVELOPE)
    assert D.bail_hours() > 547 * 4 * 11.85 / 3600  # the smoke's rate fits at the expected pairs


def test_selection_rule():
    # most probable failure (the third addition, item 2): the arms get different steps of record
    # (the teacher stops being the only difference), the earliest checkpoint is scored, a float
    # edge at the 0.02 bar flips a tie, or per-arm selection beside needs scores the joint rule
    # skipped (a short-circuit at the first failing arm)
    marks = [182, 365, 547]
    seen = []

    def correct(table):
        def f(arm, s):
            seen.append((arm, s))
            return table[(arm, s)]
        return f

    t = {("care", 547): 82, ("base", 547): 85, ("care", 365): 83, ("base", 365): 84}
    pick = D.choose(100, 85, correct(t), marks, ("care", "base"))
    assert pick["of_record"] == {"care": 365, "base": 365}  # care is 3 below at the end
    assert pick["per_arm_beside"] == {"care": 365, "base": 547}
    assert all(s != 182 for _, s in seen)
    assert ("base", 547) in seen  # scored though care failed first
    t = {k: 70 for k in t}
    assert D.choose(100, 85, correct(t), marks, ("care", "base"))["of_record"] == \
        {"care": 182, "base": 182}
    t = {("care", 547): 83, ("base", 547): 83}  # exactly 0.02 below: within the bar
    assert D.choose(100, 85, correct(t), marks, ("care", "base"))["of_record"] == \
        {"care": 547, "base": 547}


def test_selection_set_is_disjoint_from_the_gate():
    # most probable failure: a selection item is one of the gate's 200, so selection leaks into
    # the capability read it must never decide
    sel = json.loads((ST / "data" / "st2qp_selection_v1.json").read_text())["items"]
    gate = json.loads((ST / "data" / "t15_capability_v1.json").read_text())["items"]
    key = lambda i: (i["dataset"], i["index"])  # noqa: E731
    assert len(sel) == 100 and not {key(i) for i in sel} & {key(i) for i in gate}
    assert sum(i["dataset"] == "openai/gsm8k" for i in sel) == 50


def test_forced_read_and_branch():
    # most probable failure: D_f is read against none's twins instead of the arm's own
    # originals, or a protective none (rule broken on the originals) reads as a transfer
    Fn = {"a": 0.1, "a-T": 0.5, "b": 0.2, "b-T": 0.6}
    F = {"a": 0.1, "a-T": 0.9, "b": 0.2, "b-T": 0.8}
    fr = A.forced_read(F, Fn)
    assert fr["D_f"]["mean"] == pytest.approx(0.7)
    assert fr["forced_H_star"]["mean"] == pytest.approx(0.3)
    assert fr["original_shift"]["mean"] == pytest.approx(0.0)
    nf = A.none_forced(Fn)
    assert nf["informative"]
    assert A.branch(0.0, fr, True).startswith("transfers")
    assert A.branch(0.05, fr, True).startswith("non-specific")
    assert A.branch(0.0, fr, False).startswith("uninformative")
    assert not A.none_forced({"a": 0.9, "a-T": 0.9, "b": 0.8, "b-T": 0.9})["informative"]
    # the third addition, item 3: an adapter equal to none (forced H* 0) never reads as a
    # transfer, though none's own D_f clears the bar
    assert A.branch(0.0, A.forced_read(Fn, Fn), True).startswith("unregistered")
    low = {"a": 0.1, "a-T": 0.2, "b": 0.2, "b-T": 0.3}
    assert A.branch(0.0, A.forced_read(low, Fn), True).startswith("unregistered")


def test_configure_points_t16_at_qwen():
    # most probable failure: t16_driver keeps Muse's key, pin or site, or its engine runs without
    # the forced cells
    import t16_driver as T16

    D.configure()
    assert (T16.KEY, T16.PIN, T16.BLOCK, T16.STAGE) == ("qwen38_27b", "2026-10-27", 32, "st2qp")
    assert T16.gen_vllm is D.gen_vllm and T16.pairs_for is D.pairs_for


def test_runner_and_sync():
    # most probable failure: selection runs after the merge (the gate reads an unselected
    # adapter), a step runs under the wrong interpreter, or pod 1's direction is not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2qp" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-qwen38_27b'
                     r'\}"; OUT_SUB=st2qp\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-28\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "st2qp" ]; then\n    DEADLINE=')[1].split("continue")[0]
    order = [blk.index(f"--step {s}") for s in ("rr_gen", "base_vllm", "base_hf", "teacher",
                                                "train", "select", "hf_arm", "vllm_arm",
                                                "hfgen")]
    assert order == sorted(order)
    assert "$VPY $QP --step teacher" in blk and "$PY $QP --step vllm_arm" in blk
    assert "$VPY $QP --step select" in blk
    assert "test_st2qp.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_st2qp.txt").read_text().split()
    assert {"outputs/st2q/qwen38_27b/directions.npz",
            "outputs/st2q/qwen38_27b/st2q_record.json",
            "outputs/st2q/qwen38_27b/gen/h1_rollouts.jsonl"} <= set(sync)


@pytest.mark.slow
def test_dry_run(tmp_path):
    # most probable failure: a step breaks on a tiny Qwen (the steered teacher at the site, the
    # checkpoints at thirds, attaching and unloading two adapters in one selection process, the
    # forced cells in the engine and in the fallback)
    if not (POD1 / "gen" / "arm_care.jsonl").exists():
        pytest.skip("pod 1's rows are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2qp_driver.py"), "--out", str(tmp_path), "--pod1",
            str(POD1), "--dry-run"]
    steps = [["rr_gen"], ["base_vllm"], ["base_hf"], ["teacher"], ["train", "--arm", "care"],
             ["train", "--arm", "base"], ["select"], ["hf_arm", "--arm", "care"],
             ["vllm_arm", "--arm", "care"], ["hf_arm", "--arm", "base"],
             ["vllm_arm", "--arm", "base", "--dry-fail-parity"], ["hfgen", "--arm", "care"],
             ["hfgen", "--arm", "base"], ["hfgen", "--arm", "none"]]
    for st in steps:
        r = subprocess.run(base + ["--step"] + st, capture_output=True, text=True, cwd=ROOT,
                           env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    q = tmp_path / D.KEY
    rec = json.loads((q / "st2qp_record.json").read_text())["phases"]
    assert rec["train_care"]["checkpoints"] == [1, 2, 3]
    assert (q / "adapters" / "care" / "adapter_config.json").exists()
    assert rec["select"]["chosen"]["care"] == rec["select"]["chosen"]["base"]
    assert (q / "gen" / "forced_none.jsonl").exists()
    assert (q / "gen" / "forced_base.parity_failed.jsonl").exists()
    assert (q / "gen" / "forced_base.hf.jsonl").exists()
    out = A.read(q)
    assert out["branch_of_record"]


def test_a37_pairs_share_the_rejected_trace_and_the_rule():
    # most probable failure (ST-A37's fixed rule): care's and base's chosen traces are compared
    # across different rejected traces (unpaired), or the overlap is read in the wrong direction
    def row(sid, k, st, ids, nll):
        return {"scenario_id": sid, "rollout": k, "norm_status": st, "gen_ids": ids,
                "trace_nll_unsteered": nll}

    care = [row(f"S{i}", 0, "violating", [1, 2, 3, 4], 1.5) for i in range(30)]
    none = []
    for i in range(30):
        none += [row(f"S{i}", 0, "violating", [7, 8, 9, 4], 1.0),
                 row(f"S{i}", 1, "consistent", [1, 2, 3, 5], 1.2)]
    r = A.a37_family(care, none)
    assert r["n_pairs"] == 30
    assert r["d_nll"]["mean"] == pytest.approx(0.5)  # steered chosen less likely
    assert r["d_overlap"]["mean"] > 0  # steered chosen shares more tokens with the rejected
    both = A.a37({"m": (care, none), "q": (care, none)})
    assert both["branch"].startswith("both")


def test_a36_branch_and_read(tmp_path):
    # most probable failure (Amendment 59 item 4): an arm that failed parity is read anyway, or
    # the bar is applied to base instead of care
    assert A.a36_branch({"mean": 0.03}).startswith("R_item")
    assert A.a36_branch({"mean": 0.031}).startswith("R_reasoning")
    assert A.a36_branch({}).startswith("no read")
    g = tmp_path / "gen"
    g.mkdir()

    def rows(arm, p):
        return [{"scenario_id": f"I{i:02d}", "rollout": k, "norm_status":
                 "violating" if k / 8 < p else "consistent", "status": "completed"}
                for i in range(20) for k in range(8)]
    for arm, p in (("none", 0.125), ("care", 0.125), ("base", 0.5)):
        (g / f"spec_{arm}.jsonl").write_text("\n".join(json.dumps(r) for r in rows(arm, p)))
    rec = {"gen_none": {"base_tv": 0.01, "base_check_ok": True},
           "gen_care": {"parity_ok": True}, "gen_base": {"parity_ok": False}}
    (tmp_path / "st2qg_record.json").write_text(json.dumps({"phases": rec}))
    r = A.a36_read(tmp_path)
    assert r["care"]["G"]["mean"] == pytest.approx(0.0) and r["branch"].startswith("R_item")
    assert "base" not in r  # failed parity: not read


def test_st2qg_runner_and_sync():
    # most probable failure (Amendment 59): the merge runs before none's base check, the merged arm
    # runs under the HF interpreter, or pod 2's adapters or saved letters are not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2qg" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-qwen38_27b'
                     r'\}"; OUT_SUB=st2qg\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-2\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "st2qg" ]; then\n    QG=')[1].split("continue")[0]
    assert blk.index("--step gen --arm none") < blk.index("--step merge")
    assert "$VPY $QG --step merge" in blk and '$PY $QG --step gen --arm "$arm"' in blk
    sync = set((ST / "runpod" / "sync_st2qg.txt").read_text().split())
    pod2 = "outputs/st2qp/qwen38_27b/"
    assert {pod2 + f"adapters/{a}/{f}" for a in ("care", "base")
            for f in ("adapter_config.json", "adapter_model.safetensors", "st_adapter.json")} <= sync
    assert {pod2 + f"parity/hf_{a}.npy" for a in ("none", "care", "base")} <= sync
    assert pod2 + "st2qp_record.json" in sync


@pytest.mark.slow
def test_st2qg_dry_run(tmp_path):
    # most probable failure: the generation read breaks on a tiny Qwen (the panel items through
    # h1_jobs, the merge of a fresh adapter, the parity rows' letters)
    if not (POD1 / "gen" / "h1_rollouts.jsonl").exists():
        pytest.skip("pod 1's rows are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2qg_driver.py"), "--out", str(tmp_path), "--pod1",
            str(POD1), "--pod2", str(tmp_path / "qwen38_27b"), "--dry-run"]
    for st in (["gen", "--arm", "none"], ["merge", "--arm", "care"], ["gen", "--arm", "care"],
               ["merge", "--arm", "base"], ["gen", "--arm", "base"]):
        r = subprocess.run(base + ["--step"] + st, capture_output=True, text=True, cwd=ROOT,
                           env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    q = tmp_path / "qwen38_27b"
    assert all((q / "gen" / f"spec_{a}.jsonl").exists() for a in ("none", "care", "base"))
    assert A.a36_read(q)["branch"]
