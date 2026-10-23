"""L-beh-2 (scripts/st2lb_driver.py, scripts/analyze_st2lb.py; Amendment 31 item 3 and its
addition): the rule-right pool and pairs, the hooks into T1.4 and T1.5, the branches, the runner's
order, and a dry run of every step on a tiny GPT-OSS."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2lb as A  # noqa: E402
import st2lb_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
DATA = ST / "data"


def test_pool_is_outside_both_read_sets_and_reproducible():
    # most probable failure (Amendment 31's addition): a training item is also a specificity item
    # or sits in the panel guard's half, so the gate reads trained items
    import build_t15_sets as B

    pool = json.loads((DATA / "st_lbeh2_pool_v1.json").read_text())
    spec = set(json.loads((DATA / "t15_specificity_v1.json").read_text())["ids"])
    half = set(json.loads((DATA / "t15_panel_half_v1.json").read_text())["ids"])
    assert len(pool["ids"]) == 130 and not set(pool["ids"]) & (spec | half)
    assert B.lbeh2_pool()["ids"] == pool["ids"]


def _r(sid, k, ns):
    return {"scenario_id": sid, "rollout": k, "norm_status": ns, "prompt_ids": [1, 2],
            "gen_ids": [10 + k]}


def test_rule_right_pairs_choose_the_rule():
    # most probable failure: the rule-right pairs are built with H1's orientation (violating
    # chosen), teaching rule-breaking on rule-right items, or defer rollouts enter a pair
    rows = [_r("F1-A-00", k, ns) for k, ns in enumerate(
        ["consistent", "violating", "consistent", "violating", "consistent", "violating",
         "neutral"])] + [_r("F2-A-01", 0, "consistent")]
    pr = D.rule_right_pairs(rows)
    assert len(pr) == D.PAIRS_PER_ITEM == 2  # capped per item; F2-A-01 has no violating rollout
    assert all(p["chosen"] in ([10], [12]) and p["rejected"] in ([11], [13]) for p in pr)
    assert all(p["kind"] == "rule_right" for p in pr)


def test_cut_to_is_seeded_and_records_a_shortfall():
    # most probable failure: the rule-right set outnumbers the H1 pairs, or a shortfall is hidden
    pr = [{"i": i} for i in range(100)]
    a, info = D.cut_to(pr, 60)
    assert a == D.cut_to(pr, 60)[0] and len(a) == 60 and info["shortfall"] == 0
    b, info = D.cut_to(pr[:40], 60)
    assert len(b) == 40 and info["shortfall"] == 20


def test_hooks_into_t14_and_t15():
    # most probable failure: beh2 trains under L-beh's seed (identical noise) or the panel guard
    # skips it (Amendment 31 requires the guard)
    import t14_driver as T14
    import t15_driver as T15

    assert T14.SEEDS["beh2"] == 105 and len(set(T14.SEEDS.values())) == len(T14.SEEDS)
    rec = {"phases": {"beh_cap": {"comparator": "beh"}}, "panel_arms_extra": ["beh2"]}
    assert T15.panel_arms(rec) == {"beh", "anchor", "beh2"}
    assert T15.panel_arms({"phases": {}}) == {"beh", "anchor"}  # T1.5's own behaviour kept


def test_branches():
    # most probable failure: a discriminating specificity read with ΔP met is called a pass, or a
    # failed guard is ignored
    def x(cat, cls, cap=True, guard=False):
        return {"category_record": cat, "specificity": {"class": cls},
                "capability": {"passes": cap}, "guard": {"fails": guard}}
    met = "H1 criterion met"
    assert A.branch(x(met, "specific")).startswith("passes ΔP and specificity: L-beh's")
    assert "guard fails" in A.branch(x(met, "specific", guard=True))
    assert A.branch(x(met, "non-specific")).startswith("passes ΔP, fails specificity")
    assert "flagged" in A.branch(x(met, "discriminating"))
    assert A.branch(x("null", "discriminating")).startswith("fails ΔP, specificity holds")
    assert A.branch(x("null", "non-specific")).startswith("fails both")


@pytest.mark.skipif(not (ST / "outputs/t15/gpt_oss_20b/gen/h1_beh.jsonl").exists(),
                    reason="T1.5 rows not local")
def test_read_reproduces_l_beh_on_its_own_rows(tmp_path):
    # most probable failure: the read diverges from T1.5's estimators (L-beh's rows, renamed to
    # beh2, must give L-beh's committed numbers and its branch)
    t = ST / "outputs" / "t15" / "gpt_oss_20b"
    for d in ("parity", "vllm", "gen", "cap", "spec", "panel"):
        shutil.copytree(t / d, tmp_path / d)
    for a_, b_ in (("parity/hf_beh.npy", "parity/hf_beh2.npy"),
                   ("parity/hf_beh_merged.npy", "parity/hf_beh2_merged.npy"),
                   ("vllm/beh.jsonl", "vllm/beh2.jsonl"), ("gen/h1_beh.jsonl", "gen/h1_beh2.jsonl"),
                   ("cap/beh.jsonl", "cap/beh2.jsonl")):
        shutil.copy(tmp_path / a_, tmp_path / b_)
    shutil.copytree(tmp_path / "spec" / "beh", tmp_path / "spec" / "beh2")
    shutil.copytree(tmp_path / "panel" / "beh", tmp_path / "panel" / "beh2")
    rec = json.loads((t / "t15_record.json").read_text())
    for k in ("hf_beh", "vllm_beh"):
        rec["phases"][k.replace("beh", "beh2")] = rec["phases"][k]
    (tmp_path / "t15_record.json").write_text(json.dumps(rec))
    r = A.read(tmp_path, ST / "outputs" / "t13" / "gpt_oss_20b" / "panel")["arm"]
    lb = json.loads((DATA / "t15_read_v1.json").read_text())["arms"]["beh"]
    assert r["dP_record"]["mean"] == pytest.approx(lb["dP_record"]["mean"])
    assert r["specificity"]["dP_violating"]["mean"] == pytest.approx(
        lb["specificity_dP_violating"]["mean"])
    assert r["branch"].startswith("passes ΔP, fails specificity")


def test_remote_runs_the_three_steps_in_order():
    # most probable failure: training starts before the rule-right rollouts exist, the gate runs
    # in the main env (T1.5's HF stack is the vLLM venv's), or the stop is too short to train
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    blk = sh.split('if [ "$STEPS" = "st2lb" ]; then\n    DEADLINE')[1][:900]
    assert blk.index("--step rr_gen") < blk.index("--step train") < blk.index("--step gate")
    assert "$VPY $LB --step train" in blk and "$VPY $LB --step gate" in blk
    assert re.search(r'STEPS" = "st2lb" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-5\.5\}"', sh)
    for line in sh.splitlines():
        if "Amendment 10" in line and '!= "st2b"' in line or "re-priced locally" in line:
            assert '!= "st2lb"' in line, line
    sync = (ST / "runpod" / "sync_st2lb.txt").read_text().split()
    assert "outputs/t11/gpt_oss_20b/gen/h1_rollouts_read.jsonl" in sync


@pytest.mark.slow
def test_dry_run_every_step(tmp_path):
    # most probable failure: a step crashes on the pod (the rule-right pairs reach the trainer,
    # T1.5's phases attach beh2 and run the guard on it)
    o = ST / "outputs"
    if not (o / "t14_inputs/gpt_oss_20b/t14_inputs.npz").exists():
        pytest.skip("T1.4 inputs not built locally")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2lb_driver.py"), "--out", str(tmp_path),
            "--t14-inputs", str(o / "t14_inputs/gpt_oss_20b"), "--t11", str(o / "t11/gpt_oss_20b"),
            "--dry-run", "--date-pin", "2026-10-23"]
    for st in ("rr_gen", "train", "gate"):
        r = subprocess.run(base + ["--step", st], capture_output=True, text=True, cwd=ROOT,
                           env=env)
        assert r.returncode == 0, (st, r.stdout[-2000:] + r.stderr[-2000:])
    d = tmp_path / "gpt_oss_20b"
    rec = json.loads((d / "st2lb_record.json").read_text())["steps"]
    assert rec["train"]["rule_right"]["used"] > 0 and rec["gate"]["status"] == "ok"
    t15 = json.loads((d / "t15_record.json").read_text())
    assert t15["panel_arms_extra"] == ["beh2"] and t15["phases"]["hf_beh2"]["panel_cells"]
    t14 = json.loads((d / "t14_record.json").read_text())["phases"]["beh2"]
    assert t14["objective"] == "beh" and (d / "adapters" / "beh2" / "adapter_config.json").exists()
