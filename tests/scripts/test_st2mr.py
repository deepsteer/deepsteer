"""The care_harm replicate at 0.3x on Muse (scripts/st2mr_driver.py, scripts/analyze_st2mr.py;
Amendment 32): fresh seeds and fill draws, the gate and specificity of record, the branches, the
runner's order, and a dry run on a tiny Cell 4."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2mr as A  # noqa: E402
import st2mr_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
INP = ST / "outputs" / "st2mr_inputs" / "muse_glimmer_30b"
C4 = ST / "outputs" / "st2m" / "muse_glimmer_30b"
LAD = ST / "outputs" / "st2ml" / "muse_glimmer_30b"


def test_design_constants():
    # most probable failure (Amendment 32): the replicate reuses Cell 4's seeds or fill draws (the
    # spec read would reproduce the ladder's), or runs at 8 rollouts
    assert D.R == 16 and D.STAGE == "st2mr" and D.FILL_SEED == 1 and D.RUNG == "03"
    first_arm = min(i for i, p in enumerate(D.PHASES) if p.startswith("arm_"))
    assert all(i < first_arm for i, p in enumerate(D.PHASES) if p.startswith("spec_"))
    assert D.PHASES[first_arm] == "arm_none"


@pytest.mark.skipif(not (INP / "fill_seed1.npz").exists() or not (C4 / "derive.npz").exists(),
                    reason="the replicate inputs are not built locally")
def test_fresh_draws_differ_from_cell4s():
    # most probable failure: the fresh draws equal Cell 4's (seed 0), so the replicate's spec read
    # repeats the ladder's
    fu, old = np.load(INP / "fill_seed1.npz")["fill_unit"], np.load(C4 / "derive.npz")["fill_unit"]
    assert fu.shape == old.shape and np.allclose(np.linalg.norm(fu, axis=1), 1, atol=1e-5)
    assert np.abs(fu @ old.T).max() < 0.5


def test_branches():
    # most probable failure: a gate that fails against the dose-matched fill is read as a target
    def g(ok):
        return {"passes": ok}
    assert A.branch(g(True), {"class": "specific"}).startswith("target")
    assert A.branch(g(True), {"class": "discriminating"}).startswith("discriminating")
    assert A.branch(g(True), {"class": "non-specific"}).startswith("non-specific")
    assert A.branch(g(False), {"class": "specific"}).startswith("no lever")


NEEDED = [C4 / "gen" / "arm_none.jsonl", C4 / "gen" / "arm_fill.jsonl",
          C4 / "spec" / "none" / "dl_chat_neutral.jsonl", LAD / "gen" / "arm_care_03.jsonl",
          LAD / "spec" / "care_03" / "dl_chat_neutral.jsonl",
          LAD / "spec" / "fill_03" / "dl_chat_neutral.jsonl"]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="Cell 4 / ladder rows not local")
def test_read_on_stand_in_rows(tmp_path):
    # most probable failure: the gate reads the margin against the wrong fill, or the specificity
    # of record is taken against none instead of the dose-matched fill
    (tmp_path / "gen").mkdir()
    shutil.copy(C4 / "gen" / "arm_none.jsonl", tmp_path / "gen" / "arm_none.jsonl")
    shutil.copy(C4 / "gen" / "arm_fill.jsonl", tmp_path / "gen" / "arm_fill_03.jsonl")
    shutil.copy(LAD / "gen" / "arm_care_03.jsonl", tmp_path / "gen" / "arm_care_03.jsonl")
    for arm, src in (("none", C4 / "spec" / "none"), ("care_03", LAD / "spec" / "care_03"),
                     ("fill_03", LAD / "spec" / "fill_03")):
        (tmp_path / "spec" / arm).mkdir(parents=True)
        shutil.copy(src / "dl_chat_neutral.jsonl", tmp_path / "spec" / arm)
    (tmp_path / "st2mr_record.json").write_text(json.dumps({"phases": {}}))
    ids = json.loads((ST / "data" / "t15_specificity_v1.json").read_text())["ids"]
    g, s = A.gate(tmp_path), A.specificity(tmp_path, ids)
    assert g["passes"] and g["dP_vs_fill"]["mean"] == pytest.approx(0.0889, abs=1e-3)
    assert s["care_minus_fill"]["mean"] == pytest.approx(-0.0013, abs=1e-3)
    assert s["class"] == "specific" and A.branch(g, s).startswith("target")
    assert A.parity(tmp_path, C4, LAD)["spec_care_03_vs_ladder_max_abs_dlogp"] == 0.0


def test_remote_runs_the_replicate():
    # most probable failure: the runner omits the fresh draws from the pod's inputs, or the stop
    # is shorter than three arms of 720
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2mr" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-4\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "st2mr" ]; then\n    DEADLINE')[1][:500]
    assert '--inputs "$ST2MR_INPUTS"' in blk and '--cell4 "$CELL4_DIR"' in blk
    sync = (ST / "runpod" / "sync_st2mr.txt").read_text().split()
    assert "outputs/st2mr_inputs/muse_glimmer_30b/fill_seed1.npz" in sync
    assert all(not x.endswith("/") for x in sync)


@pytest.mark.slow
def test_dry_run_replicate_on_a_tiny_cell4(tmp_path):
    # most probable failure: the replicate's arms do not share fresh seeds, or a phase crashes on
    # the pod's tiny Cell 4
    qt = ST / "outputs" / "st2b" / "gpt_oss_20b" / "gen" / "c3_qwen_topup.jsonl"
    qs = ST / "outputs" / "t10" / "qwen38_27b" / "screen" / "h1_screen.jsonl"
    if not (qt.exists() and qs.exists() and (INP / "fill_seed1.npz").exists()):
        pytest.skip("inputs not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    m = [sys.executable, str(SCRIPTS / "st2m_driver.py"), "--out", str(tmp_path / "c4"),
         "--dry-run", "--date-pin", "2026-10-23"]
    for extra in (["--phases", "v1"], ["--qwen-topup", str(qt), "--qwen-screen", str(qs)]):
        r = subprocess.run(m + extra, capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2mr_driver.py"), "--out",
                        str(tmp_path / "rep"), "--cell4", str(tmp_path / "c4" / "muse_glimmer_30b"),
                        "--inputs", str(INP), "--dry-run"],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = tmp_path / "rep" / "muse_glimmer_30b"
    rec = json.loads((d / "st2mr_record.json").read_text())["phases"]
    assert all(rec[p]["finished"] for p in D.PHASES)
    keys = {a: [json.loads(x)["rollout_key"] for x in (d / "gen" / f"arm_{a}.jsonl").open()]
            for a in ("none", "fill_03", "care_03")}
    assert keys["none"] == keys["fill_03"] == keys["care_03"]
    assert all(k.startswith("st2mr|") for k in keys["none"])
