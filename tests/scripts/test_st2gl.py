"""The GPT-OSS ladder (ST_PLAN Amendment 50 and its additions; scripts/st2gl_driver.py,
scripts/analyze_st2gl.py): the doses, the on-pod 0.3x rule, the gate and branch, the ten-draw
specificity, the runner and a dry run."""

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
import analyze_st2gl as A  # noqa: E402
import st2_driver as S2  # noqa: E402
import st2gl_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"


def test_doses_are_the_registered_multiples():
    # most probable failure: the rungs scale from another norm than Cell 1's 0.2x (267.3)
    c = {"median_norm": S2.ALPHA / 0.2}
    assert D.alpha(c, "04") == pytest.approx(534.6, abs=0.1)
    assert D.alpha(c, "03") == pytest.approx(401.0, abs=0.1)
    assert D.FIRST[-1] == "arm_care_04" and D.SECOND[-1] == "arm_care_03"


def test_03_runs_only_after_an_incoherent_04():
    # most probable failure (second addition, item 1): 0.3x runs although 0.4x was coherent
    assert D.then("arm_care_04", {"coherent": True}, False) == []
    assert D.then("arm_care_04", {"coherent": False}, False) == D.SECOND
    assert D.then("spec_care_04", {"coherent": False}, False) == []
    assert D.then("arm_care_04", {"coherent": False}, True) == []


def test_branch():
    # most probable failure: a coherent 0.4x that fails the gate is read as anything but R_lever,
    # or a passing dose with specificity above the bar reads as R_dose
    ok = {"coherent": True, "passes": True}
    assert A.branch("04", {"coherent": True, "passes": False}, None).startswith("R_lever")
    assert A.branch("04", ok, {"vs_fill10": {"mean": 0.02}}).startswith("R_dose")
    assert A.branch("04", ok, {"vs_fill10": {"mean": 0.05}}).startswith("R_model")
    assert A.branch("04", {"coherent": False, "passes": False}, None) == "04: incoherent"


def _rows(arm, p_by_sid, n=8):
    return [{"scenario_id": s, "rollout_key": f"st2c1|{s}|{k}", "rollout": k, "letter": "A",
             "norm_status": "violating" if k % round(1 / p) == 0 else "consistent",
             "status": "completed",
             "option_id": "o1", "arm": arm} for s, p in p_by_sid.items() for k in range(n)]


def test_gate_pairs_with_cell1_on_the_same_keys():
    # most probable failure: the gate reads Cell 1's 16 rollouts against the ladder's 8
    sids = [f"S{i:02d}" for i in range(20)]
    care = _rows("care", {s: 0.5 for s in sids})
    none = _rows("none", {s: 0.25 for s in sids}, n=16)
    fill = _rows("fill", {s: 0.25 for s in sids}, n=16)
    g = A.gate(care, none, fill, {"passes": True})
    assert g["n_rollouts"] == 160 and g["passes"]
    assert g["dP_vs_none"]["mean"] == pytest.approx(0.25)
    assert g["dP_vs_fill"]["mean"] == pytest.approx(0.25)


def test_runner_and_sync():
    # most probable failure: the ladder runs without its fill draws, or the GPT-OSS refusal-
    # retention pass rides this pod
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2gl" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-gpt_oss_20b'
                     r'\}"; OUT_SUB=st2gl\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-3\.0\}"', sh)
    a10 = [x for x in sh.splitlines() if x.rstrip().endswith("# Amendment 10")]
    assert a10 and all('[ "$STEPS" != "st2gl" ]' in x for x in a10)
    assert "test_st2gl.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_st2gl.txt").read_text().split()
    assert "outputs/st2gl_inputs/gpt_oss_20b/fill10.npz" in sync
    assert "outputs/st2a/gpt_oss_20b/gen/c1_none.jsonl" in sync


@pytest.mark.slow
def test_dry_run(tmp_path):
    # most probable failure: a phase crashes on a tiny GPT-OSS, or the ladder's rows do not pair
    # with Cell 1's none
    inp = ST / "outputs" / "st2_inputs" / "gpt_oss_20b"
    fill = ST / "outputs" / "st2gl_inputs" / "gpt_oss_20b"
    podc = ST / "outputs" / "st2c_inputs" / "gpt_oss_20b"
    if not ((inp / "st2_inputs.npz").exists() and (fill / "fill10.npz").exists()):
        pytest.skip("the ladder's inputs are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2_driver.py"), "--out",
                        str(tmp_path / "st2"), "--dry-run", "--inputs", str(inp), "--date-pin",
                        "2026-10-26", "--phases", "c1_none"], capture_output=True, text=True,
                       cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2gl_driver.py"), "--out",
                        str(tmp_path / "gl"), "--dry-run", "--inputs", str(inp), "--podc-inputs",
                        str(podc), "--c1", str(tmp_path / "st2" / "gpt_oss_20b"), "--fill",
                        str(fill)], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads((tmp_path / "gl" / "gpt_oss_20b" / "st2gl_record.json").read_text())
    assert set(D.FIRST) <= set(rec["phases"])
    rows = [json.loads(x) for x in (tmp_path / "gl" / "gpt_oss_20b" / "spec" / "fill10_04" /
                                    "dl_chat_neutral.jsonl").read_text().splitlines() if x]
    assert len({r["fill_draw"] for r in rows}) == 10
