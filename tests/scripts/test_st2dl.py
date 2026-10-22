"""ST Phase 2 D-lite (scripts/st2dl_driver.py, scripts/analyze_st2dl.py; Amendment 29): the
random-vector references for pod C's retention and specificity reads, both branches, and a dry run
on a tiny GPT-OSS with the real harmony tokenizer."""

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
import analyze_st2dl as A  # noqa: E402
import st2dl_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
PANEL = ST / "outputs" / "t13" / "gpt_oss_20b" / "panel"
PODC = ST / "outputs" / "st2c" / "gpt_oss_20b"


def test_remote_runs_the_d_lite_in_the_drivers_order():
    # most probable failure: the runner's phase list drifts from the driver's, the D-lite re-runs
    # Amendment 10's selection pass, or its stop is left at pod C's 4.5 h
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'ST2DL_PHASES="([^"]*)"', sh).group(1).split(",") == D.PODDL
    assert D.PODDL[0] == "rr_none"  # the paired none before any fill arm
    for line in sh.splitlines():
        if "Amendment 10" in line and '!= "st2b"' in line or "re-priced locally" in line:
            assert '!= "st2dl"' in line, line
    assert re.search(r'STEPS" = "st2dl" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-2\.0\}"', sh)
    sync = (ST / "runpod" / "sync_st2dl.txt").read_text().split()
    assert {"outputs/st2_inputs/", "outputs/st2c_inputs/",
            "outputs/st2a/gpt_oss_20b/gen/c1_none.jsonl"} <= set(sync)


def _shift(src: Path, dst: Path, dv: float) -> None:
    """T1.3c's rows with the violating option's log-prob raised by ``dv`` (a known tilt)."""
    dst.mkdir(parents=True, exist_ok=True)
    with (dst / "dl_chat_neutral.jsonl").open("w") as f:
        for r in map(json.loads, (src / "dl_chat_neutral.jsonl").open()):
            r["option_logps"] = {k: v + (dv if r["norm_status_by_letter"][k] == "violating" else 0)
                                 for k, v in r["option_logps"].items()}
            f.write(json.dumps(r) + "\n")


@pytest.mark.skipif(not PANEL.exists() or not PODC.exists(), reason="T1.3c / pod C rows not local")
def test_specificity_branches(tmp_path):
    # most probable failure: S2 is entered on a fill read inside the bar (or S1 on one outside),
    # or V_qwen is made a target under S1, where pod C's failure stands
    dl = tmp_path / "dl"
    _shift(PANEL, dl / "spec" / "none", 0.0)
    _shift(PANEL, dl / "spec" / "fill", 0.0)
    _shift(PANEL, dl / "spec" / "fill_half", 0.0)
    s = A.specificity(dl, PODC, PANEL)
    assert s["branch"] == "S1" and s["gpt_oss_branch_closes"] and not s["V_qwen_target"]
    shutil.copy(PODC / "spec" / "V_qwen" / "dl_chat_neutral.jsonl",
                dl / "spec" / "fill_half" / "dl_chat_neutral.jsonl")  # fill tilts like V_qwen
    s = A.specificity(dl, PODC, PANEL)
    assert s["branch"] == "S2" and s["arm_minus_fill"]["V_qwen"]["passes"]
    assert s["V_qwen_target"] and not s["gpt_oss_branch_closes"]
    _shift(PANEL, dl / "spec" / "fill_half", 3.0)  # fill tilts far past V_qwen
    s = A.specificity(dl, PODC, PANEL)
    assert s["branch"] == "S2" and not s["V_qwen_target"] and s["gpt_oss_branch_closes"]


@pytest.mark.skipif(not PODC.exists(), reason="pod C rows not local")
def test_retention_rivals(tmp_path):
    # most probable failure: rival (a) is read when fill does not raise compliance, or (b) when
    # the arms do not rise above fill; the rollout overlap is reported across non-matching engines
    dl = tmp_path / "dl"
    (dl / "rr").mkdir(parents=True)
    shutil.copy(PODC / "rr" / "none.jsonl", dl / "rr" / "none.jsonl")
    shutil.copy(PODC / "rr" / "neg_refusal.jsonl", dl / "rr" / "fill.jsonl")
    shutil.copy(PODC / "rr" / "none.jsonl", dl / "rr" / "fill_half.jsonl")
    r = A.retention(dl, PODC)
    assert r["reading"].startswith("rival (a)") and r["fill_flips"]["shared_with_neg_refusal"] == 8
    shutil.copy(PODC / "rr" / "none.jsonl", dl / "rr" / "fill.jsonl")  # fill flat
    r = A.retention(dl, PODC)
    assert not r["reading"].startswith("rival (a)")
    rows = [json.loads(x) for x in (PODC / "rr" / "none.jsonl").open()]
    rows[0]["refusal"] = not rows[0]["refusal"]  # a none that does not reproduce pod C's
    (dl / "rr" / "none.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows))
    assert "fill_flips" not in A.retention(dl, PODC)


@pytest.mark.slow
def test_dry_run_d_lite_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: the fill arms do not steer (or run at the wrong alpha), the arms do
    # not share rr_none's seeds, or the specificity draws are not assigned by item index mod 5
    inp = ST / "outputs" / "st2_inputs" / "gpt_oss_20b"
    podc = ST / "outputs" / "st2c_inputs" / "gpt_oss_20b"
    if not (inp / "st2_inputs.npz").exists() or not (podc / "st2c_inputs.npz").exists():
        pytest.skip("Phase 2 inputs not built locally")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    a = tmp_path / "a"
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2_driver.py"), "--out", str(a),
                        "--inputs", str(inp), "--dry-run", "--date-pin", "2026-10-21",
                        "--phases", "c1_none"], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = tmp_path / "dl"
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2dl_driver.py"), "--out", str(out),
                        "--inputs", str(inp), "--podc-inputs", str(podc), "--c1",
                        str(a / "gpt_oss_20b"), "--dry-run"],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = out / "gpt_oss_20b"
    rec = json.loads((d / "st2dl_record.json").read_text())["phases"]
    assert all(rec[p]["finished"] for p in D.PODDL)
    rr = {k: [json.loads(x) for x in (d / "rr" / f"{k}.jsonl").open()]
          for k in ("none", "fill", "fill_half")}
    for k in ("fill", "fill_half"):
        assert [x["seed"] for x in rr[k]] == [x["seed"] for x in rr["none"]]
        assert all(x["n_steered"] > 0 for x in rr[k])
    assert rr["fill_half"][0]["alpha"] == pytest.approx(rr["fill"][0]["alpha"] / 2)
    ids = json.loads((ST / "data" / "t15_specificity_v1.json").read_text())["ids"]
    for k in ("fill", "fill_half"):
        sp = [json.loads(x) for x in (d / "spec" / k / "dl_chat_neutral.jsonl").open()]
        assert sp and all(x["fill_draw"] == ids.index(x["scenario_id"]) % 5 for x in sp)
    none = {(x["scenario_id"], x["rollout"]): x["option_logps"]
            for x in map(json.loads, (d / "spec" / "none" / "dl_chat_neutral.jsonl").open())}
    sp = [json.loads(x) for x in (d / "spec" / "fill" / "dl_chat_neutral.jsonl").open()]
    assert any(none[(x["scenario_id"], x["rollout"])] != x["option_logps"] for x in sp)
