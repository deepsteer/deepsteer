"""The care_harm ladder on Muse (scripts/st2ml_driver.py, scripts/analyze_st2ml.py; Amendment 31
items 1 and 4): sign classes, the branch rules on stand-in rows, the runner's order, and a dry run
on a tiny Cell 4."""

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
import analyze_st2ml as A  # noqa: E402
import st2ml_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
C4 = ST / "outputs" / "st2m" / "muse_glimmer_30b"


def test_sign_classes():
    # most probable failure (Amendment 31 item 1): a violating-down read past the bar is scored as
    # a pass or a failure instead of the flagged class
    assert A.sign_class(0.05) == "non-specific"
    assert A.sign_class(-0.048) == "discriminating"
    assert A.sign_class(0.02) == A.sign_class(-0.03) == "specific"
    assert A.sign_class(None) is None


def test_rungs_and_order():
    # most probable failure: a spec read runs after an arm, or the rungs are not Amendment 31's
    assert D.RUNGS == {"03": 0.3, "04": 0.4}
    first_arm = min(i for i, p in enumerate(D.PHASES) if p.startswith("arm_"))
    assert all(i < first_arm for i, p in enumerate(D.PHASES) if p.startswith("spec_"))
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2ml" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-3\.0\}"', sh)
    sync = (ST / "runpod" / "sync_st2ml.txt").read_text().split()
    need = {f"outputs/st2m/muse_glimmer_30b/{f}" for f in (
        "st2m_record.json", "derive.npz", "directions.npz", "gen/arm_none.jsonl",
        "gen/arm_fill.jsonl", "spec/none/dl_chat_neutral.jsonl")}
    assert need <= set(sync) and all(not x.endswith("/") for x in sync)  # files cross the excludes


# the pod syncs only some of Cell 4's files: skip unless every file this test reads is here
NEEDED = [C4 / "gen" / f"arm_{a}.jsonl" for a in ("none", "fill", "care_harm", "V_qwen")] + [
    C4 / "spec" / a / "dl_chat_neutral.jsonl" for a in ("none", "fill", "care_harm", "V_qwen")]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="Cell 4 rows not local")
def test_branches_on_stand_in_rows(tmp_path):
    # most probable failure: R_a is read without a dose rise at the top coherent rung, or R_b is
    # read while a rung is discriminating
    def build(arm03: str, arm04: str, spec: str) -> Path:
        d = tmp_path / f"{arm03}_{arm04}_{spec}"
        (d / "gen").mkdir(parents=True)
        shutil.copy(C4 / "gen" / f"arm_{arm03}.jsonl", d / "gen" / "arm_care_03.jsonl")
        shutil.copy(C4 / "gen" / f"arm_{arm04}.jsonl", d / "gen" / "arm_care_04.jsonl")
        for r in ("03", "04"):
            for k, src in (("care", spec), ("fill", "fill")):
                (d / "spec" / f"{k}_{r}").mkdir(parents=True)
                shutil.copy(C4 / "spec" / src / "dl_chat_neutral.jsonl",
                            d / "spec" / f"{k}_{r}" / "dl_chat_neutral.jsonl")
        (d / "st2ml_record.json").write_text(json.dumps({"phases": {}}))
        return d
    r = A.read(build("care_harm", "V_qwen", "care_harm"), C4)  # rises at 0.4x, rule-side reads
    assert r["branch"].startswith("R_a")
    r = A.read(build("care_harm", "care_harm", "none"), C4)  # flat, specific
    assert r["branch"].startswith("R_b")
    r = A.read(build("care_harm", "V_qwen", "V_qwen"), C4)  # rises, but violating-up
    assert r["branch"] == "unresolved"


@pytest.mark.slow
def test_dry_run_ladder_on_a_tiny_cell4(tmp_path):
    # most probable failure: the ladder reads Cell 4's files at the wrong width or seeds, or its
    # arms do not pair with Cell 4's none
    qt = ST / "outputs" / "st2b" / "gpt_oss_20b" / "gen" / "c3_qwen_topup.jsonl"
    qs = ST / "outputs" / "t10" / "qwen38_27b" / "screen" / "h1_screen.jsonl"
    if not qt.exists() or not qs.exists():
        pytest.skip("Qwen3.8's rows not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    m = [sys.executable, str(SCRIPTS / "st2m_driver.py"), "--out", str(tmp_path / "c4"),
         "--dry-run", "--date-pin", "2026-10-23"]
    for extra in (["--phases", "v1"], ["--qwen-topup", str(qt), "--qwen-screen", str(qs)]):
        r = subprocess.run(m + extra, capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    c4 = tmp_path / "c4" / "muse_glimmer_30b"
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2ml_driver.py"), "--out",
                        str(tmp_path / "lad"), "--cell4", str(c4), "--dry-run"],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    d = tmp_path / "lad" / "muse_glimmer_30b"
    rec = json.loads((d / "st2ml_record.json").read_text())["phases"]
    assert all(rec[p]["finished"] for p in D.PHASES)
    assert rec["spec_none"]["parity"]["prompts_match"].split(" of ")[0] == \
        rec["spec_none"]["parity"]["prompts_match"].split(" of ")[1]
    med = json.loads((c4 / "st2m_record.json").read_text())["phases"]["derive"]["median_norm"]
    assert rec["arm_care_04"]["alpha"] == pytest.approx(0.4 * med)
    none = {json.loads(x)["rollout_key"] for x in (c4 / "gen" / "arm_none.jsonl").open()}
    lad = [json.loads(x)["rollout_key"] for x in (d / "gen" / "arm_care_03.jsonl").open()]
    assert lad and set(lad) <= none
