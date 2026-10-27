"""The second family's pod 1 on Qwen3.8-27B (ST_PLAN Amendment 54 and its addition;
scripts/st2q_driver.py, scripts/analyze_st2q.py): the site rule, the gap bound, the re-price stop,
the gate, the branch, the runner and a dry run."""

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
import analyze_st2q as A  # noqa: E402
import st2q_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"


def test_site_rule():
    # most probable failure (the addition, item 3): block 32 taken although its PR median is below
    # 30, or the fallback goes deeper when a shallower healthy block is as near
    ok, low = [31.0] * 21, [29.0] * 21
    assert D.pick_site({24: ok, 28: ok, 32: ok, 36: ok, 40: ok})[0] == 32
    assert D.pick_site({24: ok, 28: ok, 32: low, 36: ok, 40: ok})[0] == 28
    assert D.pick_site({24: ok, 28: low, 32: low, 36: ok, 40: ok})[0] == 36
    assert D.pick_site({24: [29.0] * 10 + [31.0] * 11, 32: low})[0] == 24  # median, not seed 0
    with pytest.raises(SystemExit):
        D.pick_site({24: low, 32: low})


def test_gap_bound_is_on_the_arm_scenarios():
    # most probable failure: the gap a claim is bounded by is read over the whole set, where
    # Qwen's reasoning raises protection, instead of the arm (engaged) scenarios
    pd = {"a": 0.8, "b": 0.6, "c": 0.9}
    pn = {"a": 0.2, "b": 0.4, "c": 0.95}
    g = D.gap_bound(pd, pn, ["a", "b"])
    assert g["n"] == 2 and g["mean"] == pytest.approx(0.4)


def _rows(arm, p_by_sid, n):
    return [{"scenario_id": s, "rollout_key": f"st2q|{s}|{k}", "rollout": k, "letter": "A",
             "norm_status": "violating" if k < round(p * n) else "consistent",
             "status": "completed", "option_id": "o1", "arm": arm}
            for s, p in p_by_sid.items() for k in range(n)]


def test_gate_reads_the_site_fill_arm():
    # most probable failure: the fill clause is skipped, so a protect shift any vector of the
    # same norm makes reads as the gate passing
    sids = [f"S{i:02d}" for i in range(12)]
    care = _rows("care", {s: 0.5 for s in sids}, 16)
    none = _rows("none", {s: 0.25 for s in sids}, 16)
    g = A.gate(care, none, _rows("fill", {s: 0.5 for s in sids}, 8), {"passes": True})
    assert g["dP_vs_none"]["mean"] == pytest.approx(0.25) and not g["passes"]
    g = A.gate(care, none, _rows("fill", {s: 0.25 for s in sids}, 8), {"passes": True})
    assert g["passes"]


def test_branch():
    # most probable failure (the addition, item 2): a passing gate on fewer than 10 scenarios, or
    # a passing but non-specific direction, licenses pod 2
    ok = {"passes": True}
    spec = lambda m: {"vs_fill10": {"mean": m}}  # noqa: E731
    assert A.branch(ok, spec(0.0), False, 18).startswith("P1-pass")
    assert A.branch(ok, spec(0.05), False, 18).startswith("P1-nonspecific")
    assert A.branch(ok, spec(0.0), False, 8).startswith("P1-fail by rule")
    assert A.branch({"passes": False}, spec(0.0), False, 18).startswith("P1-fail")
    assert A.branch(None, None, True, 18).startswith("P1-fail: no coherent dose")


def test_reprice_stop(tmp_path):
    # most probable failure: the 1.5x stop compares the remaining hours alone, ignoring the
    # session's elapsed hours, so a slow pilot never stops the pod
    class Args:
        session_t0 = 0.0

    c = {"record": tmp_path / "rec.json", "engaged": ["s"] * 18, "sweep": ["x"] * 5,
         "spec_ids": list(range(60)), "r": 16}
    c["record"].write_text(json.dumps({"phases": {
        "pilot": {"seconds_per_rollout": 10.0}, "bank": {"read_seconds_per_rollout": 0.5},
        "dose0": {"seconds_per_row": 0.2}}}))
    pr = D.projection(c, Args(), now=6 * 3600.0)
    assert pr["stops"] and pr["projected_pod_hours"] > 6.0
    pr = D.projection(c, Args(), now=0.5 * 3600.0)
    assert not pr["stops"]


def test_runner():
    # most probable failure: the step syncs nothing yet needs a synced input, the stop is shorter
    # than the 1.5x re-price bound, or the HF process starts without the session's start time
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2q" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-qwen38_27b'
                     r'\}"; OUT_SUB=st2q\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-7\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "st2q" ]; then\n    DEADLINE=')[1].split("continue")[0]
    assert "--phases v1" in blk and '--session-t0 "$SESSION_T0"' in blk
    assert blk.index("--phases v1") < blk.index("--session-t0")
    assert "test_st2q.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    assert '[ "$STEPS" != "st2q" ]' in sh.split("# re-priced locally")[0].split("\n")[-1]
    assert D.PRICE_POD1 * D.STOP_FACTOR <= 7.5


@pytest.mark.slow
def test_dry_run(tmp_path):
    # most probable failure: a phase crashes on a tiny Qwen (the DeltaNet layers under the steer
    # hook, the teacher-forced prediction, the site context), or the arms' rows do not pair
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2q_driver.py"), "--out", str(tmp_path), "--dry-run",
            "--date-pin", "2026-10-27"]
    r = subprocess.run(base + ["--phases", "v1"], capture_output=True, text=True, cwd=ROOT,
                       env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    r = subprocess.run(base, capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rec = json.loads((tmp_path / D.KEY / "st2q_record.json").read_text())["phases"]
    assert all(p in rec for p in D.HF_PHASES)
    assert rec["predict"]["n_traces"] > 0
    assert rec["arm_care"]["coherence"] is not None
    out = A.read(tmp_path / D.KEY)
    assert out["branch"].startswith("P1-")
