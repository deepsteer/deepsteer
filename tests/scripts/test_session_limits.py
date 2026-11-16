"""Editable pod time limits: papers/runpod_common/session_limits.py, session_limits.sh,
set_limits.sh, their readers (cst_finalist_pod.deadline, gpc1_pod.fits), and rate_watch.line."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RC = ROOT / "papers/runpod_common"
sys.path[:0] = [str(RC), str(ROOT / "papers/cst_constitutional_steering/scripts"),
                str(ROOT / "papers/gpc_goal_pressure_conduct/scripts"),
                str(ROOT / "papers/kdg_panel/scripts"),
                str(ROOT / "papers/st_live_reference/scripts"),
                str(ROOT / "papers/mrb_morebench/scripts"),
                str(ROOT / "papers/mch_machiavelli/scripts"), str(ROOT)]

import session_limits as SL  # noqa: E402


def test_guards_hold_the_ceiling_and_the_gap():
    # most probable failure: a typo (watchdog 41 h) or a cut past the watchdog is accepted, so a
    # session runs away or the watchdog kills a stage the cut would have let finish
    assert SL.validate(10.25, 11.0) == []
    assert SL.validate(13.5, 14.0) == []
    assert SL.validate(10.25, 14.1) and SL.validate(11.0, 11.2) and SL.validate(-1, 11.0)
    assert SL.CEILING_H == 14.0 and SL.DEFAULT_CUT_H == 10.25


def test_set_moves_the_deadline_live_and_logs_the_reason(tmp_path, monkeypatch):
    # most probable failure: a set rewrites the file but the deadline readers keep the start-time
    # value (read once), or a rejected set half-writes the file
    f = tmp_path / "limits.json"
    start = time.time() - 3600
    SL.init(f, start)
    monkeypatch.setenv("CST_LIMITS_FILE", str(f))
    assert SL.deadline_epoch() == pytest.approx(start + 10.25 * 3600)
    r = SL.set_limits(f, "the registered read needs 0.5 h more", cut=10.75, watch=11.5)
    assert r["cut_h"] == 10.75 and abs(r["elapsed_h"] - 1.0) < 0.01
    assert SL.deadline_epoch() == pytest.approx(start + 10.75 * 3600) and SL.cut_h() == 10.75
    with pytest.raises(SystemExit):
        SL.set_limits(f, "runaway", watch=20.0)
    with pytest.raises(SystemExit):
        SL.set_limits(f, "", cut=9.0)
    assert SL.read(f)["watch_h"] == 11.5  # unchanged by the rejected sets
    log = [json.loads(x) for x in (tmp_path / "limits.json.log.jsonl").read_text().splitlines()]
    assert [x["action"] for x in log] == ["init", "set"]
    assert log[1]["reason"].startswith("the registered") and log[1]["old"]["cut_h"] == 10.25
    r2 = SL.set_limits(f, "shorten", cut=0.5, watch=1.0)
    assert r2["note"]  # a cut already past: stages stop at their next check


def test_readers_fall_back_to_the_environment_without_a_file(monkeypatch, tmp_path):
    # most probable failure (backward compatibility): a runner that never opted in now reads a
    # default cut instead of its own CST_DEADLINE_EPOCH, or crashes on the missing file
    import cst_finalist_pod as F

    monkeypatch.delenv("CST_LIMITS_FILE", raising=False)
    monkeypatch.setenv("CST_START_EPOCH", "1000")
    monkeypatch.setenv("CST_DEADLINE_EPOCH", str(1000 + 9.75 * 3600))
    assert F.deadline() == 1000 + 9.75 * 3600 and SL.cut_h() == pytest.approx(9.75)
    f = tmp_path / "l.json"
    SL.init(f, 2000.0, cut=10.25, watch=11.0)
    monkeypatch.setenv("CST_LIMITS_FILE", str(f))
    assert F.deadline() == pytest.approx(2000 + 10.25 * 3600)
    SL.set_limits(f, "extend", cut=11.0, watch=11.75)
    assert F.deadline() == pytest.approx(2000 + 11.0 * 3600)


def test_gpc_cut_rule_reads_the_file(monkeypatch, tmp_path):
    # most probable failure: the cut rule keeps its constant after an extension, so it cuts a
    # stage the new limit would fit
    import gpc1_pod as P

    f = tmp_path / "l.json"
    start = time.time() - 8.5 * 3600
    SL.init(f, start)
    monkeypatch.setenv("CST_LIMITS_FILE", str(f))
    monkeypatch.setenv("CST_START_EPOCH", str(start))
    assert P.cut_h() == 10.25
    assert not P.fits(tmp_path, "registered")  # 8.5 + 1.9 h of registered read > 10.25
    SL.set_limits(f, "extend for the registered read", cut=11.0, watch=11.75)
    assert P.fits(tmp_path, "registered") and P.cut_h() == 11.0
    monkeypatch.delenv("CST_LIMITS_FILE")
    assert P.cut_h() == P.CUT_H == 10.25


def _shell(script: str, env: dict) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", script], env={**os.environ, **env},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_watchdog_loop_kills_at_the_files_hour_and_respects_an_extension(tmp_path):
    # most probable failure: the watchdog still sleeps a fixed time from start (so an extension
    # does not move it), or it never fires
    tag = f"gpc_limits_dummy_{os.getpid()}"
    for extend in (False, True):
        f = tmp_path / f"l{int(extend)}.json"
        start = time.time() - 11.0 * 3600 + 3  # the watchdog hour falls 3 s from now
        SL.init(f, start)
        dummy = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep(25)  # {tag}"])
        wd = _shell(f'source "{RC}/session_limits.sh"; export CST_LIMITS_FILE="{f}"; '
                    f'limits_watchdog "{tag}"; wait $WATCHDOG',
                    {"LIMITS_POLL_S": "1", "SL_PY": str(RC / "session_limits.py")})
        if extend:
            SL.set_limits(f, "extend", cut=11.0, watch=11.5)
        time.sleep(7)
        alive = dummy.poll() is None
        dummy.kill()
        wd.kill()
        assert alive is extend, (extend, wd.stdout.read() if not extend else "")


def test_set_limits_script_end_to_end(tmp_path):
    # most probable failure: the laptop command quotes the reason so the pod's shell splits it, or
    # it reports a change the pod rejected
    repo = tmp_path / "repo"
    (repo / "papers/runpod_common").mkdir(parents=True)
    (repo / "papers/runpod_common/session_limits.py").write_text(
        (RC / "session_limits.py").read_text())
    f = repo / ".session_limits.json"
    SL.init(f, time.time() - 2 * 3600)
    cmd = ["bash", str(RC / "set_limits.sh"), "--ssh", "bash -c", "--repo", str(repo),
           "--cut", "11.0", "--reason", "it's for the registered read; keep it", "--rate", "1.15"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert SL.read(f)["cut_h"] == 11.0 and SL.read(f)["watch_h"] == 11.75
    log = (repo / ".session_limits.json.log.jsonl").read_text()
    assert "it's for the registered read; keep it" in log and "+0.86" in r.stdout
    bad = subprocess.run(cmd[:6] + ["--watch", "15", "--reason", "x"], capture_output=True,
                         text=True)
    assert bad.returncode != 0 and SL.read(f)["watch_h"] == 11.75


def test_rate_watch_prints_a_window_with_no_forecast():
    # most probable failure (GPC-1, 2026-11-15): a training stage's window has a rate but no
    # forecast, and formatting the None ratio crashes the watcher for the rest of the session
    import rate_watch as RW

    rec = {"stage": "train", "window_tps": 812.0, "forecast_tps": None, "ratio": None,
           "flag": False}
    want = ">> rate window (train): 812 tokens/s against None forecast (no forecast)"
    assert RW.line(rec) == want
    assert "no window" in RW.line({**rec, "window_tps": None})
    assert "RATE FLAG" in RW.line({**rec, "forecast_tps": 900.0, "ratio": 0.6, "flag": True})


def test_gpc_runner_opts_in_and_keeps_no_fixed_watchdog():
    # most probable failure: the runner writes the limits file but keeps its fixed-sleep watchdog
    # (so an extension past 11.0 h is killed anyway), or never saves the change log
    run = (ROOT / "papers/gpc_goal_pressure_conduct/runpod/remote_gpc1.sh").read_text()
    i, w = run.index("limits_init "), run.index("limits_watchdog ")
    assert run.index("session_limits.sh") < i < w and "limits_save" in run
    assert 'sleep "$(python3 -c "print(int($MAX_H * 3600))")"' not in run
