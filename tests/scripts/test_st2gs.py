"""The GPT-OSS site sweep (ST_PLAN Amendment 51 and its additions; scripts/build_st2gs_inputs.py,
scripts/st2gs_driver.py, scripts/analyze_st2gs.py): eligibility, the trace positions, the
ranking read, the gate, the fallback rule, the branch, the runner and a dry run."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_st2gs as A  # noqa: E402
import st2gs_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
GS = ST / "outputs" / "st2gs_inputs" / "gpt_oss_20b"
FIVE = {"b9_clause_end", "b15_onset", "b15_clause_end", "b18_onset", "b18_clause_end"}


def _inputs():
    if not (GS / "sites.json").exists():
        pytest.skip("the sweep's inputs are not local")
    return json.loads((GS / "sites.json").read_text())


def test_eligibility_is_the_registered_five():
    # most probable failure (first addition, item 1): eligibility read at seed 0 instead of the
    # median of 21 samples, or block 12's clause_end (read under Amendment 50) left in
    doc = _inputs()
    assert set(doc["candidates"]) == FIVE
    b12 = doc["sites"]["b12_clause_end"]
    assert b12["read"] and not b12["eligible"] and b12["pr_median_21"] >= 30
    assert doc["sites"]["b18_clause_end"]["pr_seed0"] < 30  # eligible only under the median rule
    for s in doc["sites"].values():
        assert s["alpha"]["04"] == pytest.approx(0.4 * s["median_norm"])
    assert doc["sites"]["b12_clause_end"]["median_norm"] == pytest.approx(1336.5, abs=0.1)


def test_trace_positions_replay_the_banked_ones():
    # most probable failure: the replayed position rule (the one the arms steer by) puts the
    # exact steer on other tokens than T1.1 banked, so the ranking steers somewhere else
    doc = _inputs()
    tr = [json.loads(x) for x in (GS / "traces.jsonl").open()]
    assert len(tr) == doc["traces"]["n"] == 90
    assert len({t["scenario_id"] for t in tr}) == 45
    assert all(t["bank_match"]["onset"] and t["bank_match"]["clause_end"] for t in tr)
    assert all(t["violating_id"] != t["consistent_id"] for t in tr)
    assert all(max(t["positions"]["clause_end"], default=0) < len(t["ids"]) for t in tr)


def test_fallback_runs_only_after_an_incoherent_04_that_fits_the_stop():
    # most probable failure (second addition, item 2): 0.3x runs although 0.4x was coherent, or
    # starts when its projected cost passes the stop
    rec = {f"{p}|s1|04": {"seconds": 100.0} for p in D.SITE_PHASES}
    rec["arm_care|s1|04"].update(coherent=False)
    rec.update({f"{p}|s2|04": {"seconds": 100.0} for p in D.SITE_PHASES})
    rec["arm_care|s2|04"].update(coherent=True)
    assert D.fallback_plan(rec, ["s1", "s2"], deadline=1000.0, now=0.0) == {
        "s1": {"projected_seconds": 400.0, "runs": True}}
    assert not D.fallback_plan(rec, ["s1", "s2"], deadline=300.0, now=0.0)["s1"]["runs"]


def test_rank_read_and_validation(tmp_path):
    # most probable failure: the ranking averages traces instead of scenarios, or AtP's Spearman
    # is read against the AtP vector itself
    names = ["b6_onset", "b9_clause_end", "b12_clause_end"]
    scen = np.repeat([f"S{i}" for i in range(10)], 2)
    exact = np.tile([0.1, 0.3, 0.2], (20, 1)).astype(float)
    exact[0, 0] = 5.0  # one trace of one scenario: the scenario mean moves by half of it
    atp = exact[:, [0, 1, 2]] * 0.5
    np.savez(tmp_path / "rank.npz", sites=np.array(names), m0=np.zeros(20), m_grad=np.zeros(20),
             exact=exact, atp=atp, atp_fill=np.zeros((20, 3, 10)), scenario=scen,
             rollout=np.zeros(20))
    out = A.rank_read(tmp_path, {"rank": {"top2": ["b9_clause_end", "b6_onset"],
                                          "atp": "computed"}})
    assert out["sites"]["b6_onset"]["exact"]["mean"] == pytest.approx(0.1 + 4.9 / 2 / 10)
    assert out["atp_spearman"] == pytest.approx(1.0) and out["atp_validates"]
    assert out["calibration_b12"]["measured_dP"]["04"] == 0.041


def _rows(arm, p_by_sid, n):
    return [{"scenario_id": s, "rollout_key": f"st2c1|{s}|{k}", "rollout": k, "letter": "A",
             "norm_status": "violating" if k < round(p * n) else "consistent",
             "status": "completed", "option_id": "o1", "arm": arm}
            for s, p in p_by_sid.items() for k in range(n)]


def test_gate_reads_the_site_fill_arm():
    # most probable failure: the gate's fill clause reads Cell 1's block 12 fill (0.2x) instead of
    # the site's own fill arm (first addition, item 4)
    sids = [f"S{i:02d}" for i in range(20)]
    care = _rows("care", {s: 0.5 for s in sids}, 16)
    none = _rows("none", {s: 0.25 for s in sids}, 16)
    site_fill = _rows("fill", {s: 0.5 for s in sids}, 8)
    c1_fill = _rows("fill", {s: 0.25 for s in sids}, 16)
    g = A.gate(care, none, site_fill, c1_fill, {"passes": True})
    assert g["dP_vs_none"]["mean"] == pytest.approx(0.25)
    assert g["dP_vs_site_fill"]["mean"] == pytest.approx(0.0) and not g["passes"]
    assert g["dP_vs_cell1_fill_beside"]["mean"] == pytest.approx(0.25)


def test_branch():
    # most probable failure: a passing but non-specific site reads as R_site, or the recipe's
    # site is not the larger dP among the specific ones
    def site(dp, passes, spec, coherent=True):
        return {"rung_of_record": "04" if coherent else None,
                "04": {"gate": {"passes": passes, "coherent": coherent,
                                "dP_vs_none": {"mean": dp}},
                       "specificity": {"vs_fill10": {"mean": spec}}}}
    assert A.branch({"a": site(0.2, True, 0.0), "b": site(0.3, True, 0.0)}).startswith(
        "R_site: a specific protective lever at b")
    assert A.branch({"a": site(0.2, True, 0.05)}).startswith("R_model")
    assert A.branch({"a": site(0.05, False, 0.0), "b": site(0.3, True, 0.0, False)}).startswith(
        "R_lever extended")


def test_runner_and_sync():
    # most probable failure: the sweep runs without its inputs, the GPT-OSS refusal-retention pass
    # rides this pod, or the stop is shorter than the sweep
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2gs" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-gpt_oss_20b'
                     r'\}"; OUT_SUB=st2gs\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-10\.0\}"', sh)
    a10 = [x for x in sh.splitlines() if x.rstrip().endswith("# Amendment 10")]
    assert a10 and all('[ "$STEPS" != "st2gs" ]' in x for x in a10)
    assert "test_st2gs.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    assert '[ "$STEPS" != "st2gs" ]' in sh.split("# re-priced locally")[0].split("\n")[-1]
    blk = sh.split('if [ "$STEPS" = "st2gs" ]; then\n    DEADLINE=')[1].split("continue")[0]
    assert '--gs "$ST2GS_INPUTS"' in blk and '--deadline "$DEADLINE"' in blk
    sync = (ST / "runpod" / "sync_st2gs.txt").read_text().split()
    for f in ("sites.json", "fill_sites.npz", "traces.jsonl"):
        assert f"outputs/st2gs_inputs/gpt_oss_20b/{f}" in sync
    assert "outputs/st2a/gpt_oss_20b/gen/c1_none.jsonl" in sync


@pytest.mark.slow
def test_dry_run(tmp_path):
    # most probable failure: a phase crashes on a tiny GPT-OSS (the AtP backward, the steer hook,
    # the site context), or the sweep's rows do not pair with Cell 1's none
    inp = ST / "outputs" / "st2_inputs" / "gpt_oss_20b"
    podc = ST / "outputs" / "st2c_inputs" / "gpt_oss_20b"
    if not ((inp / "st2_inputs.npz").exists() and (GS / "traces.jsonl").exists()):
        pytest.skip("the sweep's inputs are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    r = subprocess.run([sys.executable, str(SCRIPTS / "st2_driver.py"), "--out",
                        str(tmp_path / "st2"), "--dry-run", "--inputs", str(inp), "--date-pin",
                        "2026-10-26", "--phases", "c1_none"], capture_output=True, text=True,
                       cwd=ROOT, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    cmd = [sys.executable, str(SCRIPTS / "st2gs_driver.py"), "--out", str(tmp_path / "gs"),
           "--dry-run", "--inputs", str(inp), "--podc-inputs", str(podc), "--c1",
           str(tmp_path / "st2" / "gpt_oss_20b"), "--gs", str(GS)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    out = tmp_path / "gs" / "gpt_oss_20b"
    rec = json.loads((out / "st2gs_record.json").read_text())["phases"]
    assert rec["rank"]["atp"] == "computed" and len(rec["rank"]["top2"]) == 2
    for s in rec["rank"]["top2"]:
        assert all(f"{p}|{s}|04" in rec for p in D.SITE_PHASES)
    z = np.load(out / "rank.npz")
    assert np.isfinite(z["atp"]).all() and (z["exact"] != 0).any()
    rows = [json.loads(x) for x in (out / "gen" / f"gs_fill_{rec['rank']['top2'][0]}_04.jsonl")
            .read_text().splitlines() if x]
    assert {r["fill_draw"] for r in rows} and all(r["site"] for r in rows)
    # the fallback path on resume: one site's 0.4x care arm made incoherent, the plan removed
    s0 = rec["rank"]["top2"][0]
    full = json.loads((out / "st2gs_record.json").read_text())
    full["phases"][f"arm_care|{s0}|04"]["coherent"] = False
    del full["phases"]["fallback"]
    (out / "st2gs_record.json").write_text(json.dumps(full))
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rec = json.loads((out / "st2gs_record.json").read_text())["phases"]
    assert rec["fallback"]["plan"][s0]["runs"]
    assert all(f"{p}|{s0}|03" in rec for p in D.SITE_PHASES)
    assert not any(f"{p}|{rec['rank']['top2'][1]}|03" in rec for p in D.SITE_PHASES)
