"""§13's evaluation pod and cell A′ (scripts/c13_driver.py, scripts/analyze_c13.py; Amendments 36
and 40): the registered sites, the inputs, d1's projection-fraction convention, position validity,
the positive control, the raters' reliability rule, cell A′'s branch, the runner and a dry run."""

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
import analyze_c13 as A  # noqa: E402
import c13_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
INP = ST / "outputs" / "c13_inputs"


def test_registered_sites():
    # most probable failure (Amendment 40 item 1): the extraction reads another layer than d1's
    # headline, or a band outside the registered one
    assert D.SITES["gpt_oss_20b"]["headline"] == 12 and D.SITES["olmo3_think"]["headline"] == 16
    assert D.SITES["olmo3_think"]["band"] == tuple(range(15, 32))
    assert D.SITES["gpt_oss_20b"]["band"] == tuple(range(11, 24))
    assert D.BAND_P == (0.2, 0.6) and D.N_A2 == 60


NEEDED = [INP / f"{s}.json" for s in D.D1_SOURCES]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED), reason="the c13 inputs are not local")
def test_texts_are_d1s_clean_pairs_and_both_documents(tmp_path):
    # most probable failure: the extraction reads the full source files (unclean pairs too) or a
    # changed copy, so V_moral is not d1's
    ts = D.texts(INP, dry=False)
    assert len(ts) == 2 * (573 + 62 + 199) + 2 * (245 + 265) == 2688
    assert len({t["key"] for t in ts}) == len(ts)
    bad = tmp_path / "in"
    bad.mkdir()
    for p in NEEDED:
        (bad / p.name).write_bytes(p.read_bytes())
    (bad / NEEDED[0].name).write_bytes(NEEDED[0].read_bytes() + b" ")
    with pytest.raises(SystemExit):
        D.texts(bad, dry=False)


def test_projection_fraction_is_d1s_norm_fraction():
    # most probable failure: the squared membership is used, so the overlap sits on a different
    # scale than d1's bands and nulls
    Q = A.ortho([np.eye(4)[0], np.eye(4)[1]])
    v = np.array([1.0, 0.0, 1.0, 0.0])
    assert A.frac(Q, v) == pytest.approx(1 / np.sqrt(2))


def _reads(valid=True, u_gpt=1.2, u_olmo=0.4):
    def model(u):
        return {"position_valid": valid,
                "documents": {"spec": {"u": u, "u_boot": list(np.full(A.N_BOOT, u)
                                                              + np.linspace(-0.1, 0.1, A.N_BOOT))}}}
    return {"gpt_oss_20b": model(u_gpt), "olmo3_think": model(u_olmo)}


def test_positive_control():
    # most probable failure: an invalid position (band below its null) still yields a verdict, or
    # the sign of the registered difference is reversed
    assert A.positive_control(_reads())["passes"]
    assert not A.positive_control(_reads(u_gpt=0.3, u_olmo=0.9))["passes"]
    assert A.positive_control(_reads(valid=False))["status"].startswith("uninformative")


def test_ladder_flags_a_band_below_the_null():
    # most probable failure: a degenerate position reads as "above the band floor"
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 16))
    src = {s: rng.standard_normal(16) for s in A.SOURCES}
    lad = A.ladder(X, src, rng, 500)
    assert lad["null_q95"] > 0 and len(lad["heldout"]) == 3
    assert lad["band"][0] <= lad["band"][1]


def test_raters_reliability_rule(tmp_path, monkeypatch):
    # most probable failure: the subagent's labels are used although agreement is below 0.8, or the
    # author's letters are not decoded through the key
    key = {"key": {f"C13S-{i:03d}": {"A": "o1", "B": "o2", "C": "o3"} for i in range(1, 11)}}
    author = {f"C13S-{i:03d}": "B" for i in range(1, 11)}
    sub_hi = {f"C13S-{i:03d}": "o2" for i in range(1, 11)}
    sub_lo = {f"C13S-{i:03d}": ("o2" if i <= 5 else "o1") for i in range(1, 11)}
    monkeypatch.setattr(A, "MODELS", ())
    hi = A.family_read(tmp_path, author, key, sub_hi)
    lo = A.family_read(tmp_path, author, key, sub_lo)
    assert hi["reliability"]["agreement"] == 1.0 and hi["labels_of_record"].startswith("subagent")
    assert lo["reliability"]["agreement"] == 0.5 and lo["labels_of_record"].startswith("author")


def test_remote_runs_c13():
    # most probable failure: the GPT-OSS refusal-retention pass rides this pod, or cell A' arms run
    # on OLMo-3-Think
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "c13" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-gpt_oss_20b '
                     r'olmo3_think\}"; OUT_SUB=c13\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-2\.5\}"', sh)
    assert '[ "$STEPS" != "c13" ]; then  # Amendment 10' in sh
    assert '[ "$STEPS" != "c13" ]; then  # of record from T1.1' in sh
    blk = sh.split('if [ "$STEPS" = "c13" ]; then\n    rc=0')[1][:900]
    assert blk.index("--step base") < blk.index("--step cellA2_arm")
    assert '[ "$m" = "gpt_oss_20b" ]' in blk
    assert "test_c13.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_c13.txt").read_text().split()
    assert {f"outputs/c13_inputs/{s}.json" for s in D.D1_SOURCES} <= set(sync)
    assert all(not x.endswith("/") for x in sync)


@pytest.mark.slow
def test_dry_run_every_step(tmp_path):
    # most probable failure: a step crashes on a tiny model of either family, or the analysis
    # reads a verdict at an invalid position
    if not all(p.exists() for p in NEEDED):
        pytest.skip("the c13 inputs are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "c13_driver.py"), "--out", str(tmp_path), "--dry-run"]
    for m in D.SITES:
        r = subprocess.run(base + ["--step", "base", "--model", m, "--d1", str(INP)],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for arm in ("beh2", "beh"):
        r = subprocess.run(base + ["--step", "cellA2_arm", "--arm", arm, "--adapters",
                                   str(tmp_path / "adapters")],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads((tmp_path / "c13_record.json").read_text())["phases"]
    assert {"base_gpt_oss_20b", "base_olmo3_think", "cellA2_beh2", "cellA2_beh"} <= set(rec)
    res = {m: A.model_read(tmp_path, m) for m in D.SITES}
    for m, r in res.items():
        if not r["position_valid"]:
            assert r["documents"]["spec"]["ladder_position"].startswith("position invalid")
    a2 = A.cell_a2(tmp_path)
    assert a2["reading"] and a2["beh2"]["dP_violating"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_generic_screen_dry_run_skips_olmo3_think_on_c13():
    # most probable failure (the first c13 pod): the runner's generic per-model screen dry run reads
    # the registry's vLLM block, which OLMo-3-Think lacks, and stops the session before any read
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    blk = sh.split("dry_runs() {")[1][:600]
    assert 'if [ "$STEPS" = "c13" ] && [ "$m" = "olmo3_think" ]; then' in blk
    import pod_st_screen as P
    assert P.registry()["olmo3_think"].get("vllm") is None


def test_launcher_forwards_session_max_hours():
    # most probable failure (the T1.6 resume): the launcher drops SESSION_MAX_HOURS, so the pod
    # arms its step default; an unset value forwards as empty and must still fall to that default
    sh = (ROOT / "papers" / "d1_moral_subspace" / "runpod" / "run_session.sh").read_text()
    assert "SESSION_MAX_HOURS='${SESSION_MAX_HOURS:-}'" in sh.split("setsid bash")[0]
    r = subprocess.run(["bash", "-c", 'SESSION_MAX_HOURS=""; echo "${SESSION_MAX_HOURS:-2.5}"'],
                       capture_output=True, text=True)
    assert r.stdout.strip() == "2.5"
