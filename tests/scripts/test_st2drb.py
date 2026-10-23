"""The D_rb rider (scripts/st2drb_driver.py, scripts/analyze_st2drb.py; Amendments 34-35): the
position definition against T1.1's bank, the build gate and the D_h1 rule on synthetic shifts, the
runner's order and sync, and a dry run of every step on a tiny GPT-OSS."""

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
import analyze_st2drb as A  # noqa: E402
import st2drb_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
OUTS = ST / "outputs"
T11 = OUTS / "t11" / "gpt_oss_20b"
POOL = OUTS / "st2lb" / "gpt_oss_20b" / "gen" / "rr_rollouts.jsonl"


def test_design_constants():
    # most probable failure (Amendment 34): the rider reads a different site than L-beh-3 steers
    # (hidden 13, clause_end), or the positive-control arms read the pool L-beh-3's vector is fit on
    assert D.H_STAR == 13 and D.CLASSES[0] == "clause_end" and D.R_SPEC == 4
    assert D.ARMS[:2] == ("none", "beh") and set(D.ARMS[2:]) == {"beh2", "anchor", "beh_epoch3"}
    assert D.MANIFESTS["beh"] == "t15_adapters_v1.json"
    assert A.SPEC_OF_RECORD["beh2"] < A.SPEC_OF_RECORD["anchor"] < A.SPEC_OF_RECORD["beh"]


NEEDED = [T11 / "gen" / "h1_rollouts.jsonl", T11 / "bank" / "bank_0000.npz"]


@pytest.mark.skipif(not all(p.exists() for p in NEEDED),
                    reason="T1.1's rollouts and bank not local")
def test_positions_match_t11s_bank():
    # most probable failure: the rider's clause_end positions differ from the bank's (another
    # tracker or offset), so D_h1's base side and every class mean read other tokens
    import pod_st_screen as P
    import steering_gate as SG
    import t11_driver as T11D
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()[D.KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    fmt, punct = build_format(D.KEY, tok, spec), SG.punct_mask(tok)
    rows = [json.loads(x) for _, x in zip(range(T11D.CHUNK), (T11 / "gen" /
                                                              "h1_rollouts.jsonl").open())]
    z = np.load(T11 / "bank" / "bank_0000.npz")
    checked = 0
    for i, r in enumerate(rows[:6]):
        if not r.get("gen_ids"):
            continue
        n_p = len(r["prompt_ids"])
        cp = T11D.class_positions(fmt, tok, r["gen_ids"], punct)
        mine = sorted(n_p + j for j, cl in cp.items() if "clause_end" in cl)
        bank = sorted(int(p) for p, c in zip(z["pos"][z["row"] == i], z["cls"][z["row"] == i])
                      if c & 2)
        assert mine == bank, i
        checked += 1
    assert checked >= 4


def _write(d: Path, arm: str, keys: list[str], ce: np.ndarray, sample=None) -> None:
    n, w = ce.shape
    m = np.zeros((n, 3, w), np.float32)
    m[:, 0] = ce
    extra = {"ce_sample_h13": sample.astype(np.float16)} if sample is not None else {}
    np.savez(d / f"{arm}.npz", keys=np.array(keys), sets=np.array([k.split("|")[0] for k in keys]),
             counts=np.ones((n, 3), np.int32), classes=np.array(D.CLASSES),
             hidden=np.array(D.HIDDEN), mean_cls_h13=m,
             mean_ce_layers=np.repeat(ce[:, None], 5, 1).astype(np.float16), **extra)


def _synthetic(d: Path, *, h1_aligned: bool, beh2_sign: float, scale: float = 1.0,
               noise: float = 0.1) -> None:
    rng = np.random.default_rng(1)
    w = 48
    u, v = np.zeros(w), np.zeros(w)
    u[0], v[1] = 1.0, 1.0
    keys = ([f"spec|S{i}|{k}" for i in range(30) for k in range(4)]
            + [f"pool|P{i}|{k}" for i in range(40) for k in range(4)]
            + [f"h1|H{i}|{i * 8 + k}" for i in range(20) for k in range(4)])
    base = rng.standard_normal((len(keys), w)) * 3
    sample = rng.standard_normal((2048, w)) * np.linspace(3, 0.3, w)

    def shifted(on_spec_pool: np.ndarray, on_h1: np.ndarray, spec_only=False):
        out, ks = [], []
        for k, b in zip(keys, base):
            s = k.split("|")[0]
            if spec_only and s != "spec":
                continue
            sh = on_h1 if s == "h1" else on_spec_pool
            out.append(b + scale * sh + rng.standard_normal(w) * noise)
            ks.append(k)
        return ks, np.array(out)

    _write(d, "none", keys, base, sample)
    h1 = u if h1_aligned else v  # aligned: D_h1 the same axis, so inside D_rb's own band
    _write(d, "beh", *shifted(u, h1))
    for arm, f in (("beh_epoch3", 0.95), ("anchor", 0.3), ("beh2", beh2_sign)):
        _write(d, arm, *shifted(f * u, h1, spec_only=True))


@pytest.fixture
def fast(monkeypatch):
    monkeypatch.setattr(A, "N_BOOT", 200)
    monkeypatch.setattr(A, "N_NULL", 2000)


def test_gate_builds_on_a_clean_rule_breaking_axis(tmp_path, fast):
    # most probable failure: the positive control projects on D_rb(spec) (in-sample for L-beh)
    # instead of the pool vector L-beh-3 trains on, or the order rule is reversed
    _synthetic(tmp_path, h1_aligned=False, beh2_sign=-0.5)
    r, v = A.read(tmp_path, None)
    assert r["build_gate"]["verdict"] == "build L-beh-3", r["build_gate"]
    assert r["d_h1_rule"]["orthogonalize"] is False and r["training_vector"]["kind"] == "D_rb(pool)"
    assert abs(v[0]) > 0.9 and np.linalg.norm(v) == pytest.approx(1.0, abs=0.05)


def test_d_h1_rule_orthogonalizes_an_entangled_vector(tmp_path, fast):
    # most probable failure: the rule keeps plain D_rb when D_h1 is aligned with it, or removes the
    # wrong component (the result must be orthogonal to D_h1, not to D_rb). The rule is strict
    # (Amendment 35 item 5): |cos| must reach D_rb's own split-half q05, so the synthetic D_h1 is
    # the same axis under trace noise wide enough to open the band
    _synthetic(tmp_path, h1_aligned=True, beh2_sign=-0.5, noise=1.0)
    r, v = A.read(tmp_path, None)
    assert r["d_h1_rule"]["orthogonalize"] is True
    base, beh = A.load(tmp_path, "none"), A.load(tmp_path, "beh")
    h1 = A.diffs(base, beh, "h1")
    assert abs(A.unit(A.mean_dir(h1, sorted(h1))) @ v) < 1e-5


def test_gate_reports_when_the_order_fails(tmp_path, fast):
    # most probable failure: a failed positive control still builds L-beh-3
    _synthetic(tmp_path, h1_aligned=False, beh2_sign=2.0)
    r, _ = A.read(tmp_path, None)
    assert not r["build_gate"]["order_ok"] and not r["build_gate"]["build"]
    assert r["build_gate"]["verdict"].startswith("report to the author")


def test_gate_bails_on_noise(tmp_path, fast):
    # most probable failure: a direction indistinguishable from noise is still trained against
    _synthetic(tmp_path, h1_aligned=False, beh2_sign=-0.5, scale=0.0)
    r, _ = A.read(tmp_path, None)
    assert r["build_gate"]["bail"] and r["build_gate"]["verdict"].startswith("bail")


def test_remote_runs_the_rider_after_the_matrix():
    # most probable failure: the rider runs before the matrix (a stop would cost the registered
    # read), or its inputs are not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    blk = sh.split('if [ "$STEPS" = "st2fx" ]; then\n    rc=0')[1][:2200]
    assert blk.index("st2fx_driver.py") < blk.index("st2drb_driver.py --step gen")
    assert blk.index("--step gen") < blk.index("--step prefill")
    assert "for arm in none beh beh2 anchor beh_epoch3; do" in blk
    assert "test_st2drb.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = set((ST / "runpod" / "sync_st2fx.txt").read_text().split())
    assert {"outputs/st2lb/gpt_oss_20b/gen/rr_rollouts.jsonl",
            "outputs/t11/gpt_oss_20b/gen/h1_rollouts.jsonl",
            "outputs/t14/gpt_oss_20b/adapters/anchor/adapter_model.safetensors",
            "outputs/t14/gpt_oss_20b/adapters/beh_epoch3/adapter_model.safetensors"} <= sync
    assert re.search(r'ln -sfn "\$T14_ADAPTERS/beh_epoch3" "\$FX_ADAPTERS/beh_epoch3"', sh)


@pytest.mark.slow
def test_dry_run_every_step(tmp_path):
    # most probable failure: an arm crashes on attach, a zero adapter moves the read, or set (c)'s
    # positions disagree with the bank
    if not (POOL.exists() and NEEDED[0].exists()):
        pytest.skip("the pool and T1.1 rollouts are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2drb_driver.py"), "--out", str(tmp_path), "--dry-run"]
    r = subprocess.run(base + ["--step", "gen"], capture_output=True, text=True, cwd=ROOT, env=env)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for arm in D.ARMS:
        r = subprocess.run(base + ["--step", "prefill", "--arm", arm, "--adapters",
                                   str(tmp_path / "adapters"), "--pool", str(POOL),
                                   "--t11", str(NEEDED[0])],
                           capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = tmp_path / D.KEY / "drb"
    assert {x.split("|")[0] for x in np.load(d / "beh2.npz")["keys"]} == {"spec"}
    res, _ = A.read(d, T11 if (T11 / "bank" / "bank_0000.npz").exists() else None)
    assert res["directions"]["pool"]["norm"] == pytest.approx(0.0, abs=1e-6)  # zero adapters
    assert res["build_gate"]["bail"]
    if "bank_parity" in res:
        assert res["bank_parity"]["rows"] >= 2 and res["bank_parity"]["count_mismatches"] == 0
