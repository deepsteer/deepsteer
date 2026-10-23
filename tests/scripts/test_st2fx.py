"""The format x harm matrix (scripts/build_fxh_cells.py, scripts/st2fx_driver.py,
scripts/analyze_st2fx.py; Amendment 33): the two cells as built, their distance from every trained
or read item, the adapter manifest, the branches, the runner's order, and a dry run of every arm on
a tiny GPT-OSS."""

from __future__ import annotations

import glob
import hashlib
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
import analyze_st2fx as A  # noqa: E402
import build_fxh_cells as B  # noqa: E402
import st2fx_driver as D  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir, load_scenarios  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
DATA = ST / "data"
KDG = ROOT / "papers" / "kdg_panel" / "data"


def _cells() -> dict:
    return {k: load_scenarios(p)[0] for k, p in D.CELLS.items()}


def test_cells_as_registered():
    # most probable failure (Amendment 33): a cell's right option is the wrong status (cell A must
    # reward following the rule, cell B departing from it), so the read's sign means the opposite
    c = _cells()
    for cell, right, fam in (("A", "consistent", "FXA"), ("B", "violating", "FXB")):
        sc = c[cell]
        assert len(sc) == 60 and len({s.id for s in sc}) == 60
        assert all(s.family == fam and s.id.startswith(fam + "-") for s in sc)
        assert all({o.norm_status: o.option_id for o in s.options}[right]
                   == s.external_label["right_option_id"] for s in sc)
        assert all(sorted(o.norm_status for o in s.options) == ["consistent", "neutral",
                                                                 "violating"] for s in sc)
        assert all("after L-beh and L-beh-2 trained" in s.covariates["written"] for s in sc)
        assert len({s.agent_name for s in sc}) == 60


def test_cells_are_far_from_every_trained_or_read_item():
    # most probable failure: a written item near-copies an H1 or panel scenario (a training pair's
    # prompt, a pool, specificity or guard item), so the matrix reads a trained item
    H1, _ = load_scenario_dir([KDG / "harm_divergence" / "hdiv_set.json"])
    panel, _ = load_scenario_dir(sorted(glob.glob(str(KDG / "*_scenarios_*.json"))))
    src = [B.words(s.agent_text) for s in list(H1) + list(panel)]
    assert len(src) > 300
    src_ids = {s.id for s in list(H1) + list(panel)}
    for sc in _cells().values():
        assert not {s.id for s in sc} & src_ids
        worst = max(B.jaccard(B.words(s.agent_text), w) for s in sc for w in src)
        assert worst < 0.5, worst


def test_handcheck_recorded():
    # most probable failure: the hand-check sample is not the seeded one, or fewer than 20 per cell
    rec = json.loads((DATA / "st_fxh_build_v1.json").read_text())
    import random
    for c in "AB":
        want = [f"FX{c}-{i:03d}" for i in sorted(
            random.Random(f"20261023|handcheck|{c}").sample(range(1, 61), 20))]
        assert rec["handcheck"][c]["ids"] == want and rec["handcheck"][c]["pass"] == 20
        assert rec[c]["kept"] == 60 and rec[c]["seed"] == B.SEED


def test_rider_is_the_engaged_held_set():
    # most probable failure: the rider reads H1_train (trained) instead of the 38 engaged H1_held
    eng = json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"]
    assert len(eng) == 38
    c = D.cells(dry=False)
    assert [s.id for s in c["H1"]] == eng and len(c["A"]) == len(c["B"]) == 60


def test_adapter_manifest():
    # most probable failure: the matrix reads a different L-beh than T1.5 evaluated, or a stale
    # L-beh-2 adapter
    m = json.loads(D.MANIFEST.read_text())["arms"]
    t15 = json.loads((DATA / "t15_adapters_v1.json").read_text())["arms"]
    assert m["beh"] == t15["beh"] and set(m) == {"beh", "beh2"}
    for arm, d in (("beh", ST / "outputs" / "t14" / "gpt_oss_20b" / "adapters" / "beh"),
                   ("beh2", ST / "outputs" / "st2lb" / "gpt_oss_20b" / "adapters" / "beh2")):
        for f, sha in m[arm].items():
            if (d / f).exists():
                assert hashlib.sha256((d / f).read_bytes()).hexdigest() == sha, (arm, f)


def _rows(path: Path, items: list[str], p_viol: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for sid in items:
            for k in range(8):
                pv = p_viol[sid]
                lp = {"A": float(np.log(pv)), "B": float(np.log((1 - pv) / 2)),
                      "C": float(np.log((1 - pv) / 2))}
                fh.write(json.dumps({"scenario_id": sid, "rollout": k, "option_logps": lp,
                                     "norm_status_by_letter": {"A": "violating", "B": "consistent",
                                                               "C": "neutral"}}) + "\n")


@pytest.mark.parametrize("da,db,want", [(-0.1, +0.1, "R_a"), (+0.1, -0.1, "R_b"),
                                        (0.0, 0.0, "R_b"), (-0.1, 0.0, "split"),
                                        (+0.1, +0.1, "split")])
def test_branches(tmp_path, da, db, want):
    # most probable failure: the mirror cell is read with cell A's sign (a rise on rule-harm items
    # in the panel's format is protection, the harm reading; a drop there is not)
    rng = np.random.default_rng(0)
    items = [f"X{i}" for i in range(30)]
    base = {i: float(rng.uniform(0.3, 0.5)) for i in items}
    for cell, d in (("A", da), ("B", db), ("H1", 0.2)):
        _rows(tmp_path / "fx" / "none" / cell / "dl_chat_neutral.jsonl", items, base)
        for arm in ("beh2", "beh"):
            sh = {i: base[i] + d + float(rng.normal(0, 0.005)) for i in items}
            _rows(tmp_path / "fx" / arm / cell / "dl_chat_neutral.jsonl", items, sh)
    r = A.read(tmp_path)
    assert r["branch"].startswith(want), r["branch"]
    assert r["cells"]["H1"]["beh2"]["move"] == "rise"
    assert r["panel_diagonal_of_record"]["beh2"]["mean"] == pytest.approx(-0.093, abs=1e-3)


def test_remote_runs_the_matrix():
    # most probable failure: the runner reads an arm before none (no baseline if the stop lands),
    # or the pod syncs an adapter directory instead of its two files
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2fx" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-1\.5\}"', sh)
    assert sh.count("for arm in none beh2 beh; do") == 2
    assert 'ln -sfn "$T14_ADAPTERS/beh" "$FX_ADAPTERS/beh"' in sh
    assert "test_st2fx.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_st2fx.txt").read_text().split()
    assert len(sync) == 4 and all(not x.endswith("/") for x in sync)
    assert {Path(x).name for x in sync} == {"adapter_config.json", "adapter_model.safetensors"}


@pytest.mark.slow
def test_dry_run_every_arm(tmp_path):
    # most probable failure: an arm crashes on attach, or a zero adapter moves the read (the
    # attached path differs from none's)
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for arm in D.ARMS:
        r = subprocess.run([sys.executable, str(SCRIPTS / "st2fx_driver.py"), "--arm", arm,
                            "--out", str(tmp_path), "--adapters", str(tmp_path / "adapters"),
                            "--dry-run"], capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    d = tmp_path / D.KEY
    rec = json.loads((d / "st2fx_record.json").read_text())["arms"]
    assert set(rec) == set(D.ARMS) and all(v["finished"] for v in rec.values())
    r = A.read(d)
    for cell in A.CELLS:
        for arm in A.ARMS:
            assert r["cells"][cell][arm]["dP_violating"]["mean"] == pytest.approx(0.0, abs=1e-6)
