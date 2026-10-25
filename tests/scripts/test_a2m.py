"""ST-A27's discriminators (ST_PLAN Amendment 41): Muse cell A′'s driver
(scripts/a2m_driver.py) and the analysis (scripts/analyze_a2m.py): the band and its fallback,
care-DPO's class, R_b's status, the map, the lexicons, the pair rebuild, the runner, a dry run."""

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
import a2m_driver as D  # noqa: E402
import analyze_a2m as A  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"


def test_band_and_fallback():
    # most probable failure: the fallback band is used although the registered band holds 20, or
    # an arm is read with fewer than 20 items in band
    pv = {f"x{i}": 0.3 for i in range(25)} | {f"y{i}": 0.15 for i in range(30)}
    s = D.select(pv)
    assert s["band"] == [0.2, 0.6] and not s["fallback_used"] and len(s["kept"]) == 25
    pv = {f"x{i}": 0.3 for i in range(5)} | {f"y{i}": 0.15 for i in range(30)}
    s = D.select(pv)
    assert s["fallback_used"] and s["n_in_band"] == 35 and len(s["kept"]) == 35 and s["read"]
    s = D.select({f"x{i}": 0.3 for i in range(5)} | {f"z{i}": 0.02 for i in range(100)})
    assert not s["read"] and s["kept"] == [] and s["n_in_primary_band"] == 5
    big = D.select({f"x{i:03d}": 0.4 for i in range(150)})
    assert len(big["kept"]) == 60 and big["kept"] == D.select({f"x{i:03d}": 0.4
                                                                for i in range(150)})["kept"]


def test_care_class_unresolved_is_not_format():
    # most probable failure: a CI that includes both 0 and care's panel-set drop (-0.054) reads as
    # format, Amendment 36's low-power bias toward R_c
    assert A.care_class([-0.20, -0.01]) == "harm"
    assert A.care_class([-0.04, 0.05]) == "format"
    assert A.care_class([0.02, 0.30]) == "format"
    assert A.care_class([-0.08, 0.03]) == "unresolved"


def _cell(cls, paired_hi, n=40):
    return {"read": True, "care_class": cls, "n_read": n,
            "care_minus_base": {"ci95": [paired_hi - 0.1, paired_hi]}}


def test_map_requires_rb_closed_for_ra():
    # most probable failure: R_a is declared while R_b is still open, or a harm read whose paired
    # contrast includes 0 is credited to the teacher
    cl = {"surface_features": ["length_tokens"]}
    closed, opened = {"status": "closed (x)"}, {"status": "open (y)"}
    assert A.mapping(_cell("harm", -0.02), cl, closed).startswith("R_a")
    assert A.mapping(_cell("harm", -0.02), cl, opened).startswith("unregistered")
    assert A.mapping(_cell("harm", 0.05), cl, closed).startswith("shared")
    assert "length_tokens" in A.mapping(_cell("format", 0.1), cl, opened)
    assert "carrier unnamed" in A.mapping(_cell("format", 0.1), {"surface_features": []}, closed)
    assert A.mapping(_cell("unresolved", 0.1), cl, closed).startswith("unresolved at n = 40")


def test_rb_status_and_scaling():
    # most probable failure: R_b stays open when its premise fails, or k is taken from the wrong
    # arm order (base / care), so a scaled base reads as inconsistent
    assert A.rb_status({"likelihood_premise": {"holds": False}}, None)["status"].startswith(
        "closed")
    rng = np.random.default_rng(1)
    gb = {f"s{i}": 0.8 + 0.1 * rng.standard_normal() for i in range(50)}
    gen = {"base": gb, "care": {s: 0.4 * v for s, v in gb.items()}}
    base = {f"i{i}": 0.2 + 0.05 * rng.standard_normal() for i in range(60)}
    ok = A.scaling({"base": base, "care": {i: 0.4 * v for i, v in base.items()}}, gen)
    assert ok["k"] == pytest.approx(0.4) and ok["consistent_with_scaling"]
    bad = A.scaling({"base": base, "care": {i: -0.1 for i in base}}, gen)
    assert not bad["consistent_with_scaling"]
    assert A.rb_status({"likelihood_premise": {"holds": True}}, bad)["status"].startswith("closed")
    assert A.rb_status({"likelihood_premise": {"holds": True}}, ok)["status"].startswith("open")


def test_lexicons_whole_words():
    # most probable failure: substrings count ("harmless", "butter"), or the phrase and the hyphen
    # entries never match
    f = A.features("Harmless butter. On the other hand, the trade-off costs her; HARM.", 10)
    words = 11
    assert f["harm"] == pytest.approx(100 * 2 / words)  # costs, HARM
    assert f["weighing"] == pytest.approx(100 * 2 / words)  # on the other hand, trade-off
    assert f["person"] == pytest.approx(100 * 1 / words)  # her
    assert f["length_tokens"] == 10


def test_scenario_bootstrap_is_pair_weighted():
    # most probable failure: scenario means are averaged unweighted, which is not the pairs' mean
    r = A.scenario_boot({"a": [1.0, 1.0, 1.0], "b": [0.0]}, (2.5, 97.5))
    assert r["mean"] == pytest.approx(0.75) and r["n_pairs"] == 4 and r["n_scenarios"] == 2


def _rows(arm, sid, statuses):
    return [{"scenario_id": sid, "rollout": str(i), "norm_status": st, "gen_ids": [i + 1, 7],
             "trace_nll_unsteered": 0.5 + 0.1 * i, "arm": arm} for i, st in enumerate(statuses)]


def test_pairs_rebuild_checked_against_t16(tmp_path):
    # most probable failure: the rebuilt pairs differ from T1.6's (another pairing rule) and the
    # content read runs on traces the adapters never saw
    rep, t16 = tmp_path / "rep", tmp_path / "t16"
    (rep / "gen").mkdir(parents=True)
    care = _rows("care", "S1", ["violating", "violating", "consistent"])
    none = _rows("none", "S1", ["violating", "consistent", "consistent", "violating"])
    for name, rows in (("arm_care_03", care), ("arm_none", none)):
        (rep / "gen" / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    for arm in ("care", "base"):
        (t16 / "adapters" / arm).mkdir(parents=True)
        (t16 / "adapters" / arm / "st_adapter.json").write_text(json.dumps(
            {"pairs": {"per_scenario": {"S1": 2}}}))
    pr, info = A.pairs(t16, rep)
    assert len(pr) == 2 and all(c["arm"] == "care" and b["arm"] == "none" for c, b in pr)
    assert [b["rollout"] for _, b in pr] == ["0", "3"]
    (t16 / "adapters" / "base" / "st_adapter.json").write_text(json.dumps(
        {"pairs": {"per_scenario": {"S1": 1}}}))
    with pytest.raises(SystemExit):
        A.pairs(t16, rep)


def test_manifest_matches_returned_adapters():
    # most probable failure: the manifest names other files than the ones the T1.6 pod returned
    m = json.loads(D.MANIFEST.read_text())["arms"]
    assert set(m) == {"care", "base"}
    ad = ST / "outputs" / "t16" / "muse_glimmer_30b" / "adapters"
    if not (ad / "care" / "adapter_model.safetensors").exists():
        pytest.skip("the T1.6 adapters are not local")
    for arm in m:
        assert D.verify(ad, arm) == m[arm]


def test_remote_runs_a2m():
    # most probable failure: the a2m block reads vLLM or another model, the arms run before the
    # screen, or the adapters are not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "a2m" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-muse_glimmer_30b'
                     r'\}"; OUT_SUB=a2m\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-1\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "a2m" ]; then\n    rc=0')[1][:700]
    assert blk.index("--step screen") < blk.index("--step arm")
    assert "vllm" not in blk.lower().replace("vllm_venv", "")
    assert "test_a2m.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    assert '[ "$STEPS" != "a2m" ]' in sh.split("# re-priced locally")[0].split("\n")[-1]
    sync = (ST / "runpod" / "sync_a2m.txt").read_text().split()
    for arm in ("care", "base"):
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            assert f"outputs/t16/muse_glimmer_30b/adapters/{arm}/{f}" in sync


@pytest.mark.slow
def test_dry_run_every_step(tmp_path):
    # most probable failure: an arm step crashes on a tiny Muse (attach check, adapter layout) or
    # the analysis cannot read the arms' rows
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "a2m_driver.py"), "--out", str(tmp_path), "--dry-run"]
    for extra in (["--step", "screen"], ["--step", "arm", "--arm", "care", "--adapters",
                                         str(tmp_path / "adapters")],
                  ["--step", "arm", "--arm", "base", "--adapters", str(tmp_path / "adapters")]):
        r = subprocess.run(base + extra, capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    out = tmp_path / D.KEY
    rec = json.loads((out / "a2m_record.json").read_text())["phases"]
    assert {"screen", "arm_care", "arm_base"} <= set(rec)
    c = A.cell(out)
    assert c["read"] and c["care_class"] in ("harm", "format", "unresolved")
    # zero-initialized adapters read as the base model
    assert c["care"]["dP_violating"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_generation_branch_boundaries():
    # most probable failure (Amendment 43 item 3): a CI that reaches the floor's +0.14 reads as no
    # rise, or a rise whose lower bound sits inside the specific band reads as the floor
    assert A.gen_branch([0.05, 0.20], 0.12).startswith("rise")
    assert A.gen_branch([-0.04, 0.10], 0.03).startswith("no rise (specific")
    assert A.gen_branch([-0.15, -0.02], -0.08).startswith("no rise (drop")
    assert A.gen_branch([0.01, 0.16], 0.08).startswith("unresolved")
    assert A.gen_branch([0.02, 0.13], 0.075).startswith("no rise (non-specific")


def test_fork_sign_classes():
    # most probable failure: the fork reads the registered class instead of Amendment 31's
    c = {"care": {"dP_violating": {"mean": 0.027}}, "base": {"dP_violating": {"mean": 0.517}},
         "care_minus_base": {"mean": -0.49}}
    f = A.fork_sign_classes(c)
    assert f["care"]["class"] == "specific" and f["base"]["class"] == "non-specific"
    assert f["care_minus_base"]["class"] == "discriminating"


def _gen_rows(arm, viol):
    return [{"scenario_id": sid, "norm_status": "violating" if k < v else "consistent",
             "status": "completed", "arm": arm}
            for sid, v in viol.items() for k in range(8)]


def test_generation_read_skips_a_failed_parity_arm(tmp_path):
    # most probable failure: rows of an arm whose parity failed enter the read
    (tmp_path / "gen").mkdir()
    kept = [f"i{j}" for j in range(30)]
    for arm, v in (("none", 2), ("care", 2), ("base", 6)):
        (tmp_path / "gen" / f"cellA2_{arm}.jsonl").write_text("\n".join(
            json.dumps(r) for r in _gen_rows(arm, {i: v for i in kept})))
    rec = {"phases": {"gen_none": {"base_tv": 0.02, "base_check_ok": True},
                      "gen_care": {"parity_ok": True}, "gen_base": {"parity_ok": False}}}
    (tmp_path / "a2g_record.json").write_text(json.dumps(rec))
    g = A.gen_read(tmp_path, kept)
    assert g["care"]["dP_violating"]["mean"] == pytest.approx(0.0)
    assert g["base"] == {"read": False} and "care_minus_base" not in g
    assert g["care_branch"].startswith("no rise (specific")


def test_remote_runs_a2g():
    # most probable failure: an arm generates before the base check, a merged copy is not removed,
    # or T1.6's saved letters and the selection are not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "a2g" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-muse_glimmer_30b'
                     r'\}"; OUT_SUB=a2g\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-2\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "a2g" ]; then\n    A2G=')[1][:1100]
    steps = ("--step gen --arm none", '--step merge --arm "$arm"', '--step gen --arm "$arm"')
    order = [blk.index(x) for x in steps]
    assert order == sorted(order)
    assert "test_a2m.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    assert '[ "$STEPS" != "a2g" ]' in sh.split("# re-priced locally")[0].split("\n")[-1]
    sync = set((ST / "runpod" / "sync_a2g.txt").read_text().split())
    need = {"outputs/a2m/muse_glimmer_30b/cellA2_selection.json",
            "outputs/st2mr/muse_glimmer_30b/gen/arm_none.jsonl",
            "outputs/t16/muse_glimmer_30b/t16_record.json"}
    need |= {f"outputs/t16/muse_glimmer_30b/parity/hf_{a}.npy" for a in ("none", "care", "base")}
    need |= {f"outputs/t16/muse_glimmer_30b/adapters/{a}/{f}" for a in ("care", "base")
             for f in ("adapter_config.json", "adapter_model.safetensors")}
    assert need <= sync


@pytest.mark.slow
def test_a2g_dry_run_every_step(tmp_path):
    # most probable failure: a merge or a merged engine's step crashes on a tiny Muse
    rep = ST / "outputs" / "st2mr" / "muse_glimmer_30b"
    sel = ST / "outputs" / "a2m" / "muse_glimmer_30b" / "cellA2_selection.json"
    if not ((rep / "gen" / "arm_none.jsonl").exists() and sel.exists()):
        pytest.skip("the replicate's rows or the a2m selection are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "a2g_driver.py"), "--out", str(tmp_path), "--replicate",
            str(rep), "--t16", str(tmp_path / "no_t16"), "--selection", str(sel), "--adapters",
            str(tmp_path / "adapters"), "--dry-run"]
    for st in (["gen", "none"], ["merge", "care"], ["gen", "care"], ["merge", "base"],
               ["gen", "base"]):
        r = subprocess.run(base + ["--step", st[0], "--arm", st[1]], capture_output=True,
                           text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads((tmp_path / D.KEY / "a2g_record.json").read_text())["phases"]
    assert all(rec[f"gen_{a}"].get("status") == "ok" for a in ("none", "care", "base"))
    assert rec["merge_care"]["ok"] and rec["merge_base"]["ok"]
