"""The smoke pod for pod 2 on Qwen3.8-27B (ST_PLAN Amendment 57 and its addition;
scripts/st2qs_driver.py): the pair rule, the configuration of t16_driver, the runner and a dry run.
"""

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
import st2qs_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
POD1 = ST / "outputs" / "st2q" / "qwen38_27b"


def _r(sid, k, st):
    return {"scenario_id": sid, "rollout": k, "norm_status": st, "gen_ids": [1, 2, k]}


def test_smoke_pairs_follow_t16s_rule_without_the_base_term():
    # most probable failure: a care protect trace is paired with another scenario's rule trace, or
    # more than two pairs come from one scenario
    care = [_r("a", k, "violating") for k in range(4)] + [_r("b", 0, "violating")]
    none = [_r("a", k, "consistent") for k in range(3)] + [_r("c", 0, "consistent")]
    pr, info = D.smoke_pairs(care, none)
    assert info["per_scenario"] == {"a": 2} and len(pr) == 2
    assert all(p["chosen_key"][0] == p["rejected_key"][0] == "a" for p in pr)
    assert [p["chosen_key"][1] for p in pr] == [0, 1]


def test_configure_points_t16_at_qwen():
    # most probable failure: t16_driver keeps Muse's key, pin or site, so the smoke pod renders
    # Muse's prompts or loads Muse
    import t16_driver as T16

    D.configure()
    assert (T16.KEY, T16.PIN, T16.BLOCK, T16.STAGE) == ("qwen38_27b", "2026-10-27", 32, "st2qs")
    assert T16.pairs_for is D.pairs_for and T16.parity_scenarios is D.parity_scenarios


def test_runner_and_sync():
    # most probable failure: a step runs under the wrong interpreter (HF steps need the vLLM venv's
    # torch), or pod 1's rows are not synced
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "st2qs" \]; then\n(?:  #.*\n)*  MODELS="\$\{MODELS:-qwen38_27b'
                     r'\}"; OUT_SUB=st2qs\n  SESSION_MAX_HOURS="\$\{SESSION_MAX_HOURS:-2\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "st2qs" ]; then\n    QS=')[1].split("continue")[0]
    order = [blk.index(f"--step {s}") for s in ("base_vllm", "base_hf", "train", "hf_arm",
                                                "vllm_arm")]
    assert order == sorted(order)
    assert '$VPY $QS --step train' in blk and '$PY $QS --step vllm_arm' in blk
    assert "test_st2qs.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_st2qs.txt").read_text().split()
    assert {"outputs/st2q/qwen38_27b/gen/arm_care.jsonl",
            "outputs/st2q/qwen38_27b/gen/arm_none.jsonl"} <= set(sync)


@pytest.mark.slow
def test_dry_run(tmp_path):
    # most probable failure: a step breaks on Qwen's DeltaNet text tower (LoRA targets, the merge's
    # key matching) on a tiny Qwen, or the adapter reaches the vision tower
    if not (POD1 / "gen" / "arm_care.jsonl").exists():
        pytest.skip("pod 1's rows are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "st2qs_driver.py"), "--out", str(tmp_path), "--pod1",
            str(POD1), "--dry-run"]
    for st in (["base_vllm"], ["base_hf"], ["train", "--arm", "care"], ["hf_arm", "--arm", "care"],
               ["vllm_arm", "--arm", "care"]):
        r = subprocess.run(base + ["--step"] + st, capture_output=True, text=True, cwd=ROOT,
                           env=env)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    rec = json.loads((tmp_path / D.KEY / "st2qs_record.json").read_text())["phases"]
    assert rec["hf_arm_care"]["merge"]["saved"]["ok"] and rec["vllm_arm_care"]["parity_ok"]
    cfg = json.loads((tmp_path / D.KEY / "adapters" / "care" / "adapter_config.json").read_text())
    tm = set(cfg["target_modules"])
    assert {"in_proj_qkv", "out_proj", "q_proj", "down_proj"} <= tm
    assert not tm & {"qkv", "proj", "linear_fc1", "linear_fc2"}  # the vision tower's names
