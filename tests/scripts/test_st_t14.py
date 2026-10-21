"""ST T1.4 pod driver (scripts/t14_driver.py): Amendment 16's reading rule, and a dry run of every
phase on a tiny GPT-OSS with the real harmony tokenizer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import t14_driver as T  # noqa: E402

T11 = ROOT / "papers/st_live_reference/outputs/t11/gpt_oss_20b"
T13 = ROOT / "papers/st_live_reference/outputs/t13/gpt_oss_20b"


def _arm(passes: bool, dp: float) -> dict:
    return {"passes": passes, "dP": {"mean": dp}}


def test_reading_and_fourth_axis_follow_amendment_16():
    # most probable failure: a care_harm pass is read as care beyond harm while V_harm also
    # passes, or the fourth adapter takes the first passing arm instead of the larger dP
    assert T.gate_reading({"care_harm": _arm(True, 0.2), "V_harm": _arm(True, 0.12)}) == (
        "harm content", "care_harm")
    assert T.gate_reading({"care_harm": _arm(True, 0.11), "V_harm": _arm(False, 0.3)}) == (
        "care beyond harm", "care_harm")
    assert T.gate_reading({"care_harm": _arm(False, 0.0), "V_harm": _arm(True, 0.15)}) == (
        "harm content", "V_harm")
    assert T.gate_reading({"care_harm": _arm(False, 0.2), "V_harm": _arm(False, 0.2)}) == (
        "neither", None)


def test_recipe_is_matched_and_dropout_free():
    # most probable failure: a nonzero LoRA dropout, which PEFT's expert-parameter LoRA refuses,
    # so the first adapter on the pod dies at injection
    assert T.RECIPE["dropout"] == 0.0 and T.RECIPE["rank"] == 16


@pytest.mark.slow
def test_dry_run_every_phase_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: a phase only the pod reaches (gate arms against the reused T1.3
    # rows, the pilot's throwaway adapter, the unload and base check, the conditional fourth)
    # crashes after hours of training, or an adapter is saved without its sidecar
    if not (T11 / "gen" / "h1_rollouts_read.jsonl").exists() or not (
            T13 / "gen" / "gate_fill_rerun.jsonl").exists():
        pytest.skip("T1.1 / T1.3 outputs not present locally")
    draw = json.loads((ROOT / "papers/st_live_reference/data/t13_draw_v1.json").read_text())
    inp = tmp_path / "inputs"
    inp.mkdir()
    (inp / "t14_inputs.json").write_text(json.dumps({
        "gate": draw["gate"], "tau": {"anchor_axis": 0.5, "care_harm": 0.5, "V_harm": 0.5},
        "chat_sigma_provenance": {"dry": True}, "sha256": {}}))
    np.savez(inp / "t14_inputs.npz", anchor_axis=np.zeros(4))
    out = tmp_path / "out"
    cmd = [sys.executable, str(SCRIPTS / "t14_driver.py"), "--out", str(out), "--inputs",
           str(inp), "--t11", str(T11), "--t13", str(T13), "--dry-run"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    d = out / "gpt_oss_20b"
    rec = json.loads((d / "t14_record.json").read_text())
    for name in ("gate_arms", "beh", "anchor", "self", "fourth"):
        assert rec["phases"][name]["finished"]
    for name in ("beh", "anchor", "self"):
        assert rec["phases"][name]["base_unchanged"]
        side = json.loads((d / "adapters" / name / "st_adapter.json").read_text())["sidecar"]
        assert side["objective"] == {"beh": "beh", "anchor": "anchor", "self": "self"}[name]
        assert (d / "adapters" / name / "adapter_model.safetensors").exists()
    assert json.loads((d / "adapters" / "self" / "st_adapter.json").read_text())[
        "sidecar"]["sigma_provenance"] is not None
    # Amendment 17 (a): the epoch-3 snapshot exists and the comparator was chosen on the reserved
    # scenarios (both checkpoints generated), never on H1_held
    sel = rec["phases"]["beh"]["comparator_selection"]
    assert (d / "adapters" / "beh_epoch3" / "adapter_model.safetensors").exists()
    assert set(sel) >= {"step_matched", "epoch3", "comparator"}
    held = set(json.loads((ROOT / "papers/st_live_reference/split_v1.json").read_text())["H1_held"])
    picked = {json.loads(x)["scenario_id"] for x in (d / "gen" / "select_beh_final.jsonl").open()}
    assert picked and not picked & held
