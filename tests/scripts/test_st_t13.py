"""ST T1.3 pod driver (scripts/t13_driver.py): ΔP against the in-gate none, and a dry run of every
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
import t13_driver as T  # noqa: E402


def test_delta_p_reads_the_in_gate_none_and_keeps_defer_in_the_denominator():
    # most probable failure: ΔP is taken against T1.1's selecting sample, or defer rollouts drop
    # out of P's denominator (inflating the protective share)
    none = [{"scenario_id": "S", "norm_status": x} for x in ("violating", "consistent",
                                                              "neutral", None)]
    arm = [{"scenario_id": "S", "norm_status": x} for x in ("violating", "violating",
                                                             "neutral", "consistent")]
    d = T.delta_p(arm, none)
    assert abs(d["P_none"] - 1 / 3) < 1e-12 and abs(d["P_arm"] - 0.5) < 1e-12
    assert abs(d["mean"] - (0.5 - 1 / 3)) < 1e-12


@pytest.mark.slow
def test_dry_run_every_phase_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: a phase that only a pod would reach (the per-row vectors, the self
    # source index, the transport readout, the panel call) crashes after the gate has spent its
    # hours; and arms that do not share seeds by (scenario, rollout)
    pytest.importorskip("transformers")
    draw = json.loads((ROOT / "papers/st_live_reference/data/t13_draw_v1.json").read_text())
    inp = tmp_path / "inputs"
    inp.mkdir()
    (inp / "t13_inputs.json").write_text(json.dumps({
        "class": "onset", "alphas": [1.0, 2.0, 4.0, 8.0], "gate": draw["gate"],
        "sweep": draw["sweep"]}))
    np.savez(inp / "t13_inputs.npz", axis=np.zeros(4), fill_unit=np.zeros((5, 4)),
             self_ids=np.array(draw["gate"] + draw["sweep"]), self_src=np.zeros((25, 4)))
    r = subprocess.run([sys.executable, str(SCRIPTS / "t13_driver.py"), "--out",
                        str(tmp_path / "out"), "--inputs", str(inp), "--dry-run",
                        "--date-pin", "2026-10-19"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    d = tmp_path / "out" / "gpt_oss_20b"
    rec = json.loads((d / "t13_record.json").read_text())
    assert all(rec["phases"][p].get("finished") for p in ("sweep", "gate", "transport", "panel"))
    assert set(rec["phases"]["gate"]["arms"]) == {"fill", "care", "self_0.25", "self_0.5"}
    rows = {a: [json.loads(x) for x in (d / "gen" / f"gate_{a}.jsonl").open()]
            for a in ("none", "care", "self_0.25")}
    keys = {a: [x["rollout_key"] for x in v] for a, v in rows.items()}
    assert keys["none"] == keys["care"] == keys["self_0.25"]
    assert all(x["n_steered"] == 0 for x in rows["none"])
    steered = rows["care"] + rows["self_0.25"]
    assert all(x["n_steered"] > 0 for x in steered if x["n_gen_tokens"] > 1)  # hooks fired
    assert (d / "panel_call.json").exists()
