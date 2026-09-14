"""KDG pilot driver (papers/kdg_panel/scripts/pod_kdg_pilot.py): dry-run end-to-end on the stub
model, manifest integrity, per-rollout artifact contract, and the zero-GPU analysis path. No
model is loaded anywhere in this file."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "papers" / "kdg_panel" / "scripts"
sys.path.insert(0, str(SCRIPTS))

from kdg_pod_lib import PILOT_UNITS_BASE, PILOT_UNITS_INSTRUCT, verify_manifest  # noqa: E402

from deepsteer.kdg.schema import save_scenarios  # noqa: E402
from tests.kdg.conftest import make_scenario  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pod = _load("pod_kdg_pilot")
analyze = _load("analyze_pilot")


@pytest.fixture(scope="module")
def scenario_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("kdg_scen")
    scen = []
    for fam in ("F1", "F2", "F3", "F4", "F5"):
        for k in range(3):
            s = make_scenario(fam, id=f"{fam}-A-{k:02d}")
            scen.append(s)
        tw = make_scenario(fam, id=f"{fam}-A-00T", role="harm_twin", twin_of=f"{fam}-A-00")
        if fam not in ("F2", "F5"):
            scen.append(tw)
    p = d / "pilot_scenarios_A_test.json"
    save_scenarios(p, scen, {"generator": "test", "generator_half": "A"})
    return p


@pytest.fixture(scope="module")
def dry_run(tmp_path_factory, scenario_file):
    out = tmp_path_factory.mktemp("kdg_dry")
    manifest = pod.run(out, dry=True, models=None, units=None, scenario_files=[scenario_file])
    return out, json.loads(Path(manifest).read_text())


class TestDryRun:
    def test_every_pilot_unit_ran_ok(self, dry_run):
        # assert no cell raised on the stub (the "known to run end-to-end" gate before any pod)
        _, m = dry_run
        failed = {k: v for k, v in m["unit_status"].items() if v["status"] != "ok"}
        assert not failed, failed
        for u in PILOT_UNITS_INSTRUCT:
            assert m["unit_status"][f"olmo3_instruct/{u}"]["status"] == "ok"
        for u in PILOT_UNITS_BASE:
            assert m["unit_status"][f"olmo3_base/{u}"]["status"] == "ok"

    def test_base_never_gets_chat_cells(self, dry_run):
        # assert the base model runs raw cells only (spec §4.6: no generation parsed on base)
        _, m = dry_run
        assert not any(
            k.startswith("olmo3_base/d_chat") or k.startswith("olmo3_base/j_stated")
            for k in m["unit_status"]
        )

    def test_order_instruct_then_base(self, dry_run):
        _, m = dry_run
        assert [ld["key"] for ld in m["loads"]] == ["olmo3_instruct", "olmo3_base"]

    def test_manifest_hashes_verify(self, dry_run):
        out, m = dry_run
        assert m["artifacts"] and verify_manifest(out / "manifest_kdg.json") == []

    def test_per_rollout_artifacts_carry_the_contract(self, dry_run):
        # assert every D_chat rollout saves text, parsed option, harness label, order, and a full
        # decision log-prob row (spec §9), and the option-token ids index that row
        out, _ = dry_run
        rows = [
            json.loads(line)
            for line in (out / "olmo3_instruct" / "d_chat_dose0.jsonl").read_text().splitlines()
        ]
        z = np.load(out / "olmo3_instruct" / "d_chat_dose0.npz")
        assert len(rows) == z["logp_decision"].shape[0] == z["option_token_ids"].shape[0]
        for r in rows:
            assert {
                "text",
                "option_id",
                "norm_status",
                "order",
                "parse_method",
                "harness_version",
                "template_version",
                "prompt_sha256",
            } <= set(r)
        # assert option order actually varies across rollouts of one scenario
        orders = {
            json.dumps(r["order"], sort_keys=True)
            for r in rows
            if r["scenario_id"] == rows[0]["scenario_id"]
        }
        assert (
            len(orders) > 1
            or len([r for r in rows if r["scenario_id"] == rows[0]["scenario_id"]]) < 2
        )

    def test_f2_rows_keep_turn1_text(self, dry_run):
        out, _ = dry_run
        rows = [
            json.loads(line)
            for line in (out / "olmo3_instruct" / "d_chat_dose0.jsonl").read_text().splitlines()
        ]
        f2 = [r for r in rows if r["family"] == "F2"]
        assert f2 and all(r["turn1_text"] is not None for r in f2)

    def test_raw_cells_save_option_mass(self, dry_run):
        out, _ = dry_run
        z = np.load(out / "olmo3_base" / "d_raw.npz")
        # assert the base floor is recomputable: option mass and the full vector are both saved
        assert "option_mass" in z and z["logp_decision"].shape[1] > 4


class TestAnalysis:
    def test_analysis_runs_on_dry_outputs(self, dry_run, tmp_path):
        out, _ = dry_run
        rep = analyze.analyze(out, n_boot=50)
        # assert the gate report exists and the dry-run (random letters) does not pass the pilot
        # gate by construction being trusted: the report must say dry_run and carry the ladder
        assert rep["dry_run"] is True
        assert set(rep["ladder"]) >= {"floor", "matched_null", "measurement", "positive_band"}
        assert "gate" in rep and "n_pass" in rep["gate"]
        assert rep["base_cell"]["floor"] == 0.5


def test_scenario_id_filter(scenario_file, tmp_path):
    out = tmp_path / "sub"
    pod.run(
        out,
        dry=True,
        models=["olmo3_instruct"],
        units=["d_chat_dose0"],
        scenario_files=[scenario_file],
        scenario_ids={"F1-A-00"},
    )
    rows = [
        json.loads(line)
        for line in (out / "olmo3_instruct" / "d_chat_dose0.jsonl").read_text().splitlines()
    ]
    # assert the screened-id filter restricts the dose-arm path to the listed scenarios only
    assert {r["scenario_id"] for r in rows} == {"F1-A-00"}
