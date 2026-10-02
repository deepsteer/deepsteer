"""KDG Phase 2 harness wiring (papers/KDG_F6_F8_SPEC.md §8 G6/G7, §10): Phase 2 units, the
extended rendered-prompt identity check, and a dry run over F6–F8 items plus the turns-since-norm
rider. Each test names the failure mode it guards (CLAUDE.md local-test rule). No model loads."""

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

import kdg_pod_lib as lib  # noqa: E402

from deepsteer.kdg.phase2 import expand, save_items  # noqa: E402
from deepsteer.kdg.schema import save_scenarios  # noqa: E402
from tests.kdg.conftest import make_scenario  # noqa: E402
from tests.kdg.test_phase2 import make_item  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pod = _load("pod_kdg_phase1")


@pytest.fixture(scope="module")
def p1_file(tmp_path_factory):
    scen = [make_scenario(f, id=f"{f}-A-{k:02d}") for f in ("F1", "F3", "F5") for k in range(2)]
    p = tmp_path_factory.mktemp("p1") / "pilot_scenarios_A_test.json"
    save_scenarios(p, scen, {"generator": "test"})
    return p


@pytest.fixture(scope="module")
def item_file(tmp_path_factory):
    items = [make_item(f, low=True) for f in ("F6", "F7", "F8")]
    p = tmp_path_factory.mktemp("p2") / "p2pilot_items_A_test.json"
    save_items(p, items, {"generator": "test"})
    return p


def test_extended_identity_flags_a_middle_assistant_turn_drift(p1_file):
    # most probable failure (G7): the stage identity check covers single user turns only, so a
    # template that renders a prefilled assistant turn differently passes as "no fork"
    from deepsteer.kdg.schema import load_scenarios

    scen, _ = load_scenarios(p1_file)
    m = lib.StubModel("instruct")

    def drifted(msgs):
        msgs = [dict(x) for x in msgs]
        asst = [i for i, x in enumerate(msgs) if x["role"] == "assistant"]
        if len(asst) >= 3:
            msgs[asst[len(asst) // 2]]["content"] += "."  # one character, middle assistant turn
        return m.render_chat(msgs)

    assert lib.rendered_identity_mismatches(m, drifted, scen) == []  # legacy check is blind
    bad = lib.rendered_identity_mismatches(m, drifted, scen, units=list(lib.TSN_UNITS))
    assert bad and all(b.split("/")[0] in lib.TSN_UNITS for b in bad)
    assert lib.rendered_identity_mismatches(m, m.render_chat, scen, units=list(lib.TSN_UNITS)) == []


def test_identity_check_covers_system_turns_and_fails_loud_on_unbuilt_units(item_file):
    # most probable failure: a scheduled chat unit with no message builder is silently skipped
    from deepsteer.kdg.phase2 import load_expanded

    scen, _ = load_expanded([item_file])
    m = lib.StubModel("instruct")

    def sys_drift(msgs):
        return m.render_chat(
            [dict(x, content=x["content"] + " ") if x["role"] == "system" else x for x in msgs]
        )

    bad = lib.rendered_identity_mismatches(m, sys_drift, scen, units=["dl_chat_known_gap"])
    assert bad and all(b.startswith("dl_chat_known_gap/") for b in bad)
    with pytest.raises(KeyError):
        lib.rendered_identity_mismatches(m, m.render_chat, scen, units=["d_chat_dose0"])


def test_phase2_dry_run_routes_units_to_their_scenario_sets(tmp_path, p1_file, item_file):
    # most probable failure: the turns-since-norm rider runs on the F6–F8 items (or the pilot
    # units on the Phase 1 set), or F8's fifth letter is dropped from the readout
    out = tmp_path / "dry"
    p = pod.run(
        out,
        True,
        ["olmo3_instruct", "olmo3_sft"],
        ["P2PILOT", "TSN", "VALIDATE"],
        [p1_file],
        None,
        [item_file],
    )
    man = json.loads(Path(p).read_text())
    st = man["unit_status"]
    assert all(v["status"] == "ok" for v in st.values()), st
    sft = [ld for ld in man["loads"] if ld["key"] == "olmo3_sft"][0]
    assert sft["rendered_identity"]["mismatches"] == []
    assert set(sft["rendered_identity"]["units"]) >= set(lib.TSN_UNITS) | {"dl_chat_known_gap"}
    d = out / "olmo3_instruct"
    rows = [json.loads(x) for x in (d / "dl_chat_neutral.jsonl").read_text().splitlines()]
    assert {r["family"] for r in rows} == {"F6", "F7", "F8"}
    assert {r["level"] for r in rows if r["family"] == "F6"} == {
        "none",
        "nospk",
        "peer",
        "principal",
    }
    f8 = [r for r in rows if r["family"] == "F8"]
    assert f8 and all(len(r["option_logps"]) == 5 for r in f8)
    assert np.load(d / "dl_chat_neutral.npz")["option_token_ids"].shape[1] == 5
    tsn = [json.loads(x) for x in (d / "dl_tsn_reminder_k6.jsonl").read_text().splitlines()]
    assert tsn and {r["family"] for r in tsn} <= {"F1", "F3", "F5"}
    assert (d / "jl_chat_neutral_pressure_removed_p2.npz").exists()
    low = [r for r in rows if r["nudge"] == "low"]
    assert {r["level"] for r in low} == {"none", "ai_collective", "penalty"}
    assert len(expand(make_item("F6", low=True))) == 5


def test_pilot_gate_and_tsn_analyses_read_what_the_driver_writes(tmp_path, p1_file):
    # most probable failure: the gate reads a cell or level name the driver never writes, so every
    # family silently reads zero items (or TSN pairs reminder and neutral cells from different k)
    items = [make_item(f, low=True) for f in ("F6", "F7", "F8")]
    for it in items:
        it.external_label = {"agrees_with_construction": True, "principal_waivable": False}
    bad = make_item("F6")
    bad.id = "F6-B-01"
    bad.external_label = {"agrees_with_construction": True, "principal_waivable": True}
    items.append(bad)
    ip = tmp_path / "p2pilot_items_A_test.json"
    save_items(ip, items, {"generator": "test"})
    out = tmp_path / "p2a"
    pod.run(out, True, ["olmo3_instruct"], ["P2PILOT", "TSN", "VALIDATE"], [p1_file], None, [ip])
    ana = _load("analyze_p2_pilot")
    ana.FLOOR = 0.0  # stub masses are random over a 64-token vocab
    rep_path = tmp_path / "gate.json"
    ana.main(
        ["--out", str(tmp_path / "nope"), str(out), "--items", str(ip), "--report", str(rep_path)]
    )
    rep = json.loads(rep_path.read_text())
    assert rep["excluded"]["principal_waivable"] == ["F6-B-01"]
    fams = rep["models"]["olmo3_instruct"]["families"]
    for fam in ("F6", "F7", "F8"):
        assert fams[fam]["G1_engaged"] == 1 and fams[fam]["G4_dD"]["n"] == 1, fam
        assert fams[fam]["G3_g_band"]["n"] == 1 and fams[fam]["G5_low_nudge_ratio"] is not None
    assert set(fams["F6"]["F6_levels_pD"]) == {"none", "nospk", "peer", "principal"}
    assert set(fams["F8"]["F8_type_mass"]["act"]) == {
        "stop_report",
        "escalate_human",
        "escalate_peer",
        "hold",
        "violate",
    }
    tsn = _load("analyze_tsn")
    tsn.FLOOR = 0.0
    from deepsteer.kdg.schema import load_scenarios

    scen, _ = load_scenarios(p1_file)
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    r = tsn.analyze_model(out / "olmo3_instruct", set(status), status)
    assert r["n"] == len(status) and set(r["delta"]) == {0, 1, 3, 6}
    assert r["verdict"] in ("no_reminder_effect", "decays", "no_decay_detectable")


def test_tsn_floor_drop_flags_one_model_dropping_and_not_the_other():
    # most probable failure: an asymmetric usable-count drop between models is read as a model
    # difference instead of being flagged as a floor artifact (author rule 2026-10-02)
    tsn = _load("analyze_tsn")
    same = tsn.floor_drop({"a": {"n": 100, "n_screen": 136}, "b": {"n": 85, "n_screen": 118}})
    assert same["floor_artifact_flag"] is False
    skew = tsn.floor_drop({"a": {"n": 130, "n_screen": 136}, "b": {"n": 40, "n_screen": 118}})
    assert skew["floor_artifact_flag"] is True
    assert tsn.floor_drop({"a": {"n": 1, "n_screen": 2}}) is None


def test_filler_rotation_covers_every_position_and_lengths_hit_their_targets():
    # most probable failure (P2-A4): the rotation is a no-op (every scenario gets the fixed order,
    # so the KDG-A16 discriminator re-measures the confound), or a ladder set misses its target
    from collections import Counter

    from deepsteer.kdg.phase2 import TSN_LENGTHS, filler_path, load_filler_turns, rotate_filler

    base = load_filler_turns()
    first = Counter(rotate_filler(base, f"S-{i:03d}")[0] for i in range(600))
    assert len(first) == 6 and min(first.values()) > 60  # each exchange leads for ~1/6 of ids
    assert rotate_filler(base, "x") == rotate_filler(base, "x")  # deterministic per id
    for L in TSN_LENGTHS[1:]:
        d = json.loads(filler_path(L).read_text())
        assert len(d["turns"]) == 6 and abs(d["olmo3_tokens"] - L) <= 0.1 * L


def test_followup_units_dry_run_and_analysis(tmp_path, p1_file):
    # most probable failure: the follow-up analysis reads cell names the driver does not write
    out = tmp_path / "p2b"
    pod.run(out, True, ["olmo3_instruct"], ["TSN_ROT", "TSN_LEN"], [p1_file])
    d = out / "olmo3_instruct"
    for u in lib.TSN_ROT_UNITS + lib.TSN_LEN_UNITS:
        assert (d / f"{u}.jsonl").exists(), u
    rows = [json.loads(x) for x in (d / "dl_tsn2000_reminder_k6.jsonl").read_text().splitlines()]
    assert rows and all(r["tsn_filler_tokens"] == 2000 for r in rows)
    fu = _load("analyze_tsn_followups")
    fu.T.FLOOR = 0.0
    from deepsteer.kdg.schema import load_scenarios

    scen, _ = load_scenarios(p1_file)
    status = {s.id: {o.option_id: o.norm_status for o in s.options} for s in scen}
    ra = fu.a4a(d, set(status), status)
    rb = fu.a4b(d, set(status), status, 2000)
    assert ra["n"] == len(status) and rb["n"] == len(status)
    assert ra["verdict"] in ("filler_confound_R_a", "position_effect_R_b", "unresolved")
