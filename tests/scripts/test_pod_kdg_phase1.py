"""KDG Phase 1 harness (papers/kdg_panel/scripts/pod_kdg_phase1.py, kdg_pod_lib C1/C3 cells,
deepsteer.kdg.phase1_frames). Each test names the failure mode it guards (CLAUDE.md local-test
rule). No model is loaded anywhere in this file."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "papers" / "kdg_panel" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kdg_pod_lib as lib  # noqa: E402

from deepsteer.kdg.phase1_frames import (  # noqa: E402
    FRAMING_CLAUSES,
    LETTER_ONLY_INSTRUCTION,
    framing_prefix,
    render_letter_user_message,
)
from deepsteer.kdg.schema import _DOSE_INSTRUCTION, assign_letters, save_scenarios  # noqa: E402
from tests.kdg.conftest import make_scenario  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pod = _load("pod_kdg_phase1")


@pytest.fixture(scope="module")
def scen():
    return [
        make_scenario(f, id=f"{f}-A-{k:02d}") for f in ("F1", "F3", "F4", "F5") for k in range(2)
    ]


@pytest.fixture(scope="module")
def scenario_file(tmp_path_factory, scen):
    p = tmp_path_factory.mktemp("p1scen") / "pilot_scenarios_A_test.json"
    save_scenarios(p, scen, {"generator": "test", "generator_half": "A"})
    return p


class TestFrames:
    def test_jl_renders_third_person_frame_not_agent_frame(self, scen):
        # most probable failure:
        #   the J cell silently reuses the agent message (D == J by construction)
        s = scen[0]
        order = assign_letters(s, 0)
        j = render_letter_user_message(s, order, "eval", "neutral")
        d = render_letter_user_message(s, order, "agent", "neutral")
        assert s.eval_text in j and s.agent_text not in j
        assert s.agent_text in d and s.eval_text not in d
        assert f"Which option should {s.agent_name} choose?" in j

    def test_pressure_removed_variant_uses_the_twin_text(self, scen):
        # most probable failure: the null cell renders the pressured text (null == measurement)
        s = scen[0]
        order = assign_letters(s, 0)
        assert s.eval_text_pressure_removed in render_letter_user_message(
            s, order, "eval", "neutral", pressure_removed=True
        )
        assert s.agent_text_pressure_removed in render_letter_user_message(
            s, order, "agent", "neutral", pressure_removed=True
        )

    def test_prefix_identical_for_j_and_d_and_differs_only_in_clause(self, scen):
        # most probable failure:
        #   the E/D/N contrast also varies wrapper text or differs between frames
        s = scen[1]
        order = assign_letters(s, 3)
        bodies = {}
        for p in FRAMING_CLAUSES:
            j = render_letter_user_message(s, order, "eval", p)
            d = render_letter_user_message(s, order, "agent", p)
            assert j.split("\n\n", 1)[0] == d.split("\n\n", 1)[0] == framing_prefix(p)
            bodies[p] = (j.split("\n\n", 1)[1], d.split("\n\n", 1)[1])
        assert len(set(bodies.values())) == 1
        stripped = {framing_prefix(p).replace(c, "<C>") for p, c in FRAMING_CLAUSES.items()}
        assert len(stripped) == 1

    def test_letter_only_instruction_is_the_dose0_instruction_of_record(self):
        # most probable failure:
        #   J and D ask for the letter differently, confounding frame with instruction
        assert _DOSE_INSTRUCTION["dose0"].endswith(LETTER_ONLY_INSTRUCTION)


class _NoChatStub(lib.StubModel):
    def render_chat(self, messages):
        raise AssertionError("raw cell touched the chat template")


class _Ctx:
    @staticmethod
    def make(tmp_path, model):
        man = lib.Manifest(tmp_path, True, [])
        return lib.Ctx("k", "instruct", model, tmp_path / "k", man, True, n_raw_perm=8)


class TestCells:
    def test_letter_chat_tokenizes_without_special_tokens(self, tmp_path, scen):
        # most probable failure:
        #   forward pass adds BOS the generation path does not (readout off by one token)
        m = lib.StubModel("instruct")
        ctx = _Ctx.make(tmp_path, m)
        lib.cell_letter_chat(ctx, scen[:2], frame="agent", prefix="eval")
        assert m.last_add_special_tokens is False
        lib.cell_raw(ctx, scen[:2], frame="agent")
        assert m.last_add_special_tokens is True

    def test_stage_raw_cells_never_touch_chat_template(self, tmp_path, scen):
        # most probable failure:
        #   a raw cell on a stage checkpoint renders through its (drifted) template
        ctx = _Ctx.make(tmp_path, _NoChatStub("instruct"))
        for u in lib.RAW_UNITS:
            lib.UNITS[u][1](ctx, scen[:2])

    def test_letter_chat_rows_carry_prefix_frame_and_both_versions(self, tmp_path, scen):
        # most probable failure:
        #   rows cannot be joined back to prefix/frame, or the template version is v1.0.0
        ctx = _Ctx.make(tmp_path, lib.StubModel("instruct"))
        lib.UNITS["jl_chat_deploy_pressure_removed"][1](ctx, scen[:2])
        rows = [
            json.loads(x)
            for x in (tmp_path / "k" / "jl_chat_deploy_pressure_removed.jsonl")
            .read_text()
            .splitlines()
        ]
        assert {r["prefix"] for r in rows} == {"deploy"}
        assert {r["frame"] for r in rows} == {"eval"} and {r["variant"] for r in rows} == {
            "pressure_removed"
        }
        assert {r["template_version"] for r in rows} == {"p1-1.0.0"}
        assert all(set(r["option_logps"]) <= set("ABCD") for r in rows)

    def test_rendered_identity_detects_template_drift(self, scen):
        # most probable failure: the stage fork check passes whatever the reference renders
        m = lib.StubModel("instruct")
        assert lib.rendered_identity_mismatches(m, m.render_chat, scen) == []
        drifted = lib.rendered_identity_mismatches(m, lambda msgs: m.render_chat(msgs) + " ", scen)
        assert drifted and all(re.match(r"F\d-A-\d\d/(agent|eval)$", x) for x in drifted)


class TestRegistryAndDriver:
    def test_registry_keys_unique_and_phase1_revisions_pinned(self):
        # most probable failure:
        #   a phase1 key shadows a tier1 key, or a revision is a prefix/placeholder
        cfg, reg = pod.registry()
        for k, v in cfg["phase1"].items():
            assert re.fullmatch(r"[0-9a-f]{40}", v["revision"]), k
            if v.get("template_sha256"):
                assert re.fullmatch(r"[0-9a-f]{64}", v["template_sha256"]), k
        assert set(cfg["phase1"]) & set(cfg["tier1"]) == set()

    def test_stage_checkpoints_point_at_final_instruct_for_identity(self):
        # most probable failure: the identity check compares a stage to itself (no reference set)
        cfg = yaml.safe_load((REPO / "papers/kdg_panel/models.yaml").read_text())
        for k in ("olmo3_sft", "olmo3_dpo"):
            assert cfg["phase1"][k]["rendered_equals"] == "olmo3_instruct"

    def test_c1_has_twelve_distinct_cells(self):
        # most probable failure:
        #   the prefix x frame x variant grid collapses (closure captured the loop var)
        assert len(set(lib.C1_UNITS)) == 12
        names = {lib.UNITS[u][1].__code__ for u in lib.C1_UNITS}
        assert len(names) == 1  # one cell function, parameterised
        assert (
            len({(u.split("_")[0], u.split("_")[2], u.endswith("removed")) for u in lib.C1_UNITS})
            == 12
        )

    def test_dry_run_end_to_end(self, tmp_path, scenario_file):
        # most probable failure:
        #   raw-only base gets chat units, or a stage load record lacks the identity check
        out = tmp_path / "dry"
        p = pod.run(
            out,
            True,
            ["olmo3_instruct", "olmo3_sft", "qwen25_base"],
            ["RAW", "C1", "C3CHAT", "VALIDATE"],
            [scenario_file],
        )
        man = json.loads(Path(p).read_text())
        st = man["unit_status"]
        assert all(v["status"] == "ok" for v in st.values()), st
        assert {u.split("/", 1)[1] for u in st if u.startswith("qwen25_base/")} == set(
            lib.RAW_UNITS
        )
        sft = [ld for ld in man["loads"] if ld["key"] == "olmo3_sft"][0]
        assert sft["rendered_identity"]["mismatches"] == []
        assert (out / "olmo3_instruct" / "dl_chat_eval.npz").exists()
        assert (out / "olmo3_instruct" / "forward_matches_generate.json").exists()
        assert not lib.verify_manifest(out / "manifest_kdg.json")

    def test_unknown_unit_or_model_fails_loudly(self, tmp_path, scenario_file):
        # most probable failure: a typo in the session script silently runs nothing
        with pytest.raises(SystemExit):
            pod.expand_units(["C1", "dl_chat_evl"])
        with pytest.raises(SystemExit):
            pod.run(tmp_path / "x", True, ["olmo3_sfft"], ["RAW"], [scenario_file])


class TestSessionAAnalysis:
    def test_end_to_end_on_stub_session(self, tmp_path, scen, scenario_file):
        # most probable failure: the analysis reads a step directory the session script never
        # writes (path drift between remote_kdg_phase1.sh and the analysis), so a verdict is empty
        import csv as _csv

        ana = _load("analyze_phase1_session_a")
        p1a, old, data = tmp_path / "p1a", tmp_path / "old", tmp_path / "data"
        data.mkdir()
        (data / scenario_file.name).write_text(scenario_file.read_text())
        ids = [s.id for s in scen]
        (data / "screened_ids_a17_union.json").write_text(json.dumps({"ids": ids}))
        with open(data / "per_scenario_raw_union.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["run", "model", "scenario_id", "above_floor", "above_floor_null"])
            for i in ids:
                w.writerow(["three_cell_union", "instruct", i, "True", "True"])
        steps = [  # (step dir as in remote_kdg_phase1.sh, models, units)
            ("stages_raw", ["olmo3_sft", "olmo3_dpo"], ["RAW"]),
            ("base_new", ["olmo3_base"], ["RAW"]),
            ("final_new", ["olmo3_instruct"], ["RAW"]),
            ("final_c1", ["olmo3_instruct"], ["C1"]),
            ("stages_chat", ["olmo3_sft", "olmo3_dpo"], ["C3CHAT"]),
            ("final_dose", ["olmo3_instruct"], ["DOSE"]),
        ]
        for step, models, units in steps:
            pod.run(p1a / step, True, models, units, [scenario_file])
        pod.run(old / "kdg2", True, ["olmo3_base", "olmo3_instruct"], ["RAW"], [scenario_file])
        out = tmp_path / "rep.json"
        rc = ana.main(
            [
                "--p1a",
                str(p1a),
                "--old",
                str(old / "kdg2"),
                "--data",
                str(data),
                "--dry",
                "--write",
                str(out),
                "--floor",
                "0",
                "--min-anchored",
                "1",
            ]
        )
        rep = json.loads(out.read_text())
        assert rc == 0 and rep["spec_values"] is False  # stub run is marked non-spec
        assert rep["C3"]["n_shared"] > 0 and rep["C3"]["verdict_primary_scale_E_norm"]
        assert rep["C1"]["primary_screened_twins"]["verdict"] in {
            "R_a",
            "R_b",
            "R_c",
            "mixed_Ra_Rc",
            "unresolved",
        }
        assert rep["C1"]["A7"]["n"] > 0
        assert rep["C2"]["n_scenarios"] > 0 and rep["C2"]["verdict"] != "no_data"


class TestDoseForced:
    class _G:
        def __init__(self, ids, step):
            self.token_ids, self.decision_step = ids, step

    class _Tok:
        # decode = join of token strings, so truncation is visible
        def __init__(self, toks):
            self.toks = toks

        def decode(self, ids):
            return "".join(self.toks[i] for i in ids)

    def test_forced_text_strips_the_natural_anchor(self):
        # most probable failure: the forced prompt keeps the model's own "Answer:" and appends a
        # second one, so the read position is not the decision position
        toks = ["Weigh ", "the ", "stakes.\n", "Answer:", " B"]
        m = self._Tok(toks)
        text, natural, n = lib.forced_reasoning_text(m, self._G([0, 1, 2, 3, 4], 4), 512)
        assert natural and n == 4 and "Answer" not in text and text.endswith("stakes.")

    def test_forced_text_truncates_at_the_budget_when_no_anchor(self):
        # most probable failure: reasoning past the budget leaks into the forced prompt
        toks = [f"t{i} " for i in range(40)]
        m = self._Tok(toks)
        text, natural, n = lib.forced_reasoning_text(m, self._G(list(range(40)), -1), 10)
        assert not natural and n == 10 and text == "".join(toks[:10]).rstrip()

    def test_anchor_beyond_budget_counts_as_truncated(self):
        # most probable failure: a natural answer found after the budget is read as if in budget
        toks = [f"t{i} " for i in range(30)] + ["Answer:", " A"]
        m = self._Tok(toks)
        text, natural, n = lib.forced_reasoning_text(m, self._G(list(range(32)), 31), 16)
        assert not natural and n == 16

    def test_unit_saves_natural_and_forced_rows_one_to_one(self, tmp_path, scen):
        # most probable failure: forced rows misalign with natural rows (different counts or
        # orders), or the forced pass re-adds BOS
        m = lib.StubModel("instruct")
        ctx = _Ctx.make(tmp_path, m)
        lib.UNITS["d_chat_dose2_bf"][1](ctx, scen[:3])
        assert m.last_add_special_tokens is False
        d = tmp_path / "k"
        nat = [json.loads(x) for x in (d / "d_chat_dose2_bf.jsonl").read_text().splitlines()]
        frc = [json.loads(x) for x in (d / "d_chat_dose2_bf_forced.jsonl").read_text().splitlines()]
        assert len(nat) == len(frc) > 0
        assert [(r["scenario_id"], r["rollout"], r["order"]) for r in nat] == [
            (r["scenario_id"], r["rollout"], r["order"]) for r in frc
        ]
        assert all(r["budget"] == 512 and "forced_natural_anchor" in r for r in frc)
        assert not any(r["family"] == "F2" for r in nat)

    def test_long_rider_uses_2048_and_two_rollouts(self, tmp_path, scen):
        # most probable failure: the descriptive rider silently runs at 512 tokens or 16 rollouts
        ctx = _Ctx.make(tmp_path, lib.StubModel("instruct"))
        ctx.dry = False  # count rollouts as on the pod
        lib.UNITS["d_chat_dose2_long"][1](ctx, scen[:2])
        rows = [
            json.loads(x)
            for x in (tmp_path / "k" / "d_chat_dose2_long_forced.jsonl").read_text().splitlines()
        ]
        assert {r["budget"] for r in rows} == {2048}
        assert max(r["rollout"] for r in rows) == 1
