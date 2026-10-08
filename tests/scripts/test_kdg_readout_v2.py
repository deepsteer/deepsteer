"""Readout version 2 (mask-derived positions) for the KDG panel harness (ANOMALIES KDG-A20)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "papers" / "kdg_panel" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import analyze_phase1_session_a as A  # noqa: E402
import kdg_pod_lib as lib  # noqa: E402


@pytest.fixture(scope="module")
def tiny_dense():
    # a random 2-layer Llama-architecture model (RoPE, like every dense panel model) in bf16, on
    # the cached Qwen2.5 tokenizer; skip when the tokenizer is not cached
    import torch
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen2.5 tokenizer not cached")
    tok.padding_side = "left"
    cfg = LlamaConfig(num_hidden_layers=2, hidden_size=64, intermediate_size=128,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=len(tok),
                      max_position_embeddings=4096)
    torch.manual_seed(0)
    return types.SimpleNamespace(tok=tok, model=LlamaForCausalLM(cfg).to(torch.bfloat16).eval(),
                                 device=torch.device("cpu"), torch=torch)


def _prompts(tok):
    words = "the agent weighs the request against the posted rule and answers with one letter"
    return [" ".join(words.split()[: 4 + 2 * k]) * (1 + k) for k in range(8)]


def test_v2_padded_readout_equals_unpadded_in_bf16(tiny_dense):
    # most probable failure: the version-2 readout still reads left-padded rows at shifted
    # positions (the cells-of-record deviation), so batch composition moves a row's log-probs
    P = _prompts(tiny_dense.tok)
    read = lib.ModelWrapper.raw_next_logprobs
    batched = read(tiny_dense, P, batch_size=8, add_special_tokens=False)
    alone = read(tiny_dense, P, batch_size=1, add_special_tokens=False)
    np.testing.assert_array_equal(batched, alone)
    v1 = read(tiny_dense, P, batch_size=8, add_special_tokens=False, readout_version=1)
    assert np.abs(v1.astype(np.float32) - alone.astype(np.float32)).max() > 0  # v1 really differs


def test_rows_of_record_read_as_version_1_and_new_rows_carry_2(tmp_path):
    # most probable failure: new rows omit the field and would be read as cells of record
    assert lib.readout_version({}) == 1 and lib.readout_version({"readout_version": 2}) == 2
    rc = subprocess.run([sys.executable, str(SCRIPTS / "pod_kdg_phase1.py"), "--dry-run",
                         "--models", "olmo3_instruct", "--units",
                         "dl_chat_neutral,validate_forward_matches_generate",
                         "--out", str(tmp_path)], capture_output=True, text=True, cwd=REPO)
    assert rc.returncode == 0, rc.stderr[-2000:]
    d = tmp_path / "olmo3_instruct"
    row = json.loads((d / "dl_chat_neutral.jsonl").read_text().splitlines()[0])
    assert row["readout_version"] == lib.KDG_READOUT_VERSION == 2
    rec = json.loads((d / "forward_matches_generate.json").read_text())
    # the VALIDATE record bounds the cells of record too: both readouts, per prompt, two batches
    assert set(rec["batches"]) == {"g6", "pad_ladder"}
    r0 = rec["batches"]["pad_ladder"][0]
    assert {"pad", "v1_vs_gen", "v2_vs_gen", "unpadded_vs_gen"} <= set(r0)
    assert "max_abs_nats_v1" in rec and rec["readout_version"] == 2


def test_e_refuses_to_mix_readout_versions(tmp_path):
    # most probable failure: a new version-2 twin cell is combined with version-1 cells of record
    # into one E, which then carries the readout difference
    row = {"scenario_id": "s1", "option_logps": {"A": -1.0, "B": -1.0}, "order": {"A": "o1",
           "B": "o2"}, "option_mass": 0.9}
    names = ("d", "j", "dn", "jn")
    for n in names:
        r = dict(row, readout_version=2) if n == "jn" else row
        (tmp_path / f"{n}.jsonl").write_text(json.dumps(r) + "\n")
    with pytest.raises(ValueError, match="mix readout versions"):
        A.four_cells([tmp_path], names, {"s1": {"o1": "violating", "o2": "consistent"}})


def test_p2d_measures_on_the_stack_of_record_without_a_torch_upgrade():
    # most probable failure: the p2d block inherits p2c's torch 2.6 upgrade (or runs on whatever the
    # image ships), so the v1 bound is measured on a stack the cells of record never ran on
    sh = (REPO / "papers/kdg_panel/runpod/remote_kdg_phase2.sh").read_text()
    start = sh.index('if [ "$PROFILE" = "p2d" ]; then')
    block = sh[start:sh.index("\nfi\n", start)]  # the p2d block only
    assert '"2.4.1+cu124 5.12.1") ;;' in block and "torch==2.6.0" not in block
    assert "olmo3_instruct,llama31_instruct_meta,tulu3_final,qwen25_instruct_p1" in block
    assert block.index("--dry-run") < block.index('--out "$OUT/validate"')


def test_validate_reading_takes_the_v1_bound_over_the_panel_pad_range(tmp_path):
    # most probable failure: the bound is taken over the 2,000-token TSN rows (or v2), not over
    # pads the panel actually had, so a large in-range v1 error hides or a far one dominates
    spec = importlib.util.spec_from_file_location(
        "an", SCRIPTS / "analyze_kdg_a20_validate.py")
    an = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(an)
    d = tmp_path / "validate"
    for key in an.MODELS:
        (d / key).mkdir(parents=True)
        rows = [{"pad": 0, "v1_vs_gen": 0.0, "v2_vs_gen": 0.0, "unpadded_vs_gen": 0.0},
                {"pad": 40, "v1_vs_gen": 0.07 if key == "qwen25_instruct_p1" else 0.02,
                 "v2_vs_gen": 0.01, "unpadded_vs_gen": 0.01},
                {"pad": 2000, "v1_vs_gen": 0.9, "v2_vs_gen": 0.02, "unpadded_vs_gen": 0.01}]
        (d / key / "forward_matches_generate.json").write_text(json.dumps(
            {"batches": {"g6": rows[2:], "pad_ladder": rows[:2]}}))
    out = tmp_path / "a.json"
    assert an.main(["--dir", str(d), "--out", str(out)]) == 0
    m = json.loads(out.read_text())["models"]
    assert m["olmo3_instruct"]["v1_bound_in_range"] == 0.02
    assert m["olmo3_instruct"]["v1_outcome"] == "bounded"
    assert m["qwen25_instruct_p1"]["v1_outcome"] == "re-read (item 2)"
    assert m["olmo3_instruct"]["v2_outcome"] == "passes"


def _an_reread():
    spec = importlib.util.spec_from_file_location(
        "rr", SCRIPTS / "analyze_kdg_a20_qwen_reread.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_reread_branch_order_is_the_registered_one():
    # most probable failure: a CI inside +/-0.005 that excludes 0 reads "correction" (the
    # overlap resolved against the listed order fixed before data)
    rr = _an_reread()
    assert rr.branch([0.001, 0.004]) == "stands"
    assert rr.branch([-0.004, 0.004]) == "stands"
    assert rr.branch([0.002, 0.012]) == "correction"
    assert rr.branch([-0.009, 0.006]) == "unresolved"


def test_reread_of_identical_cells_stands(tmp_path):
    # most probable failure: the paired comparison misaligns scenarios or rows, so identical
    # readouts produce a nonzero dE; uses the local cells of record, skips without them
    import analyze_screen_rates as SR

    rec = SR.MODELS["qwen25_instruct"]
    if not (rec / "dl_chat_neutral.jsonl").exists():
        pytest.skip("Qwen2.5 cells of record not on disk")
    new = tmp_path / "reread"
    new.mkdir()
    for c in ("dl_chat_neutral", "jl_chat_neutral", "dl_chat_neutral_pressure_removed",
              "jl_chat_neutral_pressure_removed"):
        rows = [dict(json.loads(x), readout_version=2)
                for x in (rec / f"{c}.jsonl").read_text().splitlines() if x.strip()]
        (new / f"{c}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "a.json"
    assert _an_reread().main(["--reread", str(new), "--out", str(out)]) == 0
    rep = json.loads(out.read_text())
    assert rep["dE"]["mean"] == 0.0 and rep["branch"] == "stands"
    assert rep["row_abs_dlogp"]["max"] == 0.0 and rep["n"] == 586


def test_p2e_rereads_the_four_c3_cells_on_the_stack_of_record():
    # most probable failure: the re-read runs on a different stack or a different cell set, so
    # dE mixes the readout fix with other changes
    sh = (REPO / "papers/kdg_panel/runpod/remote_kdg_phase2.sh").read_text()
    start = sh.index('if [ "$PROFILE" = "p2e" ]; then')
    block = sh[start:sh.index("\nfi\n", start)]  # the p2e block only
    assert '"2.4.1+cu124 5.12.1") ;;' in block and "torch==2.6.0" not in block
    assert ("P2E_UNITS=validate_forward_matches_generate,dl_chat_neutral,jl_chat_neutral,"
            "dl_chat_neutral_pressure_removed,jl_chat_neutral_pressure_removed") in block
    assert "--models qwen25_instruct_p1" in block and "--scenario-ids-file" not in block


def test_p2g_runs_the_reread_on_the_stack_of_record_before_the_torch_upgrade():
    # most probable failure: the combined profile upgrades torch (for GPT-OSS) before the Qwen2.5
    # re-read, so the re-read runs off the stack of record; or one failure skips the other run
    sh = (REPO / "papers/kdg_panel/runpod/remote_kdg_phase2.sh").read_text()
    b = sh[sh.index('if [ "$PROFILE" = "p2g" ]; then'):sh.index("# ---- p2e: KDG-A20 item 2")]
    stack = b.index('"2.4.1+cu124 5.12.1") ;;')
    dry = b.index("--dry-run --c0-dm")
    reread = b.index('python $S --out "$OUT/reread"')
    upgrade = b.index('"torch==2.6.0"')
    c0dm = b.index('python $G --c0-dm --out "$OUT/c0dm"')
    assert stack < dry < reread < upgrade < c0dm
    assert "rc_q=$?" in b and "rc_g=$?" in b and "exit 1" in b[c0dm:]
    assert "--batch-invariance" not in b and "--require-timing" not in b


class TestSamplerV2:
    def test_scenario_seeds_are_distinct_and_stable(self):
        # most probable failure: two scenarios share a seed, so their rollout k share a stream
        from deepsteer.kdg.schema import load_scenario_dir

        files = sorted((REPO / "papers/kdg_panel/data").glob("*_scenarios_*.json"))
        S, _ = load_scenario_dir(files)
        seeds = [lib.scenario_seed(s.id) for s in S]
        assert len(set(seeds)) == len(seeds)
        assert lib.scenario_seed("F1-A-12") == lib.scenario_seed("F1-A-12")

    def test_per_scenario_call_sites_seed_by_scenario(self, tmp_path):
        # KDG-A23. most probable failure: a sampled cell still passes the shared SEED to every
        # scenario's generate call (common random numbers across scenarios)
        from deepsteer.kdg.schema import load_scenario_dir

        S, metas = load_scenario_dir(sorted((REPO / "papers/kdg_panel/data")
                                            .glob("*_scenarios_*.json")))
        S = [s for s in S if s.family != "F2"][:4]
        seen = []

        class Rec(lib.StubModel):
            def generate(self, rendered, *, seed, **kw):
                seen.append(seed)
                return super().generate(rendered, seed=seed, **kw)

        ctx = lib.Ctx("k", "instruct", Rec(), tmp_path, lib.Manifest(tmp_path, True, metas),
                      True, n_d=2, n_dose=2)
        lib.cell_d_chat(ctx, S)
        d_chat = list(seen)
        lib.cell_dose_forced(ctx, S, arm="dose2")
        forced = seen[len(d_chat):]
        # distinct across scenarios within each cell; the same scenario keeps its seed across
        # arms, so rollout k stays paired across arms within a scenario (the P1-A15 pairing)
        assert len(set(d_chat)) == len(S) and len(set(forced)) == len(S)
        assert d_chat == forced and lib.SEED not in d_chat
        row = json.loads((tmp_path / "d_chat_dose0.jsonl").read_text().splitlines()[0])
        assert row["sampler_version"] == lib.KDG_SAMPLER_VERSION == 2
        assert lib.sampler_version({}) == 1

    def test_sampling_disables_config_warpers_and_greedy_passes_none(self):
        # most probable failure: the model's generation config (top_p 0.9/0.95, Qwen's top_k 20 and
        # repetition penalty 1.05) rides along on top of the passed temperature
        import torch

        captured = []

        class FakeHF:
            def generate(self, input_ids, attention_mask, **kw):
                captured.append(kw)
                n = input_ids.shape[0]
                return types.SimpleNamespace(
                    sequences=torch.cat([input_ids, torch.ones(n, 1, dtype=torch.long)], 1),
                    logits=(torch.zeros(n, 8),))

        class Tok:
            pad_token_id = 0

            def __call__(self, texts, **kw):
                ids = torch.ones(len(texts), 3, dtype=torch.long)
                return types.SimpleNamespace(to=lambda d: {"input_ids": ids,
                                                           "attention_mask": torch.ones_like(ids)})

            def decode(self, ids, skip_special_tokens=True):
                return "A"

        m = object.__new__(lib.ModelWrapper)
        m.tok, m.model, m.device, m.torch = Tok(), FakeHF(), "cpu", torch
        m._generate_batch(["x", "y"], 2, 0.7, 5, False)
        m._generate_batch(["x"], 2, 0.0, 5, False)
        assert captured[0]["top_p"] == 1.0 and captured[0]["top_k"] == 0
        assert captured[0]["repetition_penalty"] == 1.0 and captured[0]["temperature"] == 0.7
        assert "top_p" not in captured[1] and captured[1]["do_sample"] is False
