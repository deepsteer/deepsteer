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
    block = sh[sh.index('if [ "$PROFILE" = "p2d" ]; then'):sh.index("# ---- no-model gates always")]
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
