"""ST T1.5 pod driver (scripts/t15_driver.py): Amendment 20's parity gates and sequencing, the
capability scorer, and a dry run of every phase on a tiny GPT-OSS with the real harmony tokenizer
(the vLLM phases on the screen stub)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import t15_driver as T  # noqa: E402

DATA = ROOT / "papers" / "st_live_reference" / "data"


def _done(**kw) -> dict:
    return {"finished": "t", **kw}


def test_failed_base_check_sends_none_and_every_adapter_to_hf():
    # most probable failure: a failed base check still reads adapter parity, or the adapters go
    # to HF while none stays on the bf16-vLLM engine (ΔP across two engines)
    rec = {"phases": {"beh_cap": _done(comparator="beh"), "base_hf": _done(),
                      "base_vllm": _done(base_check_ok=False, band_tv=0.02)}}
    names = T.plan(rec)
    assert names[:4] == ["beh_cap", "base_hf", "base_vllm", "hfgen_none"]
    for arm in T.ARMS_ALL:
        rec["phases"][f"vllm_{arm}"] = _done(parity_ok=False)
    names = T.plan(rec)
    for arm in T.ARMS_ALL:
        assert names.index(f"hfgen_{arm}") == names.index(f"vllm_{arm}") + 1
    assert names[-1] == "readouts"
    passed = {"phases": {"beh_cap": _done(comparator="beh"),
                         "base_vllm": _done(base_check_ok=True)}}
    assert "hfgen_none" not in T.plan(passed)


def test_comparator_leads_and_only_it_and_anchor_get_the_panel():
    # most probable failure: the panel runs on the step-matched checkpoint after the cap rerun
    # made epoch-3 the comparator, or on L-self (no panel, Amendment 18)
    rec = {"phases": {"beh_cap": _done(comparator="beh_epoch3")}}
    assert T.arms_order(rec) == ["beh_epoch3", "beh", "anchor", "self"]
    assert T.panel_arms(rec) == {"beh_epoch3", "anchor"}
    rec = {"phases": {"beh_cap": _done(comparator="beh")}}
    assert T.panel_arms(rec) == {"beh", "anchor"}
    assert "self" not in T.panel_arms(rec)


def test_tv_renormalises_over_shown_letters_and_treats_absent_letters_as_maximal():
    # most probable failure: a row whose shown letters all fall outside vLLM's top 20 gives NaN
    # and the mean TV silently passes the bar
    rows = [{"shown": ["A", "B", "C"]}, {"shown": ["A", "B"]}]
    hf = np.log(np.array([[0.5, 0.3, 0.2, 1e-9, 1e-9], [0.6, 0.4, 1e-9, 1e-9, 1e-9]]))
    assert T.tv(hf, hf, rows) == pytest.approx(0.0, abs=1e-9)
    gone = hf.copy()
    gone[1, :] = -np.inf
    assert T.tv(hf, gone, rows) == pytest.approx(0.5)


def test_capability_parse_and_fallback_slice():
    # most probable failure: a worked GSM8K answer scored on its first number, an MMLU article
    # "A" read as the answer, or the 100-item fallback taking 100 GSM8K items
    g = {"dataset": "openai/gsm8k"}
    m = {"dataset": "cais/mmlu"}
    assert T.cap_parse(g, "She has 3 + 4 = 7 apples, so 1,234 in all.\n\n1,234") == "1234"
    assert T.cap_parse(m, "B. Paris") == "B"
    assert T.cap_parse(m, "A careful reading shows the answer is C.") == "C"
    assert T.cap_parse(m, None) is None
    assert T.cap_gold(g, {"answer": "steps\n#### 1,200"}) == "1200"
    assert T.cap_gold(m, {"answer": 2}) == "C"
    items = json.loads((DATA / "t15_capability_v1.json").read_text())["items"]
    fb = T.cap_fallback(items)
    assert len(fb) == 100
    assert sum(items[i]["dataset"] == "openai/gsm8k" for i in fb) == 50
    assert fb == list(range(50)) + list(range(100, 150))


@pytest.mark.slow
def test_final_text_decodes_the_whole_harmony_final_answer():
    # most probable failure: the 64-token cut of refusal_retention.final_answer drops a worked
    # GSM8K answer's last number, or a trace that never closes is scored as answered
    import kdg_harmony as kh
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    spec = P.registry()[T.KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    sc = type("S", (), {"tok": tok, "fmt": build_format(T.KEY, tok, spec)})()
    body = tok.encode(" Step." * 80 + " So the total is 1,234", add_special_tokens=False)
    think = tok.encode("Add them.", add_special_tokens=False)
    gen = list(kh.ANALYSIS_OPEN) + think + list(kh.CANON_HEADER) + body
    text = T.final_text(sc, gen)
    assert len(body) > 64 and text.endswith("1,234")
    assert T.cap_parse({"dataset": "openai/gsm8k"}, text) == "1234"
    assert T.final_text(sc, list(kh.ANALYSIS_OPEN) + think) is None


@pytest.mark.slow
def test_save_bf16_keeps_the_expert_weights_after_a_dequantizing_load(tmp_path):
    # most probable failure: save_pretrained after an MXFP4 dequantize load reverts the load's
    # weight conversions and writes only the expert biases (measured on transformers 5.17), so
    # vLLM serves a checkpoint without gate_up_proj / down_proj
    import torch
    from transformers import AutoModelForCausalLM
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.mxfp4 import Mxfp4Dequantize

    sys.path.insert(0, str(ROOT / "tests" / "scripts"))
    import st_tiny

    st_tiny.build(T.KEY, tmp_path / "tiny")
    m = AutoModelForCausalLM.from_pretrained(str(tmp_path / "tiny"), dtype=torch.bfloat16)
    m._weight_conversions = [  # what a Mxfp4Config(dequantize=True) load attaches
        WeightConverter(source_patterns=["down_proj_blocks", "down_proj_scales"],
                        target_patterns=r"down_proj$", operations=[Mxfp4Dequantize(None)]),
        WeightConverter(source_patterns=["gate_up_proj_blocks", "gate_up_proj_scales"],
                        target_patterns=["gate_up_proj$"], operations=[Mxfp4Dequantize(None)])]
    m.config.quantization_config = {"quant_method": "mxfp4", "dequantize": True}
    info = T.save_bf16(m, tmp_path / "out")
    cfg = json.loads((tmp_path / "out" / "config.json").read_text())
    assert "quantization_config" not in cfg and info["n_tensors"] == len(m.state_dict())
    r = AutoModelForCausalLM.from_pretrained(str(tmp_path / "out"), dtype=torch.bfloat16)
    for k, v in m.state_dict().items():
        assert torch.equal(r.state_dict()[k], v), k
    assert any(k.endswith("experts.gate_up_proj") for k in r.state_dict())


def test_adapters_are_verified_against_the_t14_hashes(tmp_path):
    # most probable failure: a partial or stale adapter upload evaluated as the T1.4 adapter
    for arm in T.ARMS_ALL:
        (tmp_path / arm).mkdir()
        (tmp_path / arm / "adapter_config.json").write_text("{}")
        (tmp_path / arm / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(SystemExit):
        T.verify_adapters({"adapters": tmp_path})
    real = ROOT / "papers/st_live_reference/outputs/t14/gpt_oss_20b/adapters"
    if (real / "beh" / "adapter_model.safetensors").exists():
        T.verify_adapters({"adapters": real})


def test_parity_rows_are_the_gate_scenarios_on_h1_train():
    # most probable failure: parity rows drawn from H1_held (the ΔP scenarios) instead of the
    # gate's 20 H1_train scenarios
    draw = json.loads((DATA / "t13_draw_v1.json").read_text())["gate"]
    split = json.loads((ROOT / "papers/st_live_reference/split_v1.json").read_text())
    assert len(draw) == 20 and set(draw) <= set(split["H1_train"])
    assert not set(draw) & set(split["H1_held"])
    assert T.N_PARITY == 64 and 4 * len(draw) >= T.N_PARITY


def _dry(tmp_path, *extra: str) -> Path:
    inp = tmp_path / "inputs"
    inp.mkdir()
    reserved = json.loads((ROOT / "papers/st_live_reference/split_v1.json").read_text())[
        "H1_train"][:10]
    (inp / "t14_inputs.json").write_text(json.dumps({"reserved": reserved}))
    np.savez(inp / "t14_inputs.npz", anchor_axis=np.ones(8, np.float32))
    out, adapters = tmp_path / "out", tmp_path / "adapters"
    cmd = [sys.executable, str(SCRIPTS / "t15_driver.py"), "--out", str(out), "--t14-inputs",
           str(inp), "--adapters", str(adapters), "--dry-run", "--date-pin", "2026-10-21", *extra]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return out / "gpt_oss_20b"


@pytest.mark.slow
def test_dry_run_every_phase_on_a_tiny_gpt_oss(tmp_path):
    # most probable failure: a phase only the pod reaches (the merge saved for vLLM, the base
    # check, a vLLM engine on a local checkpoint, the capability scorer, the readouts' re-built
    # prompts) crashes hours into the session, or the arms' H1_held seeds differ (ΔP unpaired)
    d = _dry(tmp_path)
    rec = json.loads((d / "t15_record.json").read_text())
    names = T.plan(rec)
    assert all(rec["phases"][n]["finished"] for n in names)
    comp = rec["phases"]["beh_cap"]["comparator"]
    assert comp in ("beh", "beh_epoch3")
    assert names.index(f"hf_{comp}") < names.index("hf_anchor")
    for arm in ["none", *T.ARMS_ALL]:
        assert (d / "gen" / f"h1_{arm}.jsonl").exists(), arm
        assert (d / "cap" / f"{arm}.jsonl").exists(), arm
        assert (d / "readouts" / f"{arm}.jsonl").exists(), arm
    for arm in T.ARMS_ALL:
        assert (d / "spec" / arm / "dl_chat_neutral.jsonl").exists()
        has_panel = (d / "panel" / arm).exists()
        assert has_panel == (arm in {comp, "anchor"}), arm
    # the arms' H1_held and capability jobs carry the same seeds (paired against none)
    seeds = {}
    for arm in ["none", *T.ARMS_ALL]:
        jobs = [json.loads(x) for x in (d / "vllm" / f"{arm}.jobs.jsonl").open()]
        seeds[arm] = [(j["job_id"], j["seed"]) for j in jobs if not j["job_id"].startswith("lp|")]
    assert all(v == seeds["none"] for v in seeds.values())
    # the bf16 engine runs the local checkpoint with the registry's tokenizer
    eng = json.loads((d / "vllm" / "none.engine.json").read_text())
    assert eng["model"].endswith("merged/none") and eng["tokenizer"]
    mx = json.loads((d / "vllm" / "mxfp4_parity.engine.json").read_text())
    assert mx["model"] == eng["tokenizer"]
    assert {"band_tv", "base_tv", "base_check_ok"} <= set(rec["phases"]["base_vllm"])
    capr = [json.loads(x) for x in (d / "cap" / "none.jsonl").open()]
    assert {"pred", "gold", "correct", "final_text"} <= set(capr[0])
    assert any(x["answered"] for x in capr)  # the stub's traces close: the scorer ran
    # the readouts ran a forward per picked rollout (2 of rr per scenario) for every arm
    for arm in ["none", *T.ARMS_ALL]:
        ro = [json.loads(x) for x in (d / "readouts" / f"{arm}.jsonl").open()]
        assert ro and all(np.isfinite(x["margin_anchor_axis"]) for x in ro), arm
        assert any(x["post_letter_logp"] is not None for x in ro), arm
    for arm in T.ARMS_ALL:
        assert "kept_share" in rec["phases"][f"hf_{arm}"]["merge_check"]


@pytest.mark.slow
def test_dry_run_failed_base_check_takes_every_arm_to_hf(tmp_path):
    # most probable failure: the HF fallback (reached only when the bf16 engine misreads GPT-OSS)
    # crashes on the pod, generates the brake's 94 scenarios before ΔP's 38, or leaves a 42 GB
    # merged checkpoint per adapter on the pod's disk
    d = _dry(tmp_path, "--dry-fail-base")
    rec = json.loads((d / "t15_record.json").read_text())
    names = T.plan(rec)
    assert all(rec["phases"][n]["finished"] for n in names)
    assert rec["phases"]["base_vllm"]["base_check_ok"] is False
    engaged = set(json.loads((DATA / "t15_engaged_held_v1.json").read_text())["ids"])
    assert (d / "gen" / "h1_none.parity_failed.jsonl").exists()
    for arm in ["none", *T.ARMS_ALL]:
        assert f"hfgen_{arm}" in names, arm
        rows = [json.loads(x) for x in (d / "gen" / f"h1_{arm}.jsonl").open()]
        assert rows[0]["scenario_id"] in engaged  # ΔP's scenarios generate first
        assert {r["engine"] for r in rows} == {"hf" if arm == "none" else "hf_adapter"}
        assert (d / "cap" / f"{arm}.jsonl").exists()
        assert (d / "readouts" / f"{arm}.jsonl").exists()
    for arm in T.ARMS_ALL:
        assert rec["phases"][f"vllm_{arm}"]["parity_ok"] is False
        assert not (d / "merged" / arm).exists(), arm
