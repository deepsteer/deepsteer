"""T1.6 on Muse (scripts/t16_driver.py, scripts/analyze_t16.py; Amendments 37-39): the matched
pairs, the in-place merge, the fallback rule, the branches, the runner's order and sync, and a dry
run of every step on a tiny Muse with the fallback forced."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "papers" / "st_live_reference" / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
import analyze_t16 as A  # noqa: E402
import t16_driver as D  # noqa: E402

ST = ROOT / "papers" / "st_live_reference"
REP = ST / "outputs" / "st2mr" / "muse_glimmer_30b"


def test_design_constants():
    # most probable failure (Amendment 37): base trains before care (no steps to reuse), or the
    # arms differ in seed or budget
    assert D.ARMS == ("care", "base") and D.SEED == 105 and D.BUDGET_H == 2.0
    assert D.BASE_BAR == 0.10 and D.PARITY_MARGIN == 0.05 and D.PAIRS_PER_SCENARIO == 2
    assert D.PIN == "2026-10-23" and D.BLOCK == 26


def _row(sid, k, status):
    return {"scenario_id": sid, "rollout": k, "norm_status": status, "gen_ids": [100 + k]}


def test_harm_pairs_differ_only_in_the_chosen_trace():
    # most probable failure: the arms reject different rule traces or get different pair counts,
    # so the teacher is not the only difference
    care = ([_row("S1", k, "violating") for k in range(4)] + [_row("S2", 0, "violating")]
            + [_row("S3", k, "violating") for k in range(3)])
    none = ([_row("S1", 4, "violating"), _row("S1", 5, "consistent"), _row("S1", 6, "consistent"),
             _row("S1", 7, "violating"), _row("S1", 8, "consistent")]
            + [_row("S2", 1, "consistent")]  # S2: no base protect trace, drops from both arms
            + [_row("S3", 4, "violating"), _row("S3", 5, "neutral")])  # S3: no rule trace
    out, info = D.harm_pairs(care, none)
    assert info["per_scenario"] == {"S1": 2, "S2": 0, "S3": 0} and info["n_pairs"] == 2
    assert info["care_alone_pairs"] == 3  # S1 2 + S2 1: what per-scenario matching costs care
    assert [p["rejected"] for p in out["care"]] == [p["rejected"] for p in out["base"]]
    assert [p["chosen"] for p in out["care"]] == [[100], [101]]
    assert [p["chosen"] for p in out["base"]] == [[104], [107]]


def test_match_keys():
    # most probable failure: a delta lands on the vision tower's same-named tensor, or two adapter
    # modules map to one tensor
    deltas = {"model.language_model.layers.0.self_attn.q_proj": None,
              "model.language_model.layers.1.mlp.down_proj": None}
    keys = ["model.language_model.layers.0.self_attn.q_proj.weight",
            "language_model.model.layers.1.mlp.down_proj.weight",
            "model.vision_tower.layers.1.mlp.down_proj.weight"]
    m = D.match_keys(deltas, keys)
    assert m["model.language_model.layers.0.self_attn.q_proj"] == keys[0]
    assert m["model.language_model.layers.1.mlp.down_proj"] == keys[1]
    with pytest.raises(ValueError):
        D.match_keys({"model.language_model.layers.2.mlp.up_proj": None}, keys)


def _checkpoint(d: Path, dtype) -> dict:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)
    t = {"model.language_model.layers.0.self_attn.q_proj.weight":
         torch.randn(8, 6, generator=g).to(dtype),
         "model.language_model.layers.0.input_layernorm.weight": torch.ones(6).to(dtype),
         "model.vision_tower.layers.0.self_attn.q_proj.weight": torch.randn(8, 6, generator=g)
         .to(dtype)}
    d.mkdir(parents=True)
    save_file(t, str(d / "model.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text("{}")
    return t


def _adapter(d: Path, scale_ab: float) -> torch.Tensor:
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(1)
    A_ = torch.randn(4, 6, generator=g) * scale_ab
    B_ = torch.randn(8, 4, generator=g) * scale_ab
    stem = "base_model.model.model.language_model.layers.0.self_attn.q_proj"
    d.mkdir(parents=True)
    save_file({f"{stem}.lora_A.weight": A_, f"{stem}.lora_B.weight": B_},
              str(d / "adapter_model.safetensors"))
    (d / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    return (B_ @ A_) * 2.0


def test_write_merged_is_exact_in_fp32_and_leaves_the_rest_untouched(tmp_path):
    # most probable failure (the author's merge): the delta lands on the wrong tensor or with the
    # wrong scale, or an untouched tensor (the vision tower's same-named one) is rewritten
    from safetensors.torch import load_file

    base = _checkpoint(tmp_path / "src", torch.float32)
    delta = _adapter(tmp_path / "ad", 0.5)
    res = D.write_merged(tmp_path / "src", tmp_path / "ad", tmp_path / "dst")
    out = load_file(str(tmp_path / "dst" / "model.safetensors"))
    k = "model.language_model.layers.0.self_attn.q_proj.weight"
    assert res["ok"] and res["n_changed"] == 1 and res["keys_identical"]
    assert torch.allclose(out[k], base[k] + delta, atol=1e-6)
    v = "model.vision_tower.layers.0.self_attn.q_proj.weight"
    assert torch.equal(out[v], base[v])
    assert res["weight_kept_share"] == pytest.approx(1.0, abs=1e-5)
    assert (tmp_path / "dst" / "config.json").exists()


def test_bf16_rounding_shows_in_the_weight_kept_share(tmp_path):
    # most probable failure: a delta below bf16's resolution vanishes silently (the dry run's tiny
    # adapter did); the record must show it
    _checkpoint(tmp_path / "src", torch.bfloat16)
    _adapter(tmp_path / "ad", 0.003)
    res = D.write_merged(tmp_path / "src", tmp_path / "ad", tmp_path / "dst")
    assert res["ok"] and res["weight_kept_share"] < 0.9


def test_fallback_rule():
    # most probable failure: an arm on HF is compared with none on vLLM (rows of two engines)
    class C(dict):
        pass
    rec = {"phases": {"base_hf": {"base_check_ok": True},
                      "vllm_arm_care": {"parity_ok": True}, "vllm_arm_base": {"parity_ok": False}}}
    c = C()
    orig = D.phase
    try:
        D.phase = lambda _c, name: rec["phases"].get(name) or {}
        assert not D.needs_hf(c, "care") and D.needs_hf(c, "base") and D.needs_hf(c, "none")
        rec["phases"]["vllm_arm_base"]["parity_ok"] = True
        assert not D.needs_hf(c, "none")
        rec["phases"]["base_hf"]["base_check_ok"] = False
        assert all(D.needs_hf(c, a) for a in ("care", "base", "none"))
    finally:
        D.phase = orig
    assert A.engine_of(rec, "care") == "hf"


def test_branches():
    # most probable failure: a teacher difference is read as branch A without its CI above 0, or a
    # mixed outcome is forced into a registered branch
    def arm(cat):
        return {"category_record": cat}
    ok, no = "H1 criterion met", "partial"
    assert A.branch({"care": arm(no), "base": arm(no)}, {"ci95": [0.02, 0.2]}).startswith("A")
    assert A.branch({"care": arm(ok), "base": arm(ok)}, {"ci95": [-0.05, 0.1]}).startswith("B")
    assert A.branch({"care": arm(no), "base": arm(no)}, {"ci95": [-0.05, 0.1]}).startswith("C")
    assert A.branch({"care": arm(ok), "base": arm(no)},
                    {"ci95": [-0.05, 0.1]}).startswith("unregistered")


@pytest.mark.skipif(not (REP / "gen" / "arm_care_03.jsonl").exists(),
                    reason="the replicate's rows are not local")
def test_prompts_rerender_to_the_replicates_hashes():
    # most probable failure: the pairs' prompts are rendered under another pin, effort or
    # permutation than the replicate's rollouts (render_prompts stops on any mismatch)
    import pod_st_screen as P
    from reasoning_formats import build_format
    from transformers import AutoTokenizer

    from deepsteer.kdg.schema import load_scenario_dir

    spec = P.registry()[D.KEY]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    Hs, _ = load_scenario_dir([P.H1_SET])
    c = {"spec": spec, "tok": tok, "fmt": build_format(D.KEY, tok, spec), "out": Path("/tmp"),
         "H": {s.id: s for s in Hs}, "layer": D.BLOCK + 1}
    rows = [json.loads(x) for _, x in zip(range(48), (REP / "gen" / "arm_care_03.jsonl").open())]
    got = D.render_prompts(c, rows)
    assert len(got) >= 48


def test_remote_runs_t16_in_order():
    # most probable failure: base trains before care, or vLLM reads a merged copy before its gate
    sh = (ST / "runpod" / "remote_st_screen.sh").read_text()
    assert re.search(r'STEPS" = "t16" \]; then\n(?:  #.*\n)*  MODELS=.*\n  SESSION_MAX_HOURS='
                     r'"\$\{SESSION_MAX_HOURS:-9\.5\}"', sh)
    blk = sh.split('if [ "$STEPS" = "t16" ]; then\n    DEADLINE')[1][:2200]
    order = ["--step rr_gen", "--step base_vllm", "--step base_hf", "--step train",
             "--step hf_arm", "--step vllm_arm", "--step hfgen"]
    assert [blk.index(x) for x in order] == sorted(blk.index(x) for x in order)
    assert "for arm in care base; do   # care first" in blk
    assert "test_t16.py" in sh.split("GATE_TESTS=")[1].split("\n")[0]
    sync = (ST / "runpod" / "sync_t16.txt").read_text().split()
    assert sync == ["outputs/st2mr/muse_glimmer_30b/gen/arm_care_03.jsonl",
                    "outputs/st2mr/muse_glimmer_30b/gen/arm_none.jsonl"]


@pytest.mark.slow
def test_dry_run_every_step_with_the_fallback(tmp_path):
    # most probable failure: a step crashes on the pod's tiny Muse, the attach check rejects a
    # good adapter, or the fallback never runs
    if not (REP / "gen" / "arm_care_03.jsonl").exists():
        pytest.skip("the replicate's rows are not local")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    base = [sys.executable, str(SCRIPTS / "t16_driver.py"), "--out", str(tmp_path),
            "--replicate", str(REP), "--dry-run"]
    seq = [["--step", "rr_gen"], ["--step", "base_vllm"], ["--step", "base_hf"],
           ["--step", "train", "--arm", "care"], ["--step", "train", "--arm", "base"],
           ["--step", "hf_arm", "--arm", "care"], ["--step", "vllm_arm", "--arm", "care"],
           ["--step", "hf_arm", "--arm", "base"],
           ["--step", "vllm_arm", "--arm", "base", "--dry-fail-parity"],
           ["--step", "hfgen", "--arm", "care"], ["--step", "hfgen", "--arm", "base"],
           ["--step", "hfgen", "--arm", "none"]]
    for s in seq:
        r = subprocess.run(base + s, capture_output=True, text=True, cwd=ROOT, env=env)
        assert r.returncode == 0, " ".join(s) + "\n" + r.stdout[-2000:] + r.stderr[-2000:]
    d = tmp_path / D.KEY
    rec = json.loads((d / "t16_record.json").read_text())["phases"]
    assert rec["train_base"]["max_steps_run"] == rec["train_care"]["max_steps_run"]
    assert rec["hfgen_care"].get("skipped") and not rec["hfgen_base"].get("skipped")
    assert (d / "gen" / "h1_base.hf.jsonl").exists() and (d / "gen" / "h1_none.hf.jsonl").exists()
    assert rec["hf_arm_care"]["merge"]["saved"]["n_changed"] == \
        rec["hf_arm_care"]["merge"]["saved"]["n_deltas"] > 0
    held = sorted({json.loads(x)["scenario_id"] for x in (d / "gen" / "h1_none.jsonl").open()})
    r = json.loads((d / "t16_record.json").read_text())
    r["phases"]["base_hf"]["engaged_held"] = held  # a tiny model engages nothing: every path
    (d / "t16_record.json").write_text(json.dumps(r))
    res = A.read(d)
    assert res["status"] == "ok" and set(res["arms"]) == {"care", "base"}
    assert res["arms"]["care"]["engine"] == "vllm" and res["arms"]["base"]["engine"] == "hf"
    assert not np.isnan(res["arms"]["care"]["dP_record"]["mean"])
