"""ST T0.6: the training harness (deepsteer.steering.objective_lora + scripts/st_train.py) on tiny
checkpoints of every architecture, and (slow) end to end on OLMo-2-1B with a synthetic target."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
torch = pytest.importorskip("torch")
pytest.importorskip("peft")
import reasoning_formats as rf  # noqa: E402
import st_train as ST  # noqa: E402

from deepsteer.steering import objective_lora as OL  # noqa: E402


def _wb(path):
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    return WhiteBoxModel(str(path), access_tier=AccessTier.WEIGHTS, device="cpu",
                         torch_dtype=torch.float32)


@pytest.fixture(scope="module")
def ck(tmp_path_factory):
    import st_tiny

    d = tmp_path_factory.mktemp("ck")
    return {k: st_tiny.build(k, d / k) for k in ("olmo3_think", "qwen38_27b", "muse_glimmer_30b",
                                                  "gpt_oss_20b")}


def test_lora_targets_stay_in_the_text_decoder(ck):
    # most probable failure (Amendment 2 item 5): the VLM's vision tower, which also has q/k/v_proj,
    # gets adapters; or Qwen3.8's DeltaNet projections / GPT-OSS's expert tensors get none
    mods, params = OL.lora_targets(_wb(ck["muse_glimmer_30b"]))
    assert mods and not any("vision" in m or "visual" in m for m in mods)
    assert all(".language_model.layers." in m for m in mods)
    mq, _ = OL.lora_targets(_wb(ck["qwen38_27b"]))
    assert any(m.endswith("in_proj_qkv") for m in mq) and any(m.endswith("q_proj") for m in mq)
    mo, po = OL.lora_targets(_wb(ck["gpt_oss_20b"]))
    assert set(po) == set(OL.EXPERT_PARAMS) and any(m.endswith("q_proj") for m in mo)


def _trace(m, n_gen=10):
    tok = m.tokenizer
    prompt = tok.encode("Decide. ", add_special_tokens=False)
    gen = tok.encode("First thought. Then a second one; and more words here.",
                     add_special_tokens=False)[:n_gen]
    return prompt, gen


def test_reference_is_the_adapter_disabled_model(ck):
    # most probable failure: the DPO / KL reference is the adapted model itself (zero gradient) or
    # a second copy; at initialization (LoRA B = 0) the margin and the KL must both be exactly 0
    m = _wb(ck["olmo3_think"])
    OL.inject_lora(m, rank=4)
    p, g = _trace(m)
    loss, info = OL.dpo_loss(m, p, g, g[::-1])
    assert info["dpo_margin"] == pytest.approx(0.0, abs=1e-6)
    assert float(OL.kl_general(m, p + g)) == pytest.approx(0.0, abs=1e-6)
    assert loss.requires_grad


def test_anchor_and_self_objectives_move_their_targets(ck):
    # most probable failure: the representation term is read at the wrong hidden-state index or
    # with masks that select nothing, so training reduces only the KL term
    m = _wb(ck["olmo3_think"])
    p, g = _trace(m)
    fmt = rf.ThinkTextFormat(key="x", effort_kwarg=None, effort_ladder=(None,), dose0=())
    import steering_gate as SG

    am, km = ST.trace_masks(fmt, m.tokenizer, p, g, "onset", SG.punct_mask(m.tokenizer))
    assert sum(am) == len(g) and not any(km)  # every trace token is inside onset here
    hid = m.model.config.hidden_size
    axis = np.random.default_rng(0).standard_normal(hid).astype(np.float32)
    axis /= np.linalg.norm(axis)
    tr = [{"ids": p + g, "anchor_mask": am, "kl_mask": km, "scenario_id": "s"}]
    rec = ST.run(m, "anchor", layer=2, traces=tr, axis=axis, tau=0.5, max_steps=12, lr=5e-3)
    rep = [s["rep"] for s in rec["steps"]]
    assert rep[-1] < rep[0]
    assert rec["steps"][-1]["info"]["margin_mean"] > rec["steps"][0]["info"]["margin_mean"]
    m2 = _wb(ck["olmo3_think"])
    tgt = {"s": np.ones(hid, np.float32)}
    rec2 = ST.run(m2, "self", layer=2, traces=tr, targets=tgt, sigma=np.ones(hid, np.float32),
                  max_steps=12, lr=5e-3)
    assert rec2["steps"][-1]["rep"] < rec2["steps"][0]["rep"]


def test_merged_adapter_reproduces_the_adapted_model(ck, tmp_path):
    # the evaluation path: every adapter is evaluated merged at the base's precision; the merged
    # checkpoint must compute what the PEFT model computes
    m = _wb(ck["olmo3_think"])
    p, g = _trace(m)
    hid = m.model.config.hidden_size
    axis = np.ones(hid, np.float32) / np.sqrt(hid)
    tr = [{"ids": p + g, "anchor_mask": [False] * len(p) + [True] * len(g),
           "kl_mask": [False] * (len(p) + len(g)), "scenario_id": "s"}]
    ST.run(m, "anchor", layer=2, traces=tr, axis=axis, tau=0.9, max_steps=3, lr=1e-2)
    ids = torch.tensor([p + g])
    with torch.no_grad():
        want = m.model(input_ids=ids).logits
    m.model.save_pretrained(str(tmp_path / "adapter"))
    base = _wb(ck["olmo3_think"])
    OL.merge_bf16(base, str(tmp_path / "adapter"), str(tmp_path / "merged"))
    got_m = _wb(tmp_path / "merged")
    with torch.no_grad():
        got = got_m.model(input_ids=ids).logits
    assert torch.allclose(want, got, atol=1e-4)


def test_tau_axis_sigma_and_pairs():
    # most probable failures: defer rollouts enter DPO pairs; the anchor axis points from protect
    # to rule; chat_sigma is accepted without its provenance (Amendment 1 item 3)
    rng = np.random.default_rng(0)
    assert ST.compute_tau(np.arange(101)) == pytest.approx(75.0)
    v = np.linalg.qr(rng.standard_normal((8, 2)))[0].T
    prot = rng.standard_normal((20, 8)) + 3 * v[0]
    rule = rng.standard_normal((20, 8)) - 3 * v[0]
    ax = ST.anchor_axis(v, prot, rule)
    assert ST.margins(prot, ax).mean() > ST.margins(rule, ax).mean()
    with pytest.raises(ValueError, match="provenance"):
        ST.chat_sigma(prot, {"model": "x"})
    rows = [{"scenario_id": "a", "action": a, "prompt_ids": [1], "gen_ids": [i]}
            for i, a in enumerate(["protect", "rule", "defer", "protect", "defer"])]
    pairs, info = ST.dpo_pairs(rows)
    assert info["n_pairs"] == 1 and pairs[0]["chosen"] == [0] and pairs[0]["rejected"] == [1]
    assert not info["meets_floor"]


@pytest.mark.slow
def test_end_to_end_on_olmo2_1b_with_a_synthetic_target(tmp_path):
    # T0.6's shakedown: on-policy traces from the gate's own loop, tau from the unadapted margins,
    # the anchor objective with the general-slice KL, then the merged-adapter readout
    import steering_gate as SG
    from kdg_pod_lib import ModelWrapper

    try:
        mw = ModelWrapper("allenai/OLMo-2-0425-1B", None, torch_dtype=torch.float32)
    except OSError:
        pytest.skip("OLMo-2-0425-1B not available offline")
    fmt = rf.ThinkTextFormat(key="x", effort_kwarg=None, effort_ladder=(None,), dose0=())
    tok = mw.tok
    prompts = [tok.encode(f"Case {i}: should the clerk follow the rule? Reasoning:",
                          add_special_tokens=False) for i in range(4)]
    rows = SG.generate(mw, prompts, fmt=fmt, cap=20, seeds=[1, 2, 3, 4], batch=4)
    punct = SG.punct_mask(tok)
    tr = []
    for p, r in zip(prompts, rows):
        am, km = ST.trace_masks(fmt, tok, p, r["gen_ids"], "constant", punct)
        tr.append({"ids": p + r["gen_ids"], "anchor_mask": am, "kl_mask": km, "scenario_id": "s"})
    wb = mw.wb
    hid = wb.model.config.hidden_size
    axis = np.random.default_rng(0).standard_normal(hid).astype(np.float32)
    axis /= np.linalg.norm(axis)

    def mean_margin(model, trs):
        out = []
        for t in trs:
            with torch.no_grad():
                hs = model.model(input_ids=torch.tensor([t["ids"]], device=model.model.device),
                                 output_hidden_states=True).hidden_states[8][0]
            h = hs[torch.tensor(t["anchor_mask"], device=hs.device)].float().cpu().numpy()
            out.append(ST.margins(h, axis).mean())
        return float(np.mean(out))
    before = mean_margin(wb, tr[2:])
    tau = before + 0.05
    general = [tok.encode("The weather today is mild and the river is calm.",
                          add_special_tokens=False)]
    rec = ST.run(wb, "anchor", layer=8, traces=tr[:2], general=general, axis=axis, tau=tau,
                 max_steps=15, lr=5e-4)
    after = mean_margin(wb, tr[2:])
    assert after > before  # held-out traces move toward the target
    assert rec["steps"][-1]["kl"] < 0.5
    ST.save_adapter(wb, tmp_path / "anchor", rec, {"axis": "synthetic", "tau": tau})
    base = ModelWrapper("allenai/OLMo-2-0425-1B", None, torch_dtype=torch.float32).wb
    OL.merge_bf16(base, str(tmp_path / "anchor"), str(tmp_path / "merged"))
    merged = ModelWrapper(str(tmp_path / "merged"), None, torch_dtype=torch.float32).wb
    assert mean_margin(merged, tr[2:]) == pytest.approx(after, abs=1e-3)
