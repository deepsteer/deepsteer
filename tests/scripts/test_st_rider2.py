"""ST rider pod 2 (G1 items 3-4, Amendment 9; 2026-10-19): the fp32 head, the bias fold, the
readout-mode rule, the stack-parity reader and comparison, the MoE probe, and the driver's rider2
mode, on tiny checkpoints with the real tokenizers (CPU)."""

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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import screen_vllm as SV  # noqa: E402


@pytest.fixture(scope="module")
def tiny_oss(tmp_path_factory):
    pytest.importorskip("transformers")
    import st_tiny

    return st_tiny.build("gpt_oss_20b", tmp_path_factory.mktemp("ck") / "gpt_oss_20b")


def _letters(m, rows, lids):
    import torch

    out = []
    for x in rows:
        ids = torch.tensor([x])
        with torch.no_grad():
            o = m(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1)
        out.append(torch.log_softmax(o.logits[0, -1].float(), -1)[lids].numpy())
    return np.stack(out)


def test_bias_fold_equals_a_hook_at_the_block_output(tiny_oss, tmp_path):
    # most probable failure (G1 item 3): the fold writes the wrong tensor, or the router weights
    # do not sum to 1 so the block output gains a multiple of v, not v
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny_oss, dtype=torch.float32).eval()
    rows, lids = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]], [32, 33, 34, 35, 36]
    v = np.random.default_rng(1).standard_normal(m.config.hidden_size).astype(np.float32) * 3
    base = _letters(m, rows, lids)
    vt = torch.as_tensor(v)
    h = m.model.layers[1].register_forward_hook(
        lambda _m, _i, o: (o[0] + vt,) + tuple(o[1:]) if isinstance(o, tuple) else o + vt)
    hooked = _letters(m, rows, lids)
    h.remove()
    built = SV.build_folded(tiny_oss, 1, v, tmp_path / "fold")
    mf = AutoModelForCausalLM.from_pretrained(built["dst"], dtype=torch.float32).eval()
    folded = _letters(mf, rows, lids)
    assert np.abs(hooked - base).max() > 1e-3  # v moves the letters (the positive control)
    assert np.abs(hooked - folded).max() < 1e-4
    other = {p.name for p in Path(built["dst"]).iterdir() if p.is_symlink()}
    assert built["shard"] not in other and "config.json" in other


def test_head32_switches_and_restores(tiny_oss):
    # most probable failure: the fp32 head also upcasts tied embeddings, or does not restore
    import head32
    import torch
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(tiny_oss, dtype=torch.bfloat16).eval()
    ids = torch.tensor([[1, 2, 3, 4]])
    lo = m(input_ids=ids, logits_to_keep=1).logits
    assert head32.set_head_fp32(m, True)["final_norm"] == "model.norm"
    assert m(input_ids=ids, logits_to_keep=1).logits.dtype == torch.float32
    assert m.get_input_embeddings().weight.dtype == torch.bfloat16
    head32.set_head_fp32(m, False)
    assert torch.equal(m(input_ids=ids, logits_to_keep=1).logits, lo)


def test_readout_mode_rule():
    # most probable failure: batched admitted on a floor whose 2x would raise the bar past 0.05
    def dp(floor, meas, cal=True):
        return {"bf16": {"rung": {"kept": "fast"}}, "fp32": {
            "noise_floor_replicate_unpadded": floor, "max_abs_nats": meas, "calibrated": cal,
            "rung": {"kept": "fast + compile"}}}
    assert SV.readout_mode(dp(0.01, 0.03))["mode"] == "batched"
    assert SV.readout_mode(dp(0.01, 0.03))["readout_path"] == "fast + compile"
    assert SV.readout_mode(dp(0.03, 0.03))["mode"] == "single-row"  # 2 x 0.03 > 0.05
    assert SV.readout_mode(dp(0.01, 0.06))["mode"] == "single-row"
    assert SV.readout_mode(dp(0.01, 0.03, cal=False))["head"] == "bf16"


def test_stack_read_and_compare_same_stack_is_inside(tiny_oss, tmp_path):
    # most probable failure: the comparison is uncalibrated or reads a different row order; the
    # same stack read twice must sit inside the bar with its negative control outside it
    rows = {"ids": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], "letter_ids": [32, 33, 34, 35, 36]}
    (tmp_path / "rows.json").write_text(json.dumps(rows))
    for tag in ("a", "b"):
        r = subprocess.run([sys.executable, str(SCRIPTS / "stack_read.py"), "--repo",
                            str(tiny_oss), "--rows", str(tmp_path / "rows.json"), "--out",
                            str(tmp_path / f"{tag}.npz"), "--batch", "3"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
    a = dict(np.load(tmp_path / "a.npz"))
    rec = {"dose0_position": {"bf16": {"per_row": {"letter_logprobs_alone":
                                                   np.round(a["alone_bf16"], 4).tolist()}}}}
    (tmp_path / "rec.json").write_text(json.dumps(rec))
    r = subprocess.run([sys.executable, str(SCRIPTS / "stack_compare.py"), "--a",
                        str(tmp_path / "a.npz"), "--b", str(tmp_path / "b.npz"), "--out",
                        str(tmp_path / "v.json"), "--tie", str(tmp_path / "rec.json")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    v = json.loads((tmp_path / "v.json").read_text())
    for head in ("bf16", "fp32"):
        lad = v["ladders"][head]
        assert lad["inside"] and lad["max_abs_nats"] == 0.0 and lad["bar"] >= 0.05
    assert v["tie_ok"]


def test_moe_probe_records_each_path(tiny_oss, tmp_path):
    # most probable failure: a path that cannot load (kernels absent locally) crashes the pod
    # step instead of being recorded
    (tmp_path / "p.json").write_text(json.dumps({"ids": [[1, 2, 3, 4], [5, 6, 7]]}))
    for ex in ("eager", "batched_mm", "kernels"):
        out = tmp_path / f"{ex}.json"
        r = subprocess.run([sys.executable, str(SCRIPTS / "moe_probe.py"), "--repo",
                            str(tiny_oss), "--prompts", str(tmp_path / "p.json"), "--out",
                            str(out), "--experts", ex, "--batches", "1,2", "--decode-steps", "2"],
                           capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr[-2000:]
        rec = json.loads(out.read_text())
        assert rec.get("error") or len(rec["probes"]) == 2


@pytest.mark.parametrize("model", ("gpt_oss_20b", "muse_glimmer_30b"))
def test_rider2_dry_run(tmp_path, model):
    # the driver's rider2 mode end to end on the dry stub: no phase V, both heads at the dose-0
    # position, the readout-mode rule, the 64 stack-parity rows from the dose-0 renders
    pytest.importorskip("transformers")
    r = subprocess.run([sys.executable, str(SCRIPTS / "pod_st_screen.py"), "--model", model,
                        "--dry-run", "--date-pin", "2026-10-19", "--out", str(tmp_path),
                        "--rider2", "--effort", "medium"], capture_output=True, text=True,
                       cwd=ROOT)
    rec = json.loads((tmp_path / model / "screen_record.json").read_text())
    assert r.returncode == 0 and rec["status"] == "ok_rider2", r.stderr[-2000:]
    assert set(rec["dose0_position"]) >= {"candidate", "bf16", "fp32"}
    assert rec["readout_mode"]["mode"] in ("batched", "single-row")
    rows = json.loads((tmp_path / model / "stack" / "rows64.json").read_text())
    assert len(rows["letter_ids"]) == 5 and rows["effort"] == "medium"
    assert not list((tmp_path / model).glob("vllm/*"))
    assert (tmp_path / model / "stack" / "probe_prompts.json").exists() == (model == "gpt_oss_20b")
