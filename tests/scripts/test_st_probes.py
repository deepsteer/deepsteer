"""Amendment 7 item 4: the HF memory-and-throughput probe (engine_probes.py)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "papers" / "st_live_reference" / "scripts"))
sys.path.insert(0, str(ROOT / "papers" / "kdg_panel" / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # st_tiny
import engine_probes as E  # noqa: E402


def test_kv_bytes_counts_sliding_layers_at_their_window():
    # most probable failure: sliding-window layers priced as full attention (GPT-OSS alternates
    # 128-token windows), which would overstate the cap's KV by about 2x and shrink the batch
    cfg = SimpleNamespace(num_key_value_heads=8, head_dim=64, num_hidden_layers=4,
                          layer_types=["sliding_attention", "full_attention"] * 2,
                          sliding_window=128)
    per_layer_token = 2 * 8 * 64 * 2
    assert E.kv_bytes_per_token(cfg, 1000) == pytest.approx(per_layer_token * (2 * 1000 + 2 * 128))


def test_kv_bytes_skip_linear_attention_layers():
    cfg = SimpleNamespace(text_config=SimpleNamespace(
        num_key_value_heads=4, head_dim=256, num_hidden_layers=4,
        layer_types=["linear_attention"] * 3 + ["full_attention"], sliding_window=None))
    assert E.kv_bytes_per_token(cfg, 100) == pytest.approx(2 * 4 * 256 * 2 * 100)


def test_summarize_picks_largest_fitting_batch():
    probes = [{"batch": 132, "fits": True, "seconds_per_step": 1.4},
              {"batch": 264, "fits": False, "seconds_per_step": None, "oom": "x"}]
    s = E.summarize(probes)
    assert s["largest_fitting_batch"] == 132 and s["seconds_per_step_at_largest"] == 1.4


@pytest.mark.slow
def test_probe_runs_on_a_tiny_checkpoint(tmp_path):
    # the real generate path at two batch sizes on CPU (memory fields are CUDA-only)
    import st_tiny
    import torch
    from kdg_pod_lib import ModelWrapper

    ck = st_tiny.build("olmo3_think", tmp_path / "ck")
    m = ModelWrapper(str(ck), None, torch_dtype=torch.float32)
    ids = [m.tok.encode("hello there " * k, add_special_tokens=False) for k in (3, 5, 9)]
    for b in (2, 4):
        r = E.hf_batch_probe(m, ids, b, cap=16, decode_steps=4)
        # CPU timing of a tiny model is noisy; the slope estimator must at least be finite and the
        # longer decode must not be faster by more than noise
        assert r["oom"] is None and r["t_decode_2n"] > 0.5 * r["t_decode_n"]
