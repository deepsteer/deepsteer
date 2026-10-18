"""Tiny random checkpoints of the three screen architectures, saved with each model's real pinned
tokenizer, for the T0.11 local tests (no weights downloaded; configs and tokenizers only)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
REG = yaml.safe_load((ROOT / "papers/kdg_panel/models.yaml").read_text())["steering"]
TEXT = dict(hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8)


def build(key: str, out: Path) -> Path:
    """Write a tiny ``key`` checkpoint (4 text layers) plus the real tokenizer to ``out``."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    spec = REG[key]
    tok = AutoTokenizer.from_pretrained(spec["repo"], revision=spec["revision"])
    cfg = AutoConfig.from_pretrained(spec["repo"], revision=spec["revision"]).to_dict()
    torch.manual_seed(0)
    if key == "olmo3_think":
        cfg.update(TEXT, num_hidden_layers=4, layer_types=cfg["layer_types"][:4])
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(**cfg))
    else:
        t = cfg["text_config"]
        t.update(TEXT, num_hidden_layers=4, layer_types=t["layer_types"][:4])
        for k in ("linear_num_key_heads", "linear_num_value_heads"):
            if k in t:
                t[k] = 2
        for k in ("linear_key_head_dim", "linear_value_head_dim"):
            if k in t:
                t[k] = 8
        if "mtp_num_hidden_layers" in t:
            t["mtp_num_hidden_layers"] = 0
        v = cfg["vision_config"]
        for k in ("depth", "num_hidden_layers"):
            if k in v:
                v[k] = 1
        if "layer_types" in v:
            v["layer_types"] = v["layer_types"][:1]
        for k, val in (("hidden_size", 32), ("intermediate_size", 64), ("out_hidden_size", 32),
                       ("num_heads", 2), ("num_attention_heads", 2), ("head_dim", 16)):
            if k in v:
                v[k] = val
        model = AutoModelForImageTextToText.from_config(AutoConfig.for_model(**cfg))
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out
