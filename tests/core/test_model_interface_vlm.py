"""WhiteBoxModel on vision-language wrappers (ST_PLAN Amendment 2 item 5).

Qwen3.8-27B (``qwen3_5``) and Muse Glimmer 30B (``muse_glimmer``) keep the text decoder at
``model.language_model.layers``; Muse also has ``model.vision_tower.layers``. Tiny random
configs, no downloads.
"""

from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    MuseGlimmerConfig,
    Qwen3_5Config,
)

from deepsteer.core.model_interface import WhiteBoxModel, _auto_model_class

_TEXT = dict(
    hidden_size=64, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, vocab_size=512,
)


def _tiny_muse() -> MuseGlimmerConfig:
    return MuseGlimmerConfig(
        text_config=dict(
            _TEXT, num_hidden_layers=4,
            layer_types=["sliding_attention"] * 3 + ["full_attention"],
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_attention_heads=2, num_hidden_layers=3,
            layer_types=["full_attention"] * 3,
        ),
    )


def _tiny_qwen() -> Qwen3_5Config:
    return Qwen3_5Config(
        text_config=dict(
            _TEXT, num_hidden_layers=4,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_num_key_heads=2, linear_num_value_heads=2,
            linear_key_head_dim=16, linear_value_head_dim=16,
        ),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
                           out_hidden_size=64),
    )


def _wrap(model: torch.nn.Module) -> WhiteBoxModel:
    wb = WhiteBoxModel.__new__(WhiteBoxModel)
    wb._model = model
    return wb


def test_muse_resolves_text_layers_not_vision_tower():
    # most probable failure: the probe finds model.vision_tower.layers (3) instead of the
    # 4 text decoder layers, so every hook lands in the ViT
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(_tiny_muse())
    wb = _wrap(model)
    assert wb._detect_n_layers() == 4
    assert wb._get_layer_module(2) is model.model.language_model.layers[2]


def test_muse_loads_with_image_text_to_text():
    # Muse has no causal-LM mapping; AutoModelForCausalLM would raise on its config
    assert _auto_model_class(_tiny_muse()) is AutoModelForImageTextToText


def test_qwen_loads_text_tower_via_causal_lm():
    cfg = _tiny_qwen()
    assert _auto_model_class(cfg) is AutoModelForCausalLM
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    wb = _wrap(model)
    assert wb._detect_n_layers() == 4
    assert wb._get_layer_module(3) is model.model.layers[3]


def test_qwen_vlm_wrapper_resolves_language_model_layers():
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(_tiny_qwen())
    wb = _wrap(model)
    assert wb._detect_n_layers() == 4
    assert wb._get_layer_module(0) is model.model.language_model.layers[0]


def test_hook_on_resolved_layer_fires_on_text_forward():
    # the resolved module is the one a text forward pass runs through
    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_config(_tiny_muse()).eval()
    seen = []
    handle = _wrap(model)._get_layer_module(1).register_forward_hook(
        lambda m, i, o: seen.append(o[0] if isinstance(o, tuple) else o)
    )
    with torch.no_grad():
        model(input_ids=torch.tensor([[1, 2, 3, 4]]))
    handle.remove()
    assert len(seen) == 1 and seen[0].shape[-1] == 64
