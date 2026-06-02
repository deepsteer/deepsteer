"""Direction ablation experiments.

Projects out concept directions from hidden states during the forward pass
and measures specificity: does ablating a direction affect on-target
continuations more than off-target ones?

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/direction_ablation.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)


def ablate_and_measure(
    model: WhiteBoxModel,
    direction: np.ndarray,
    layer: int,
    prompt: str,
    continuations: list[str],
) -> dict[str, float]:
    """Ablate a direction at one layer and measure log-probability change.

    Projects out *direction* from hidden states at *layer* during the forward
    pass, then compares continuation log-probabilities to baseline.

    Args:
        model: WhiteBoxModel with hook support.
        direction: Unit direction vector to ablate.
        layer: Layer index.
        prompt: Text prompt.
        continuations: Continuation tokens to measure.

    Returns:
        ``{continuation: delta_logprob}`` for each continuation.
    """
    tokenizer = model.tokenizer
    device = model._device  # noqa: SLF001

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_out = model.model(**inputs)
        bl_logprobs = F.log_softmax(baseline_out.logits[0, -1], dim=-1)

    with model.ablate_direction(layer, direction):
        with torch.no_grad():
            ablated_out = model.model(**inputs)
            ab_logprobs = F.log_softmax(ablated_out.logits[0, -1], dim=-1)

    deltas: dict[str, float] = {}
    for cont_text in continuations:
        cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
        if cont_tokens:
            token_id = cont_tokens[0]
            deltas[cont_text] = float(ab_logprobs[token_id] - bl_logprobs[token_id])
    return deltas


def ablation_sweep(
    model: WhiteBoxModel,
    directions: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    prompts: list[dict],
    target_key: str = "target_foundation",
    continuation_key: str = "continuations",
) -> dict[str, dict[int, dict]]:
    """Run ablation for each direction at each target layer.

    For each (ablated_group, layer) pair, measures on-target and off-target
    log-probability deltas to compute specificity.

    Args:
        model: WhiteBoxModel.
        directions: ``group → layer → unit direction vector``.
        layers: Layer indices to test.
        prompts: List of dicts, each with keys:

            - ``"prompt"``: text prompt.
            - *target_key*: which group this prompt targets.
            - *continuation_key*: list of ``{"text": str, "is_target": bool}``
              continuation specifications.

        target_key: Key in prompt dicts for the target group label.
        continuation_key: Key in prompt dicts for continuations.

    Returns:
        ``results[ablated_group][layer] = {on_target_mean_delta, off_target_mean_delta, specificity}``.
    """
    results: dict[str, dict[int, dict]] = {}

    for ablated_group in directions:
        results[ablated_group] = {}

        for layer in layers:
            direction = directions[ablated_group].get(layer)
            if direction is None:
                continue

            deltas_on: list[float] = []
            deltas_off: list[float] = []

            for prompt_data in prompts:
                prompt_text = prompt_data["prompt"]
                prompt_target = prompt_data[target_key]
                target_conts = [
                    c["text"] for c in prompt_data[continuation_key]
                    if c.get("is_target")
                ]
                if not target_conts:
                    continue

                delta_map = ablate_and_measure(
                    model, direction, layer, prompt_text, target_conts,
                )
                for delta in delta_map.values():
                    if prompt_target == ablated_group:
                        deltas_on.append(delta)
                    else:
                        deltas_off.append(delta)

            results[ablated_group][layer] = {
                "on_target_mean_delta": round(
                    float(np.mean(deltas_on)), 6
                ) if deltas_on else 0.0,
                "off_target_mean_delta": round(
                    float(np.mean(deltas_off)), 6
                ) if deltas_off else 0.0,
                "specificity": round(
                    float(np.mean(deltas_on)) - float(np.mean(deltas_off)), 6
                ) if deltas_on and deltas_off else 0.0,
                "n_on_target": len(deltas_on),
                "n_off_target": len(deltas_off),
            }

    return results
