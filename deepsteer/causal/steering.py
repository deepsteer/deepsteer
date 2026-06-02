"""Steering vector injection experiments.

Adds scaled multiples of concept directions to hidden states and measures
dose-response curves for foundation-specific amplification.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/steering_injection.py
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from deepsteer.core.model_interface import WhiteBoxModel

logger = logging.getLogger(__name__)

DEFAULT_ALPHAS: list[float] = [1.0, 2.0, 5.0, 10.0, 20.0]


def inject_and_measure(
    model: WhiteBoxModel,
    direction: np.ndarray,
    layer: int,
    prompt: str,
    continuations: list[str],
    alpha: float = 1.0,
) -> dict[str, float]:
    """Inject a direction at one layer and measure log-probability change.

    Adds ``alpha * direction`` to hidden states at *layer*, then compares
    continuation log-probabilities to baseline.

    Args:
        model: WhiteBoxModel with hook support.
        direction: Unit direction vector to inject.
        layer: Layer index.
        prompt: Text prompt.
        continuations: Continuation tokens to measure.
        alpha: Injection strength.

    Returns:
        ``{continuation: delta_logprob}`` for each continuation.
    """
    tokenizer = model.tokenizer
    device = model._device  # noqa: SLF001

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        baseline_out = model.model(**inputs)
        bl_logprobs = F.log_softmax(baseline_out.logits[0, -1], dim=-1)

    with model.inject_direction(layer, direction, alpha=alpha):
        with torch.no_grad():
            injected_out = model.model(**inputs)
            inj_logprobs = F.log_softmax(injected_out.logits[0, -1], dim=-1)

    deltas: dict[str, float] = {}
    for cont_text in continuations:
        cont_tokens = tokenizer.encode(cont_text, add_special_tokens=False)
        if cont_tokens:
            token_id = cont_tokens[0]
            deltas[cont_text] = float(inj_logprobs[token_id] - bl_logprobs[token_id])
    return deltas


def steering_sweep(
    model: WhiteBoxModel,
    directions: dict[str, dict[int, np.ndarray]],
    layers: list[int],
    prompts: list[dict],
    alphas: list[float] | None = None,
    target_key: str = "target_foundation",
    continuation_key: str = "continuations",
) -> dict[str, dict[int, dict[float, dict]]]:
    """Run injection for each direction at each layer with multiple alphas.

    Args:
        model: WhiteBoxModel.
        directions: ``group → layer → unit direction vector``.
        layers: Layer indices to test.
        prompts: List of dicts (same format as :func:`ablation_sweep`).
        alphas: Injection strengths. Default: ``[1.0, 2.0, 5.0, 10.0, 20.0]``.
        target_key: Key in prompt dicts for the target group label.
        continuation_key: Key in prompt dicts for continuations.

    Returns:
        ``results[group][layer][alpha] = {on_target_mean_delta, off_target_mean_delta, specificity}``.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    results: dict[str, dict[int, dict[float, dict]]] = {}

    for injected_group in directions:
        results[injected_group] = {}

        for layer in layers:
            direction = directions[injected_group].get(layer)
            if direction is None:
                continue

            alpha_results: dict[float, dict] = {}
            for alpha in alphas:
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

                    delta_map = inject_and_measure(
                        model, direction, layer, prompt_text, target_conts, alpha,
                    )
                    for delta in delta_map.values():
                        if prompt_target == injected_group:
                            deltas_on.append(delta)
                        else:
                            deltas_off.append(delta)

                alpha_results[alpha] = {
                    "on_target_mean_delta": round(
                        float(np.mean(deltas_on)), 6
                    ) if deltas_on else 0.0,
                    "off_target_mean_delta": round(
                        float(np.mean(deltas_off)), 6
                    ) if deltas_off else 0.0,
                    "specificity": round(
                        float(np.mean(deltas_on)) - float(np.mean(deltas_off)), 6
                    ) if deltas_on and deltas_off else 0.0,
                }

            results[injected_group][layer] = alpha_results

    return results
