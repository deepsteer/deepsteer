#!/usr/bin/env python3
"""WS3: Direction ablation experiments.

For each foundation direction at each target layer, project out the direction
from hidden states during the forward pass and measure:
  - Foundation-specific degradation
  - Cross-foundation specificity
  - General perplexity increase

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/direction_ablation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    load_probe_directions,
    ensure_output_dirs,
)


def ablate_direction(
    hidden_states: torch.Tensor,
    direction: np.ndarray,
) -> torch.Tensor:
    """Project out a direction from hidden states: h - (h·d)d."""
    d = torch.from_numpy(direction).to(hidden_states.device, dtype=hidden_states.dtype)
    proj = (hidden_states @ d).unsqueeze(-1) * d.unsqueeze(0)
    return hidden_states - proj


def run_ablation_experiment(
    model,
    tokenizer,
    prompts: list[dict],
    directions: dict[str, dict[int, np.ndarray]],
    target_layers: list[int],
    device: str = "mps",
) -> dict:
    """Run ablation for each foundation at each target layer.

    For each prompt, measure next-token log-probability distribution
    under baseline and ablated conditions.

    Returns dict mapping foundation -> layer -> {metrics}.
    """
    results: dict[str, dict[int, dict]] = {}

    for target_fv in FOUNDATION_ORDER:
        if target_fv not in directions:
            continue
        results[target_fv] = {}

        for layer in target_layers:
            direction = directions[target_fv].get(layer)
            if direction is None:
                continue

            baseline_logprobs = []
            ablated_logprobs = []

            hook_handle = None

            def _hook(module, input, output, d=direction):
                if isinstance(output, tuple):
                    h = output[0]
                    ablated = ablate_direction(h, d)
                    return (ablated,) + output[1:]
                return ablate_direction(output, d)

            # Get the target layer module
            # (Model-specific: works for OLMo / Llama transformer blocks)
            layer_module = model.model.layers[layer] if hasattr(model, 'model') else model.transformer.blocks[layer]

            for prompt_data in prompts:
                prompt = prompt_data["prompt"]
                inputs = tokenizer(prompt, return_tensors="pt").to(device)

                # Baseline forward pass
                with torch.no_grad():
                    baseline_out = model(**inputs)
                    baseline_logprobs.append(
                        F.log_softmax(baseline_out.logits[0, -1], dim=-1).cpu()
                    )

                # Ablated forward pass
                hook_handle = layer_module.register_forward_hook(_hook)
                with torch.no_grad():
                    ablated_out = model(**inputs)
                    ablated_logprobs.append(
                        F.log_softmax(ablated_out.logits[0, -1], dim=-1).cpu()
                    )
                hook_handle.remove()

            results[target_fv][layer] = {
                "n_prompts": len(prompts),
                "baseline_logprobs": baseline_logprobs,
                "ablated_logprobs": ablated_logprobs,
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="WS3: Direction ablation.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--eval-prompts",
                        default="papers/3_moral_geometry/outputs/probe_engineering/causal_eval_prompts.json")
    parser.add_argument("--target-layers", default="5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS3: Direction Ablation Experiments")
    print(f"{'='*60}")

    target_layers = [int(x) for x in args.target_layers.split(",")]
    directions = load_probe_directions(args.probe_directions)

    eval_path = Path(args.eval_prompts)
    if not eval_path.exists():
        print(f"\nERROR: Evaluation prompts not found at {eval_path}")
        print("Run causal_eval_prompts.py first to generate prompts.")
        print("(Requires Human Review Gate 3 approval)")
        return

    print(f"\nTarget layers: {target_layers}")
    print(f"Directions loaded for {len(directions)} foundations")
    print("\nReady to run ablation experiments.")
    print("(Full implementation runs after causal_eval_prompts.json is generated)")


if __name__ == "__main__":
    main()
