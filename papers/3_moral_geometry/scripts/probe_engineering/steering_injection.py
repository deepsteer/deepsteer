#!/usr/bin/env python3
"""WS3: Steering vector injection experiments.

For each foundation direction at each target layer, add a scaled steering
vector to hidden states and measure foundation-specific amplification,
dose response, and coherence preservation.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/steering_injection.py
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

ALPHA_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0]


def inject_direction(
    hidden_states: torch.Tensor,
    direction: np.ndarray,
    alpha: float,
) -> torch.Tensor:
    """Add scaled steering vector: h + alpha * d."""
    d = torch.from_numpy(direction).to(hidden_states.device, dtype=hidden_states.dtype)
    return hidden_states + alpha * d.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="WS3: Steering injection.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--eval-prompts",
                        default="papers/3_moral_geometry/outputs/probe_engineering/causal_eval_prompts.json")
    parser.add_argument("--target-layers", default="5,6,7,8,9,10,11,12,13,14")
    parser.add_argument("--alphas", default=",".join(str(a) for a in ALPHA_VALUES))
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS3: Steering Vector Injection")
    print(f"{'='*60}")

    target_layers = [int(x) for x in args.target_layers.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    directions = load_probe_directions(args.probe_directions)

    print(f"Target layers: {target_layers}")
    print(f"Alpha values: {alphas}")
    print(f"Directions loaded for {len(directions)} foundations")
    print("\nReady to run injection experiments.")
    print("(Requires causal_eval_prompts.json — see causal_eval_prompts.py)")


if __name__ == "__main__":
    main()
