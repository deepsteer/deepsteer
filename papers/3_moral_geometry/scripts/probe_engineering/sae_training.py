#!/usr/bin/env python3
"""WS5: SAE training on OLMo-2 1B residual stream.

Train a sparse autoencoder (width 16,384 = 8x expansion) on cached
residual stream activations at layers 7-10. Uses SAELens if available,
falls back to minimal custom implementation.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/sae_training.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shared import ensure_output_dirs


SAE_WIDTH = 16_384
HIDDEN_DIM = 2048
TARGET_LAYERS = [7, 8, 9, 10]


def main() -> None:
    parser = argparse.ArgumentParser(description="WS5: SAE training.")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-layers", default=",".join(str(l) for l in TARGET_LAYERS))
    parser.add_argument("--sae-width", type=int, default=SAE_WIDTH)
    parser.add_argument("--n-tokens", type=int, default=100_000_000,
                        help="Tokens for activation caching (default 100M, full: 1B)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()
    target_layers = [int(x) for x in args.target_layers.split(",")]

    print(f"{'='*60}")
    print("WS5: SAE Training")
    print(f"{'='*60}")
    print(f"\nModel: {args.model}")
    print(f"Target layers: {target_layers}")
    print(f"SAE width: {args.sae_width} ({args.sae_width / HIDDEN_DIM:.0f}x expansion)")
    print(f"Training tokens: {args.n_tokens:,}")
    print(f"\nEstimated resources:")
    print(f"  Model (fp16): ~3 GB")
    print(f"  SAE (fp32): ~{args.sae_width * HIDDEN_DIM * 2 * 4 / 1e9:.1f} GB")
    print(f"  Activation cache: depends on batch size")
    print(f"\nPhases:")
    print(f"  1. Cache activations from Dolma/RedPajama (decoupled from model)")
    print(f"  2. Train SAE on cached activations")
    print(f"  3. Analyze learned features")
    print(f"\n(Implementation follows feasibility assessment)")


if __name__ == "__main__":
    main()
