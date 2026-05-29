#!/usr/bin/env python3
"""WS5: SAE moral feature identification.

After SAE training, identify features that activate selectively on moral
text and compare their decoder directions with probe-derived directions.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/sae_moral_features.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    load_probe_directions,
    ensure_output_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="WS5: SAE moral feature identification.")
    parser.add_argument("--sae-path", default=None, help="Path to trained SAE checkpoint")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS5: SAE Moral Feature Identification")
    print(f"{'='*60}")

    directions = load_probe_directions(args.probe_directions)
    print(f"Probe directions loaded for {len(directions)} foundations")

    if args.sae_path is None:
        print("\nNo SAE checkpoint provided.")
        print("Run sae_training.py first to train the SAE.")
        print("\nPipeline:")
        print("  1. Run moral dataset through SAE encoder")
        print("  2. Compute moral-neutral activation differential per feature")
        print("  3. Rank features by moral specificity")
        print("  4. Compare top-100 feature decoder directions with probe directions")
        return

    print(f"\nSAE path: {args.sae_path}")
    print("(Feature analysis implementation follows SAE training)")


if __name__ == "__main__":
    main()
