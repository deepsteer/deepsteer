#!/usr/bin/env python3
"""WS4: Behavioral benchmarking.

Evaluate whether probe activations predict model performance on moral
reasoning benchmarks (ETHICS, MoralBench, MFV). Build a 6×N prediction
matrix mapping foundation directions to benchmark dimensions.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/behavioral_benchmarking.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    load_probe_directions,
    ensure_output_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="WS4: Behavioral benchmarking.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir, _ = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS4: Behavioral Benchmarking")
    print(f"{'='*60}")

    directions = load_probe_directions(args.probe_directions)
    print(f"Directions loaded for {len(directions)} foundations")
    print("\nBenchmark setup:")
    print("  - ETHICS (Hendrycks et al., 2021): justice, deontology, virtue, util, commonsense")
    print("  - MoralBench (Yu et al., 2024): MFT-tagged scenarios")
    print("  - Moral Foundations Vignettes (Clifford et al., 2015)")
    print("\nRequires benchmark datasets to be downloaded first.")
    print("(Implementation follows WS3 causal validation results)")


if __name__ == "__main__":
    main()
