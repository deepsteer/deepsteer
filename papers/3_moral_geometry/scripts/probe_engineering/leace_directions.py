#!/usr/bin/env python3
"""WS2: LEACE direction extraction.

Compute foundation directions using LEACE (Belrose et al., NeurIPS 2023) —
the Fisher LDA direction (Sigma^{-1} @ (mu_1 - mu_0)) at each layer.
Compare with probe-weight and mean-diff directions.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/leace_directions.py
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
    compute_cosine_matrix,
    compute_effective_dimensionality,
    compute_mean_diff_directions,
    full_geometric_analysis,
    load_model_and_collect_activations,
    load_probe_directions,
    pair_accuracy,
    ensure_output_dirs,
    OUTPUT_DIR,
)


def compute_leace_directions(
    all_activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    foundation_indices: dict[str, list[int]],
) -> dict[str, dict[int, np.ndarray]]:
    """Compute LEACE (Fisher LDA) directions: Sigma^{-1} @ (mu_1 - mu_0).

    In the binary case, the LEACE eraser direction simplifies to the
    Fisher LDA direction. This is the optimal linear direction for
    separating two class-conditional Gaussians under shared covariance.
    """
    directions: dict[str, dict[int, np.ndarray]] = {}

    for fv in FOUNDATION_ORDER:
        if fv not in foundation_indices:
            continue
        pair_indices = foundation_indices[fv]
        directions[fv] = {}

        for layer in range(n_layers):
            X, _ = all_activations[layer]
            moral_rows = [pi * 2 for pi in pair_indices]
            neutral_rows = [pi * 2 + 1 for pi in pair_indices]

            moral_acts = X[moral_rows].numpy().astype(np.float64)
            neutral_acts = X[neutral_rows].numpy().astype(np.float64)

            mu_1 = moral_acts.mean(axis=0)
            mu_0 = neutral_acts.mean(axis=0)
            diff = mu_1 - mu_0

            # Pooled covariance (regularized for numerical stability)
            all_acts = np.vstack([moral_acts, neutral_acts])
            mu_pooled = all_acts.mean(axis=0)
            centered = all_acts - mu_pooled
            n_samples = centered.shape[0]
            Sigma = (centered.T @ centered) / (n_samples - 1)

            # Regularize: Sigma + lambda * I
            reg = 1e-4 * np.trace(Sigma) / Sigma.shape[0]
            Sigma_reg = Sigma + reg * np.eye(Sigma.shape[0])

            # Fisher LDA direction: Sigma^{-1} @ (mu_1 - mu_0)
            direction = np.linalg.solve(Sigma_reg, diff)
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            directions[fv][layer] = direction

    return directions


def main() -> None:
    parser = argparse.ArgumentParser(description="WS2: LEACE direction extraction.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=40)
    args = parser.parse_args()

    output_dir, figures_dir = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS2: LEACE Direction Extraction")
    print(f"{'='*60}")

    # Load model + collect activations
    all_train, all_test, dataset, n_layers, foundation_indices = (
        load_model_and_collect_activations(
            model_name=args.model,
            device=args.device,
            target_per_foundation=args.target_per_foundation,
            collect_test=True,
        )
    )

    # Compute directions from all three methods
    print("\nComputing LEACE directions...")
    leace_dirs = compute_leace_directions(all_train, n_layers, foundation_indices)

    print("Computing mean-diff directions...")
    md_dirs = compute_mean_diff_directions(all_train, n_layers, foundation_indices)

    print("Loading probe-weight directions...")
    pw_dirs = load_probe_directions(args.probe_directions)

    # Pairwise alignment between the three methods
    print("\n--- 3-Method Direction Alignment ---")
    print(f"{'Foundation':<14s}  {'LEACE↔PW':>9s}  {'LEACE↔MD':>9s}  {'MD↔PW':>8s}")
    print("-" * 50)

    alignment: dict[str, dict] = {}
    for fv in FOUNDATION_ORDER:
        leace_pw, leace_md, md_pw = [], [], []
        for layer in range(n_layers):
            l = leace_dirs.get(fv, {}).get(layer)
            m = md_dirs.get(fv, {}).get(layer)
            p = pw_dirs.get(fv, {}).get(layer)
            if l is None or m is None or p is None:
                continue
            leace_pw.append(abs(float(np.dot(l, p))))
            leace_md.append(abs(float(np.dot(l, m))))
            md_pw.append(abs(float(np.dot(m, p))))

        alignment[fv] = {
            "leace_vs_pw": round(float(np.mean(leace_pw)), 4) if leace_pw else 0,
            "leace_vs_md": round(float(np.mean(leace_md)), 4) if leace_md else 0,
            "md_vs_pw": round(float(np.mean(md_pw)), 4) if md_pw else 0,
        }
        print(f"  {FOUNDATION_SHORT[fv]:<12s}  {alignment[fv]['leace_vs_pw']:>9.4f}  "
              f"{alignment[fv]['leace_vs_md']:>9.4f}  {alignment[fv]['md_vs_pw']:>8.4f}")

    # Test-set pair accuracy for each method
    if all_test is not None:
        print("\n--- Test-Set Pair Accuracy (per method) ---")
        test_foundation_idx: dict[str, list[int]] = {}
        from collections import defaultdict
        test_foundation_idx = defaultdict(list)
        for i, pair in enumerate(dataset.test):
            test_foundation_idx[pair.foundation.value].append(i)

        print(f"{'Foundation':<14s}  {'LEACE':>7s}  {'Mean-diff':>9s}  {'Probe-wt':>8s}")
        print("-" * 45)
        for fv in FOUNDATION_ORDER:
            test_idx = test_foundation_idx.get(fv, [])
            if not test_idx:
                continue
            accs = {}
            for label, dirs in [("leace", leace_dirs), ("md", md_dirs), ("pw", pw_dirs)]:
                layer_accs = []
                for layer in range(n_layers):
                    d = dirs.get(fv, {}).get(layer)
                    if d is None:
                        continue
                    X_test, _ = all_test[layer]
                    layer_accs.append(pair_accuracy(d, X_test, test_idx))
                accs[label] = max(layer_accs) if layer_accs else 0
            print(f"  {FOUNDATION_SHORT[fv]:<12s}  {accs['leace']:>7.3f}  "
                  f"{accs['md']:>9.3f}  {accs['pw']:>8.3f}")

    # Geometric analysis with LEACE directions
    print("\n--- Geometric Analysis (LEACE) ---")
    geo_results: dict[str, dict] = {}
    for layer in range(n_layers):
        result = full_geometric_analysis(leace_dirs, layer)
        if result is not None:
            geo_results[str(layer)] = result
            mc = result["mean_cosine_similarity"]
            ed = result["effective_dimensionality"]
            mft = result["dendrogram"]["mft_match"]
            print(f"  Layer {layer:2d}: cos={mc:.4f}, dim={ed}, MFT={mft}")

    # Save results
    results = {
        "analysis": "leace_directions",
        "n_layers": n_layers,
        "alignment": alignment,
        "geometry": geo_results,
    }
    out_path = output_dir / "leace_directions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
