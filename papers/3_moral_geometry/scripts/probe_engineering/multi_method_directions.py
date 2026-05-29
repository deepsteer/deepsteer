#!/usr/bin/env python3
"""WS2: Multi-method direction comparison.

Run all four direction extraction methods (probe-weight, mean-diff, LEACE,
RepE PCA) on the same dataset and produce a unified comparison: pairwise
cosine alignment, per-method pair accuracy, bootstrap stability, and
geometric analysis.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/multi_method_directions.py
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
    compute_mean_diff_directions,
    full_geometric_analysis,
    load_model_and_collect_activations,
    load_probe_directions,
    pair_accuracy,
    ensure_output_dirs,
)
from leace_directions import compute_leace_directions
from concept_directions import compute_repe_directions


def bootstrap_direction_stability(
    all_activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    n_layers: int,
    foundation_indices: dict[str, list[int]],
    direction_fn,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, dict[int, float]]:
    """Compute bootstrap stability: mean cosine between bootstrap-resampled directions."""
    rng = np.random.RandomState(seed)
    stability: dict[str, dict[int, float]] = {}

    for fv in FOUNDATION_ORDER:
        if fv not in foundation_indices:
            continue
        pair_indices = foundation_indices[fv]
        n_pairs = len(pair_indices)
        stability[fv] = {}

        for layer in range(n_layers):
            directions = []
            for _ in range(n_bootstrap):
                boot_idx = rng.choice(pair_indices, size=n_pairs, replace=True)
                boot_foundation_idx = {fv: list(range(n_pairs))}
                # Rebuild activations for bootstrap sample
                X, _ = all_activations[layer]
                boot_moral = torch.stack([X[pi * 2] for pi in boot_idx])
                boot_neutral = torch.stack([X[pi * 2 + 1] for pi in boot_idx])
                boot_X = torch.zeros(n_pairs * 2, X.shape[1])
                for j in range(n_pairs):
                    boot_X[j * 2] = boot_moral[j]
                    boot_X[j * 2 + 1] = boot_neutral[j]
                boot_acts = {0: (boot_X, None)}
                d = direction_fn(boot_acts, 1, boot_foundation_idx)
                if fv in d and 0 in d[fv]:
                    directions.append(d[fv][0])

            if len(directions) < 2:
                stability[fv][layer] = 0.0
                continue

            # Mean pairwise cosine between bootstrap directions
            mat = np.stack(directions)
            cos_mat = mat @ mat.T
            n = cos_mat.shape[0]
            upper = [abs(cos_mat[i, j]) for i in range(n) for j in range(i + 1, n)]
            stability[fv][layer] = float(np.mean(upper))

    return stability


def main() -> None:
    parser = argparse.ArgumentParser(description="WS2: Multi-method direction comparison.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--target-per-foundation", type=int, default=40)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    args = parser.parse_args()

    output_dir, figures_dir = ensure_output_dirs()

    print(f"{'='*60}")
    print("WS2: Multi-Method Direction Comparison")
    print(f"{'='*60}")

    all_train, all_test, dataset, n_layers, foundation_indices = (
        load_model_and_collect_activations(
            model_name=args.model,
            device=args.device,
            target_per_foundation=args.target_per_foundation,
            collect_test=True,
        )
    )

    # Compute all four direction sets
    print("\nComputing directions (4 methods)...")
    pw_dirs = load_probe_directions(args.probe_directions)
    md_dirs = compute_mean_diff_directions(all_train, n_layers, foundation_indices)
    leace_dirs = compute_leace_directions(all_train, n_layers, foundation_indices)
    repe_dirs = compute_repe_directions(all_train, n_layers, foundation_indices)

    methods = {
        "probe_weight": pw_dirs,
        "mean_diff": md_dirs,
        "leace": leace_dirs,
        "repe": repe_dirs,
    }

    # Pairwise alignment between all methods (4x4 at each layer)
    print("\n--- Pairwise Alignment (mean |cos| across layers) ---")
    method_names = list(methods.keys())
    alignment_matrix: dict[str, dict[str, float]] = {}
    for m1 in method_names:
        alignment_matrix[m1] = {}
        for m2 in method_names:
            cosines = []
            for fv in FOUNDATION_ORDER:
                for layer in range(n_layers):
                    d1 = methods[m1].get(fv, {}).get(layer)
                    d2 = methods[m2].get(fv, {}).get(layer)
                    if d1 is not None and d2 is not None:
                        cosines.append(abs(float(np.dot(d1, d2))))
            alignment_matrix[m1][m2] = round(float(np.mean(cosines)), 4) if cosines else 0.0

    # Print alignment matrix
    header = f"{'':14s}" + "".join(f"{m:>12s}" for m in method_names)
    print(header)
    for m1 in method_names:
        row = f"  {m1:<12s}" + "".join(f"{alignment_matrix[m1][m2]:>12.4f}" for m2 in method_names)
        print(row)

    # Bootstrap stability for each training-data-dependent method
    print(f"\n--- Bootstrap Stability ({args.n_bootstrap} resamples) ---")
    # Note: probe_weight directions come from a saved file, so we skip bootstrap for them
    stability_results: dict[str, dict[str, dict[int, float]]] = {}
    for method_name, direction_fn in [
        ("mean_diff", compute_mean_diff_directions),
        ("leace", compute_leace_directions),
        ("repe", compute_repe_directions),
    ]:
        print(f"  Bootstrapping {method_name}...")
        stability_results[method_name] = bootstrap_direction_stability(
            all_train, n_layers, foundation_indices,
            direction_fn, n_bootstrap=args.n_bootstrap,
        )

    # Summary: which method is most stable?
    for method_name in stability_results:
        all_stabilities = [
            stability_results[method_name][fv][layer]
            for fv in FOUNDATION_ORDER if fv in stability_results[method_name]
            for layer in range(n_layers) if layer in stability_results[method_name].get(fv, {})
        ]
        mean_stab = np.mean(all_stabilities) if all_stabilities else 0
        print(f"  {method_name}: mean stability = {mean_stab:.4f}")

    # Save results
    results = {
        "analysis": "multi_method_direction_comparison",
        "n_layers": n_layers,
        "methods": method_names,
        "alignment_matrix": alignment_matrix,
        "stability": {
            method: {
                fv: {str(l): round(v, 4) for l, v in layers.items()}
                for fv, layers in fnd.items()
            }
            for method, fnd in stability_results.items()
        },
    }
    out_path = output_dir / "multi_method_directions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
