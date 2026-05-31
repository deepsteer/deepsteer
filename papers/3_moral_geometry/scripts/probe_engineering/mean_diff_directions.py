#!/usr/bin/env python3
"""B.1: Mean-difference foundation directions.

Compute foundation directions as the normalized mean activation difference
between moral and neutral texts, instead of using trained probe weight vectors.
Compare with probe-weight directions and run the full geometric analysis.

If directions agree (cosine > 0.9), the geometry is robust to direction
extraction method. If they disagree, investigate which is more stable.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/mean_diff_directions.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING,
    BINDING,
    compute_cosine_matrix,
    compute_effective_dimensionality,
    compute_mean_diff_directions,
    permutation_test_mft,
    check_dendrogram_mft,
    load_probe_directions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="B.1: Mean-difference directions.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/probe_engineering")
    parser.add_argument("--figures-dir",
                        default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

    print(f"{'='*60}")
    print("B.1: Mean-Difference Directions")
    print(f"{'='*60}")

    # Load dataset
    dataset = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # Build per-foundation index into training pairs
    foundation_indices: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.train):
        foundation_indices[pair.foundation.value].append(i)

    # Load model
    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    # Collect activations for all training pairs
    print("\nCollecting activations for training set...")
    t0 = time.time()
    all_train = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    print(f"Collected in {time.time() - t0:.1f}s")

    # Free model memory
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Compute mean-difference directions
    print("\nComputing mean-difference directions...")
    md_directions = compute_mean_diff_directions(all_train, n_layers, foundation_indices)

    # Load probe-weight directions for comparison
    pw_directions = load_probe_directions(args.probe_directions)

    # Compare directions: cosine similarity between mean-diff and probe-weight
    print("\n--- Direction Comparison ---")
    comparison: dict[str, dict[int, float]] = {}
    for fv in FOUNDATION_ORDER:
        comparison[fv] = {}
        cosines = []
        for layer in range(n_layers):
            pw = pw_directions.get(fv, {}).get(layer)
            if pw is None or fv not in md_directions:
                continue
            md = md_directions[fv][layer]
            cos = abs(float(np.dot(pw, md)))
            comparison[fv][layer] = cos
            cosines.append(cos)
        mean_cos = np.mean(cosines) if cosines else 0
        min_cos = np.min(cosines) if cosines else 0
        print(f"  {FOUNDATION_SHORT[fv]:12s}: mean cos = {mean_cos:.4f}, min = {min_cos:.4f}")

    # Full geometric analysis with mean-diff directions
    print("\n--- Geometric Analysis (Mean-Diff Directions) ---")
    geo_results: dict[str, dict] = {}
    md_mean_cosines = {}
    md_eff_dims = {}

    for layer in range(n_layers):
        cos_sim = compute_cosine_matrix(md_directions, layer)
        if cos_sim is None:
            continue
        n = 6
        upper_tri = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        mc = float(np.mean(upper_tri))
        md_mean_cosines[layer] = mc
        md_eff_dims[layer] = compute_effective_dimensionality(md_directions, layer)

        perm = permutation_test_mft(cos_sim)
        dendro = check_dendrogram_mft(cos_sim)

        geo_results[str(layer)] = {
            "mean_cosine_similarity": round(mc, 6),
            "effective_dimensionality": md_eff_dims[layer],
            "permutation_test_p": round(perm["p_value"], 6),
            "mft_dendrogram_match": dendro["mft_match"],
            "dendrogram_left": dendro["left"],
            "dendrogram_right": dendro["right"],
        }

    pw_mean_cosines = {}
    for layer in range(n_layers):
        cos_sim = compute_cosine_matrix(pw_directions, layer)
        if cos_sim is not None:
            upper_tri = [cos_sim[i, j] for i in range(6) for j in range(i + 1, 6)]
            pw_mean_cosines[layer] = float(np.mean(upper_tri))

    peak_md = min(md_mean_cosines, key=md_mean_cosines.get)
    peak_pw = min(pw_mean_cosines, key=pw_mean_cosines.get)
    print(f"\nPeak separation (mean-diff): layer {peak_md} (cos = {md_mean_cosines[peak_md]:.4f})")
    print(f"Peak separation (probe-wt):  layer {peak_pw} (cos = {pw_mean_cosines[peak_pw]:.4f})")

    # Effective dimensionality comparison
    md_dims = [md_eff_dims[l] for l in range(n_layers) if l in md_eff_dims]
    print(f"Effective dim (mean-diff): {md_dims}")

    # MFT dendrogram match at peak
    peak_result = geo_results[str(peak_md)]
    print(f"MFT match at peak (mean-diff): {peak_result['mft_dendrogram_match']}")
    print(f"  Left:  {peak_result['dendrogram_left']}")
    print(f"  Right: {peak_result['dendrogram_right']}")
    print(f"  Permutation p = {peak_result['permutation_test_p']:.4f}")

    # Save results
    results = {
        "analysis": "mean_diff_directions",
        "n_layers": n_layers,
        "direction_comparison": {
            fv: {str(k): round(v, 6) for k, v in comparison[fv].items()}
            for fv in FOUNDATION_ORDER if fv in comparison
        },
        "mean_diff_geometry": geo_results,
        "probe_weight_peak_layer": peak_pw,
        "mean_diff_peak_layer": peak_md,
        "mean_diff_peak_cosine": round(md_mean_cosines[peak_md], 6),
        "probe_weight_peak_cosine": round(pw_mean_cosines[peak_pw], 6),
    }

    out_path = output_dir / "mean_diff_directions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Generate comparison figure
    generate_figure(md_mean_cosines, pw_mean_cosines, md_eff_dims, comparison, n_layers, figures_dir)


def generate_figure(
    md_cosines: dict, pw_cosines: dict, md_dims: dict,
    comparison: dict, n_layers: int, figures_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel a: mean cosine comparison
    ax = axes[0]
    ax.plot(layers, [pw_cosines.get(l, 0) for l in layers],
            "o-", color="#1E88E5", linewidth=2, markersize=5, label="Probe weight")
    ax.plot(layers, [md_cosines.get(l, 0) for l in layers],
            "s-", color="#E53935", linewidth=2, markersize=5, label="Mean difference")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    ax.set_title("(a) Geometric Agreement", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel b: effective dimensionality
    ax = axes[1]
    ax.plot(layers, [md_dims.get(l, 5) for l in layers],
            "s-", color="#E53935", linewidth=2, markersize=5, label="Mean difference")
    ax.axhline(5, color="#1E88E5", linestyle="--", linewidth=1.5, alpha=0.7, label="Probe weight (=5)")
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Effective Dimensionality", fontsize=11)
    ax.set_title("(b) Dimensionality", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0.5, 6.5)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel c: direction alignment
    ax = axes[2]
    colors = {
        "care_harm": "#E53935", "fairness_cheating": "#1E88E5",
        "liberty_oppression": "#43A047", "loyalty_betrayal": "#FB8C00",
        "authority_subversion": "#8E24AA", "sanctity_degradation": "#00ACC1",
    }
    for fv in FOUNDATION_ORDER:
        if fv in comparison:
            vals = [comparison[fv].get(l, 0) for l in layers]
            ax.plot(layers, vals, "o-", color=colors[fv], linewidth=1.5,
                    markersize=4, label=FOUNDATION_SHORT[fv])
    ax.axhline(0.9, color="#9E9E9E", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("|cos(probe, mean-diff)|", fontsize=11)
    ax.set_title("(c) Direction Alignment", fontsize=12, fontweight="bold")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle("B.1: Mean-Difference vs. Probe-Weight Directions",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_b1_mean_diff_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_b1_mean_diff_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figures_dir / 'fig_b1_mean_diff_comparison.png'}")


if __name__ == "__main__":
    main()
