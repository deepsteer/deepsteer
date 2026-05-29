#!/usr/bin/env python3
"""A.2: Project dilemma directions onto the full 5D moral subspace.

Instead of projecting each dilemma direction onto only the 2D subspace
of its two component foundations, project onto the full subspace spanned
by all 6 foundation directions (which has effective dimensionality ~5).

This tests whether dilemma representations are compositional over the
*full* moral vocabulary, not just the two directly conflicting foundations.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/full_subspace_projection.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    DILEMMA_PAIRS,
    DILEMMA_PAIR_KEYS,
    load_probe_directions,
    orthonormal_basis,
    subspace_membership,
)


def null_subspace_membership(hidden_dim: int, subspace_dim: int, n_samples: int = 10000, seed: int = 42) -> dict:
    """Expected membership of random unit vectors projected onto a random subspace."""
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_samples):
        random_basis = np.linalg.qr(rng.randn(hidden_dim, subspace_dim))[0].T
        random_dir = rng.randn(hidden_dim)
        random_dir /= np.linalg.norm(random_dir)
        scores.append(subspace_membership(random_dir, random_basis))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "expected_analytic": subspace_dim / hidden_dim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="5D subspace projection of dilemma directions.")
    parser.add_argument("--foundation-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--dilemma-directions",
                        default="papers/3_moral_geometry/outputs/dilemma_probing/dilemma_probe_directions.npz")
    parser.add_argument("--probing-results",
                        default="papers/3_moral_geometry/outputs/dilemma_probing/dilemma_probing.json")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/probe_engineering")
    parser.add_argument("--figures-dir",
                        default="papers/3_moral_geometry/outputs/figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    foundation_npz = np.load(args.foundation_directions)
    dilemma_npz = np.load(args.dilemma_directions)

    with open(args.probing_results) as f:
        probing_data = json.load(f)

    n_layers = probing_data["n_layers"]
    hidden_dim = int(foundation_npz[f"{FOUNDATION_ORDER[0]}_layer0"].shape[0])

    print(f"{'='*60}")
    print("A.2: Full 5D Subspace Projection Analysis")
    print(f"{'='*60}")
    print(f"Model: {n_layers} layers, {hidden_dim}-dim representations")

    # Load 2D membership scores from existing probing results for comparison
    two_d_memberships = {}
    for pk in DILEMMA_PAIR_KEYS:
        pair_data = probing_data.get("per_foundation_pair", {}).get(pk, {})
        peak_layer = pair_data.get("peak_subspace_layer")
        if peak_layer is not None:
            layer_data = pair_data.get("per_layer", {}).get(str(peak_layer), {})
            two_d_memberships[pk] = layer_data.get("subspace_membership", 0)

    # Compute null distribution for 5D subspace
    print("\nComputing null distribution (5D)...")
    null_5d = null_subspace_membership(hidden_dim, 5, n_samples=10000)
    print(f"  Null 5D: mean={null_5d['mean']:.6f}, expected={null_5d['expected_analytic']:.6f}")

    null_2d = null_subspace_membership(hidden_dim, 2, n_samples=10000)
    print(f"  Null 2D: mean={null_2d['mean']:.6f}, expected={null_2d['expected_analytic']:.6f}")

    # Per-layer analysis
    per_layer_results = {}
    five_d_means = []
    two_d_means = []

    for layer in range(n_layers):
        # Load 6 foundation directions at this layer
        foundation_dirs = []
        for fv in FOUNDATION_ORDER:
            key = f"{fv}_layer{layer}"
            if key in foundation_npz:
                d = foundation_npz[key]
                d = d / (np.linalg.norm(d) + 1e-12)
                foundation_dirs.append(d)

        if len(foundation_dirs) < 6:
            continue

        foundation_matrix = np.stack(foundation_dirs)
        basis_5d = orthonormal_basis(foundation_matrix)
        actual_dim = basis_5d.shape[0]

        # Project each dilemma direction onto the 5D subspace
        five_d_scores = {}
        two_d_scores_at_layer = {}

        for pk in DILEMMA_PAIR_KEYS:
            dkey = f"dilemma_{pk}_layer{layer}"
            if dkey not in dilemma_npz:
                continue
            dilemma_dir = dilemma_npz[dkey]
            dilemma_dir = dilemma_dir / (np.linalg.norm(dilemma_dir) + 1e-12)

            five_d_scores[pk] = subspace_membership(dilemma_dir, basis_5d)

            # Also compute 2D membership at this layer for comparison
            a, b = pk.split("-")
            fv_a = next(f for f in FOUNDATION_ORDER if f.startswith(a))
            fv_b = next(f for f in FOUNDATION_ORDER if f.startswith(b))
            idx_a = FOUNDATION_ORDER.index(fv_a)
            idx_b = FOUNDATION_ORDER.index(fv_b)
            basis_2d = orthonormal_basis(foundation_matrix[[idx_a, idx_b]])
            two_d_scores_at_layer[pk] = subspace_membership(dilemma_dir, basis_2d)

        if five_d_scores:
            mean_5d = float(np.mean(list(five_d_scores.values())))
            mean_2d = float(np.mean(list(two_d_scores_at_layer.values())))
            five_d_means.append(mean_5d)
            two_d_means.append(mean_2d)

            per_layer_results[layer] = {
                "subspace_dim": actual_dim,
                "mean_5d_membership": round(mean_5d, 6),
                "mean_2d_membership": round(mean_2d, 6),
                "ratio_5d_to_2d": round(mean_5d / (mean_2d + 1e-12), 3),
                "per_pair_5d": {k: round(v, 6) for k, v in five_d_scores.items()},
                "per_pair_2d": {k: round(v, 6) for k, v in two_d_scores_at_layer.items()},
            }

    # Summary
    peak_5d_layer = max(per_layer_results, key=lambda l: per_layer_results[l]["mean_5d_membership"])
    mean_5d_overall = float(np.mean(five_d_means))
    mean_2d_overall = float(np.mean(two_d_means))

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Mean 5D membership (across layers): {mean_5d_overall:.4f}")
    print(f"Mean 2D membership (across layers): {mean_2d_overall:.4f}")
    print(f"Ratio (5D/2D): {mean_5d_overall / (mean_2d_overall + 1e-12):.2f}x")
    print(f"Peak 5D layer: {peak_5d_layer} "
          f"(5D={per_layer_results[peak_5d_layer]['mean_5d_membership']:.4f}, "
          f"2D={per_layer_results[peak_5d_layer]['mean_2d_membership']:.4f})")
    print(f"Null baselines: 5D={null_5d['mean']:.6f}, 2D={null_2d['mean']:.6f}")
    print(f"Ratio over null: 5D={mean_5d_overall / null_5d['mean']:.1f}x, "
          f"2D={mean_2d_overall / null_2d['mean']:.1f}x")

    # Per-pair breakdown at peak layer
    print(f"\nPer-pair at peak 5D layer ({peak_5d_layer}):")
    peak_data = per_layer_results[peak_5d_layer]
    for pk in DILEMMA_PAIR_KEYS:
        s5 = peak_data["per_pair_5d"].get(pk, 0)
        s2 = peak_data["per_pair_2d"].get(pk, 0)
        ratio = s5 / (s2 + 1e-12)
        print(f"  {pk:25s}: 5D={s5:.4f}  2D={s2:.4f}  ratio={ratio:.2f}x")

    # Save results
    results = {
        "analysis": "full_5d_subspace_projection",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "mean_5d_membership": round(mean_5d_overall, 6),
        "mean_2d_membership": round(mean_2d_overall, 6),
        "ratio_5d_to_2d": round(mean_5d_overall / (mean_2d_overall + 1e-12), 3),
        "peak_5d_layer": peak_5d_layer,
        "null_5d": null_5d,
        "null_2d": null_2d,
        "per_layer": {str(k): v for k, v in per_layer_results.items()},
    }

    out_path = output_dir / "full_subspace_projection.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # Generate figure: 2D vs 5D membership comparison
    generate_figure(results, figures_dir)


def generate_figure(results: dict, figures_dir: Path) -> None:
    """Bar chart comparing 2D vs 5D subspace membership with null baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = results["n_layers"]
    layers = sorted(int(k) for k in results["per_layer"].keys())

    five_d = [results["per_layer"][str(l)]["mean_5d_membership"] for l in layers]
    two_d = [results["per_layer"][str(l)]["mean_2d_membership"] for l in layers]

    null_5d = results["null_5d"]["mean"]
    null_2d = results["null_2d"]["mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel a: layer-wise comparison
    ax1.plot(layers, five_d, "o-", color="#1E88E5", linewidth=2, markersize=5, label="5D subspace")
    ax1.plot(layers, two_d, "s-", color="#E53935", linewidth=2, markersize=5, label="2D subspace")
    ax1.axhline(null_5d, color="#1E88E5", linestyle=":", alpha=0.5, label=f"5D null ({null_5d:.4f})")
    ax1.axhline(null_2d, color="#E53935", linestyle=":", alpha=0.5, label=f"2D null ({null_2d:.4f})")
    ax1.set_xlabel("Layer", fontsize=11)
    ax1.set_ylabel("Mean Subspace Membership", fontsize=11)
    ax1.set_title("(a) Subspace Membership Across Layers", fontsize=12, fontweight="bold")
    ax1.set_xticks(layers)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel b: summary bar chart
    peak_layer = results["peak_5d_layer"]
    peak_data = results["per_layer"][str(peak_layer)]
    categories = ["2D\nComponent", "5D\nFull Moral", "2D Null", "5D Null"]
    values = [peak_data["mean_2d_membership"], peak_data["mean_5d_membership"], null_2d, null_5d]
    colors = ["#E53935", "#1E88E5", "#FFCDD2", "#BBDEFB"]

    x = np.arange(len(categories))
    bars = ax2.bar(x, values, color=colors, alpha=0.85, width=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylabel("Subspace Membership Score", fontsize=11)
    ax2.set_title(f"(b) Peak Layer ({peak_layer}) Summary", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.002,
                 f"{val:.4f}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Dilemma Direction Projection: 2D Component vs. 5D Full Moral Subspace",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_5d_subspace_projection.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_5d_subspace_projection.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figures_dir / 'fig_5d_subspace_projection.png'}")


if __name__ == "__main__":
    main()
