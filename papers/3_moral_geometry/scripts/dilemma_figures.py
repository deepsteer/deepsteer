#!/usr/bin/env python3
"""Script 8: Generate all figures for the dilemma extension.

Produces 6 figures:
    1. Subspace membership heatmap (15 pairs × 16 layers)
    2. Mean subspace membership across layers (with null baseline)
    3. Component balance at peak layer
    4. 21-direction dendrogram (6 foundation + 15 dilemma)
    5. Complexity-fragility gradient (pooled vs. single vs. dilemma)
    6. Shared-component similarity distributions

Usage:
    python papers/3_moral_geometry/scripts/dilemma_figures.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FOUNDATION_PAIRS = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]

DILEMMA_PAIR_KEYS = [f"{a}-{b}" for a, b in FOUNDATION_PAIRS]

FOUNDATION_SHORT = {
    "care_harm": "Care",
    "fairness_cheating": "Fairness",
    "liberty_oppression": "Liberty",
    "loyalty_betrayal": "Loyalty",
    "authority_subversion": "Authority",
    "sanctity_degradation": "Sanctity",
}

FOUNDATION_ORDER = list(FOUNDATION_SHORT.keys())


def generate_figure_1_subspace_heatmap(probing_data: dict, null_data: dict | None, figures_dir: Path) -> None:
    """Subspace membership by foundation pair × layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = probing_data["per_foundation_pair"]
    n_layers = probing_data["n_layers"]

    pair_keys = [pk for pk in DILEMMA_PAIR_KEYS if pk in pairs]
    matrix = np.zeros((len(pair_keys), n_layers))

    for i, pk in enumerate(pair_keys):
        for l in range(n_layers):
            layer_data = pairs[pk].get("per_layer", {}).get(str(l), {})
            matrix[i, l] = layer_data.get("subspace_membership", 0)

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    # Null threshold line
    if null_data:
        p95 = null_data.get("p95", 0.006)
        # Mark cells below null as gray hatching would be complex; use a contour
        ax.contour(matrix, levels=[p95], colors=["white"], linewidths=[1.5], linestyles=["--"])

    ax.set_xticks(range(n_layers))
    ax.set_xticklabels([str(l) for l in range(n_layers)], fontsize=9)
    ax.set_yticks(range(len(pair_keys)))
    ax.set_yticklabels(pair_keys, fontsize=9)
    ax.set_xlabel("Layer", fontsize=11)
    ax.set_ylabel("Foundation Pair", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Subspace Membership Score")
    ax.set_title("Dilemma Direction Subspace Membership\n"
                 "(fraction of variance explained by component foundation subspace)",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_subspace_heatmap.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_subspace_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 1: {figures_dir / 'fig_dilemma_subspace_heatmap.png'}")


def generate_figure_2_subspace_across_layers(probing_data: dict, null_data: dict | None, figures_dir: Path) -> None:
    """Mean subspace membership across layers with null baseline band."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = probing_data["per_foundation_pair"]
    n_layers = probing_data["n_layers"]

    pair_keys = [pk for pk in DILEMMA_PAIR_KEYS if pk in pairs]
    layers = list(range(n_layers))

    # Compute mean and std across foundation pairs at each layer
    means = []
    stds = []
    for l in layers:
        vals = []
        for pk in pair_keys:
            layer_data = pairs[pk].get("per_layer", {}).get(str(l), {})
            vals.append(layer_data.get("subspace_membership", 0))
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers, means, "o-", color="#E53935", linewidth=2, markersize=6, label="Mean membership")
    ax.fill_between(layers,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color="#E53935", label="±1 SD")

    if null_data:
        ax.axhline(y=null_data.get("mean", 0.001), color="#9E9E9E", linestyle=":",
                   linewidth=1, label=f"Null mean ({null_data.get('mean', 0):.4f})")
        ax.axhspan(0, null_data.get("p95", 0.006), alpha=0.1, color="#9E9E9E",
                   label=f"Null 95th ({null_data.get('p95', 0):.4f})")

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Subspace Membership Score", fontsize=12)
    ax.set_title("Mean Dilemma Subspace Membership Across Layers\n"
                 "(Does compositionality increase with depth?)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(layers)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_subspace_layers.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_subspace_layers.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 2: {figures_dir / 'fig_dilemma_subspace_layers.png'}")


def generate_figure_3_component_balance(probing_data: dict, figures_dir: Path) -> None:
    """Component balance at peak layer for each foundation pair."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = probing_data["per_foundation_pair"]
    pair_keys = [pk for pk in DILEMMA_PAIR_KEYS if pk in pairs]

    # Use peak subspace membership layer for each pair
    balances = []
    labels = []
    for pk in pair_keys:
        peak_layer = pairs[pk].get("peak_subspace_layer", 8)
        layer_data = pairs[pk].get("per_layer", {}).get(str(peak_layer), {})
        balance = layer_data.get("component_balance", 0.5)
        balances.append(balance)
        labels.append(pk)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    colors = ["#E53935" if b > 0.6 or b < 0.4 else "#43A047" for b in balances]

    ax.barh(x, balances, color=colors, alpha=0.8)
    ax.axvline(x=0.5, color="#9E9E9E", linestyle="--", linewidth=1.5, label="Perfect balance")
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Component Balance\n(0 = all foundation B, 0.5 = balanced, 1 = all foundation A)", fontsize=10)
    ax.set_title("Dilemma Direction Component Balance at Peak Layer\n"
                 "(near 0.5 = balanced activation of both foundations)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_balance.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_balance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 3: {figures_dir / 'fig_dilemma_balance.png'}")


def generate_figure_4_dendrogram(geometry_data: dict, figures_dir: Path) -> None:
    """21-direction dendrogram at peak layer."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    # Find the layer with the best shared-component difference
    per_layer = geometry_data.get("per_layer", {})
    best_layer = geometry_data.get("peak_shared_component_diff_layer")

    if best_layer is None:
        # Fall back to middle layer
        layers = sorted(int(k) for k in per_layer.keys())
        best_layer = layers[len(layers) // 2] if layers else 8

    layer_data = per_layer.get(str(best_layer), {})
    linkage_data = layer_data.get("linkage")
    combined_labels = layer_data.get("combined_labels")

    if linkage_data is None or combined_labels is None:
        print("  Figure 4: SKIPPED (no linkage data)")
        return

    Z = np.array(linkage_data)

    fig, ax = plt.subplots(figsize=(14, 7))

    # Color foundations vs dilemmas
    n_foundations = 6
    n_total = len(combined_labels)

    leaf_colors = {}
    for i in range(n_total):
        if i < n_foundations:
            leaf_colors[i] = "#1E88E5"  # Foundation directions
        else:
            leaf_colors[i] = "#E53935"  # Dilemma directions

    def _color_func(k):
        if k < n_total:
            return leaf_colors.get(k, "#666")
        return "#666"

    dendrogram(
        Z, labels=combined_labels, ax=ax,
        leaf_font_size=8, leaf_rotation=45,
        link_color_func=_color_func,
    )

    ax.set_ylabel("Ward Distance (1 - cosine similarity)", fontsize=11)
    ax.set_title(
        f"Hierarchical Clustering: 6 Foundation + 15 Dilemma Directions (Layer {best_layer})\n"
        f"Blue = foundation, Red = dilemma",
        fontsize=12, fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_dendrogram.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_dendrogram.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 4: {figures_dir / 'fig_dilemma_dendrogram.png'}")


def generate_figure_5_fragility_gradient(
    dilemma_fragility: dict,
    foundation_fragility_path: Path | None,
    figures_dir: Path,
) -> None:
    """Complexity-fragility gradient: pooled vs single-foundation vs dilemma."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Dilemma critical noise
    per_type = dilemma_fragility.get("per_dilemma_type", {})
    dilemma_criticals = [
        r["mean_critical_noise"] for r in per_type.values()
        if r.get("mean_critical_noise") is not None
    ]
    mean_dilemma = float(np.mean(dilemma_criticals)) if dilemma_criticals else None

    pooled_data = dilemma_fragility.get("pooled_dilemma", {})
    mean_pooled_dilemma = pooled_data.get("mean_critical_noise")

    # Try to load foundation fragility for comparison
    mean_foundation = None
    if foundation_fragility_path and foundation_fragility_path.exists():
        with open(foundation_fragility_path) as f:
            found_frag = json.load(f)
        per_foundation = found_frag.get("per_foundation", {})
        foundation_criticals = [
            r["mean_critical_noise"] for r in per_foundation.values()
            if r.get("mean_critical_noise") is not None
        ]
        mean_foundation = float(np.mean(foundation_criticals)) if foundation_criticals else None

    fig, ax = plt.subplots(figsize=(8, 6))

    categories = []
    values = []
    colors = []

    if mean_pooled_dilemma is not None:
        categories.append("Pooled\nDilemma")
        values.append(mean_pooled_dilemma)
        colors.append("#43A047")

    if mean_foundation is not None:
        categories.append("Single-\nFoundation")
        values.append(mean_foundation)
        colors.append("#1E88E5")

    if mean_dilemma is not None:
        categories.append("Per-Type\nDilemma")
        values.append(mean_dilemma)
        colors.append("#E53935")

    if not values:
        print("  Figure 5: SKIPPED (no fragility data)")
        return

    x = np.arange(len(categories))
    ax.bar(x, values, color=colors, alpha=0.8, width=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Mean Critical Noise (higher = more robust)", fontsize=11)
    ax.set_title("Complexity-Fragility Gradient",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(values):
        ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_fragility_gradient.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_fragility_gradient.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 5: {figures_dir / 'fig_dilemma_fragility_gradient.png'}")


def generate_figure_6_shared_component(geometry_data: dict, figures_dir: Path) -> None:
    """Shared-component vs no-shared-component cosine similarity distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Find best layer
    per_layer = geometry_data.get("per_layer", {})
    best_layer = geometry_data.get("peak_shared_component_diff_layer")
    if best_layer is None:
        layers = sorted(int(k) for k in per_layer.keys())
        best_layer = layers[len(layers) // 2] if layers else 8

    layer_data = per_layer.get(str(best_layer), {})
    sc = layer_data.get("shared_component_analysis", {})

    # We need the full cosine lists for histogram
    # If not in the truncated results, compute from the cosine matrix
    cosine_matrix = layer_data.get("dilemma_cosine_matrix")
    dilemma_labels = layer_data.get("dilemma_labels", [])

    if cosine_matrix is None or len(dilemma_labels) < 2:
        print("  Figure 6: SKIPPED (no cosine data)")
        return

    cos_mat = np.array(cosine_matrix)
    n = len(dilemma_labels)

    shared_cos = []
    no_shared_cos = []
    for i in range(n):
        parts_i = set(dilemma_labels[i].split("-"))
        for j in range(i + 1, n):
            parts_j = set(dilemma_labels[j].split("-"))
            val = cos_mat[i, j]
            if parts_i & parts_j:
                shared_cos.append(val)
            else:
                no_shared_cos.append(val)

    fig, ax = plt.subplots(figsize=(10, 6))

    bins = np.linspace(-0.5, 1.0, 30)
    ax.hist(shared_cos, bins=bins, alpha=0.6, color="#1E88E5", edgecolor="white",
            label=f"Shared component (n={len(shared_cos)}, mean={np.mean(shared_cos):.3f})")
    ax.hist(no_shared_cos, bins=bins, alpha=0.6, color="#E53935", edgecolor="white",
            label=f"No shared component (n={len(no_shared_cos)}, mean={np.mean(no_shared_cos):.3f})")

    ax.axvline(np.mean(shared_cos), color="#1E88E5", linestyle="--", linewidth=2)
    ax.axvline(np.mean(no_shared_cos), color="#E53935", linestyle="--", linewidth=2)

    ax.set_xlabel("Cosine Similarity Between Dilemma Directions", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Shared-Component Similarity Analysis (Layer {best_layer})\n"
                 f"Dilemma pairs sharing a foundation vs. not",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(figures_dir / "fig_dilemma_shared_component.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig_dilemma_shared_component.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure 6: {figures_dir / 'fig_dilemma_shared_component.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate all dilemma extension figures.")
    parser.add_argument("--probing-results",
                        default="papers/3_moral_geometry/outputs/dilemma_probing/dilemma_probing.json")
    parser.add_argument("--geometry-results",
                        default="papers/3_moral_geometry/outputs/dilemma_geometry/dilemma_geometry.json")
    parser.add_argument("--fragility-results",
                        default="papers/3_moral_geometry/outputs/dilemma_fragility/dilemma_fragility.json")
    parser.add_argument("--foundation-fragility",
                        default="papers/3_moral_geometry/outputs/exp7_fragility/exp7_olmo_fragility.json")
    parser.add_argument("--figures-dir",
                        default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print("GENERATING DILEMMA EXTENSION FIGURES")
    print(f"{'='*60}")

    # Load data
    probing_path = Path(args.probing_results)
    geometry_path = Path(args.geometry_results)
    fragility_path = Path(args.fragility_results)
    foundation_fragility_path = Path(args.foundation_fragility)

    probing_data = None
    geometry_data = None
    fragility_data = None

    if probing_path.exists():
        with open(probing_path) as f:
            probing_data = json.load(f)
        print(f"Loaded probing results: {probing_path}")
    else:
        print(f"WARNING: {probing_path} not found. Skipping probing figures.")

    if geometry_path.exists():
        with open(geometry_path) as f:
            geometry_data = json.load(f)
        print(f"Loaded geometry results: {geometry_path}")
    else:
        print(f"WARNING: {geometry_path} not found. Skipping geometry figures.")

    if fragility_path.exists():
        with open(fragility_path) as f:
            fragility_data = json.load(f)
        print(f"Loaded fragility results: {fragility_path}")
    else:
        print(f"WARNING: {fragility_path} not found. Skipping fragility figure.")

    null_data = probing_data.get("null_distribution") if probing_data else None

    # Figure 1: Subspace membership heatmap
    if probing_data:
        print("\nFigure 1: Subspace membership heatmap")
        generate_figure_1_subspace_heatmap(probing_data, null_data, figures_dir)

    # Figure 2: Mean subspace membership across layers
    if probing_data:
        print("Figure 2: Mean subspace membership across layers")
        generate_figure_2_subspace_across_layers(probing_data, null_data, figures_dir)

    # Figure 3: Component balance
    if probing_data:
        print("Figure 3: Component balance")
        generate_figure_3_component_balance(probing_data, figures_dir)

    # Figure 4: 21-direction dendrogram
    if geometry_data:
        print("Figure 4: 21-direction dendrogram")
        generate_figure_4_dendrogram(geometry_data, figures_dir)

    # Figure 5: Complexity-fragility gradient
    if fragility_data:
        print("Figure 5: Complexity-fragility gradient")
        generate_figure_5_fragility_gradient(
            fragility_data,
            foundation_fragility_path if foundation_fragility_path.exists() else None,
            figures_dir,
        )

    # Figure 6: Shared-component similarity
    if geometry_data:
        print("Figure 6: Shared-component similarity")
        generate_figure_6_shared_component(geometry_data, figures_dir)

    print(f"\nAll figures: {figures_dir}")


if __name__ == "__main__":
    main()
