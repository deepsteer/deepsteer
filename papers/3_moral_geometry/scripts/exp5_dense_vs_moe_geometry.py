#!/usr/bin/env python3
"""Experiment 5: Dense vs. MoE framework geometry comparison.

Repeats Experiments 1-2 (foundation-specific probing + geometry analysis)
on OLMoE-1B-7B for comparison with OLMo-2 1B results. Generates
side-by-side comparison figures.

Paper 2 showed MoE encoding is uniform across experts (no specialization).
This experiment asks whether the *framework-level* geometry differs between
architectures despite the uniform expert-level encoding.

Target models:
    - OLMo-2 1B:   allenai/OLMo-2-0425-1B    (already run in Exp 1-3)
    - OLMoE-1B-7B: allenai/OLMoE-1B-7B-0924  (new)

Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS
Estimated runtime: ~15 min (OLMoE probing + geometry)

Usage:
    python papers/3_moral_geometry/scripts/exp5_dense_vs_moe_geometry.py
    python papers/3_moral_geometry/scripts/exp5_dense_vs_moe_geometry.py --quick
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"
OLMO_REPO = "allenai/OLMo-2-0425-1B"

# MPS histc fix (same as Paper 2 scripts)
from deepsteer.core.device import enable_mps_histc_fallback  # noqa: E402
enable_mps_histc_fallback()

# Reuse all geometry functions from Experiments 1-3
from exp1_2_3_framework_geometry import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING,
    BINDING,
    compute_cosine_similarity_matrix,
    compute_effective_dimensionality,
    permutation_test_mft_groups,
    run_experiment_1,
    run_experiment_2,
    train_probe_with_direction,
)


from deepsteer.core.device import clear_memory as _clear_memory  # shared helper


def generate_comparison_figures(
    olmo_exp1: dict,
    olmo_exp2: dict,
    olmoe_exp1: dict,
    olmoe_exp2: dict,
    figures_dir: Path,
) -> None:
    """Generate side-by-side dense vs. MoE geometry comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    n_layers = olmo_exp1["n_layers"]

    # -- Figure: Side-by-side cosine similarity heatmaps --
    # Use layer 7 (stable directions, good accuracy) for both
    display_layer = 7

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, exp1, exp2, title in [
        (axes[0], olmo_exp1, olmo_exp2, "OLMo-2 1B (dense)"),
        (axes[1], olmoe_exp1, olmoe_exp2, "OLMoE-1B-7B (MoE)"),
    ]:
        foundations = [f for f in FOUNDATION_ORDER if f in exp1["directions"]]
        cos_sim = compute_cosine_similarity_matrix(
            exp1["directions"], display_layer, foundations,
        )
        if cos_sim is None:
            continue

        n = len(foundations)
        short_labels = [FOUNDATION_SHORT[f] for f in foundations]

        im = ax.imshow(cos_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(short_labels, fontsize=10)

        for i in range(n):
            for j in range(n):
                val = cos_sim[i, j]
                text_color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color=text_color)

        if n == 6:
            ax.axhline(y=2.5, color="black", linewidth=2)
            ax.axvline(x=2.5, color="black", linewidth=2)

        upper = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_cos = np.mean(upper)
        ax.set_title(f"{title}\nMean pairwise = {mean_cos:.4f}", fontsize=12, fontweight="bold")

    fig.colorbar(axes[1].images[0], ax=axes, shrink=0.8, label="Cosine Similarity")
    fig.suptitle(f"Framework Geometry Comparison (Layer {display_layer})",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp5_cosine_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp5_cosine_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison heatmap: {figures_dir / 'exp5_cosine_comparison.png'}")

    # -- Figure: Layer-wise geometry comparison (2-panel) --
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label, exp2, color, marker in [
        ("OLMo-2 1B (dense)", olmo_exp2, "#2196F3", "o"),
        ("OLMoE-1B-7B (MoE)", olmoe_exp2, "#F44336", "s"),
    ]:
        if not exp2 or "mean_cosine" not in exp2:
            continue
        mc = exp2["mean_cosine"]
        ed = exp2["effective_dims"]

        layers_mc = sorted(mc.keys())
        layers_ed = sorted(ed.keys())

        axes[0].plot(layers_mc, [mc[l] for l in layers_mc],
                     f"{marker}-", color=color, linewidth=2, markersize=5, label=label)
        axes[1].plot(layers_ed, [ed[l] for l in layers_ed],
                     f"{marker}-", color=color, linewidth=2, markersize=5, label=label)

    axes[0].set_xlabel("Layer", fontsize=11)
    axes[0].set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    axes[0].set_title("(a) Collapse Metric", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Layer", fontsize=11)
    axes[1].set_ylabel("Effective Dimensionality (90% var)", fontsize=11)
    axes[1].set_title("(b) Direction Set Dimensionality", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.5, 6.5)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Dense vs. MoE: Layer-Wise Framework Geometry",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp5_layerwise_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp5_layerwise_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Layer-wise comparison: {figures_dir / 'exp5_layerwise_comparison.png'}")

    # -- Figure: Side-by-side dendrograms --
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, exp1, title in [
        (axes[0], olmo_exp1, "OLMo-2 1B (dense)"),
        (axes[1], olmoe_exp1, "OLMoE-1B-7B (MoE)"),
    ]:
        foundations = [f for f in FOUNDATION_ORDER if f in exp1["directions"]]
        cos_sim = compute_cosine_similarity_matrix(
            exp1["directions"], display_layer, foundations,
        )
        if cos_sim is None:
            continue

        n = len(foundations)
        short_labels = [FOUNDATION_SHORT[f] for f in foundations]
        dist = 1 - cos_sim
        condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]

        Z = linkage(np.array(condensed), method="ward")
        dendrogram(Z, labels=short_labels, ax=ax, leaf_font_size=10)
        ax.set_ylabel("Ward Distance", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")

    fig.suptitle(f"Foundation Clustering Comparison (Layer {display_layer})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp5_dendrogram_comparison.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp5_dendrogram_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Dendrogram comparison: {figures_dir / 'exp5_dendrogram_comparison.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Dense vs. MoE framework geometry.",
    )
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/exp5_dense_vs_moe")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument("--model", default=OLMO_REPO,
                        help="HuggingFace model ID for the dense model.")
    parser.add_argument("--quick", action="store_true",
                        help="Skip dense model re-run, load from Exp 1-3 JSON (1B only).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path("papers/3_moral_geometry/outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target, dataset_version="v2")
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # --- OLMo-2 1B (dense) ---
    olmo_exp1 = None
    olmo_exp2 = None

    exp1_3_dir = Path("papers/3_moral_geometry/outputs/exp1_2_3")
    npz_path = exp1_3_dir / "exp1_probe_directions.npz"
    json_path = exp1_3_dir / "exp1_foundation_probing.json"
    geo_path = exp1_3_dir / "exp2_framework_geometry.json"

    if args.quick and npz_path.exists() and json_path.exists():
        print("\nLoading OLMo-2 1B results from Exp 1-3...")
        with open(json_path) as f:
            exp1_json = json.load(f)
        npz = np.load(npz_path)

        directions: dict[str, dict[int, np.ndarray]] = {}
        accuracies: dict[str, dict[int, float]] = {}
        n_layers = exp1_json["n_layers"]
        hidden_dim = exp1_json["hidden_dim"]

        for fv, fdata in exp1_json["per_foundation"].items():
            directions[fv] = {}
            accuracies[fv] = {}
            for layer_str, acc in fdata["per_layer_accuracy"].items():
                layer = int(layer_str)
                key = f"{fv}_layer{layer}"
                if key in npz:
                    directions[fv][layer] = npz[key]
                accuracies[fv][layer] = acc

        olmo_exp1 = {
            "directions": directions,
            "accuracies": accuracies,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
        }

        if geo_path.exists():
            with open(geo_path) as f:
                geo_json = json.load(f)
            olmo_exp2 = {
                "mean_cosine": {int(k): v["mean_cosine_similarity"]
                                for k, v in geo_json["per_layer"].items()
                                if "mean_cosine_similarity" in v},
                "effective_dims": {int(k): v["effective_dimensionality"]
                                   for k, v in geo_json["per_layer"].items()
                                   if "effective_dimensionality" in v},
            }
        print(f"  Loaded {len(directions)} foundations, {n_layers} layers")
    else:
        print(f"\n{'='*60}")
        print(f"Loading dense model: {args.model}")
        print(f"{'='*60}")
        t0 = time.time()
        olmo_model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
        print(f"Loaded in {time.time() - t0:.1f}s")

        olmo_dir = output_dir / "olmo"
        olmo_dir.mkdir(exist_ok=True)

        olmo_exp1 = run_experiment_1(olmo_model, dataset, olmo_dir)
        olmo_exp2 = run_experiment_2(olmo_exp1, olmo_dir)

        del olmo_model
        _clear_memory()

    # --- OLMoE-1B-7B (MoE) ---
    print(f"\n{'='*60}")
    print(f"Loading OLMoE-1B-7B: {OLMOE_REPO}")
    print(f"{'='*60}")
    t0 = time.time()
    olmoe_model = WhiteBoxModel(OLMOE_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    print(f"Loaded in {time.time() - t0:.1f}s "
          f"({olmoe_model.info.n_params / 1e9:.1f}B params, "
          f"{olmoe_model.info.n_layers} layers)")

    olmoe_dir = output_dir / "olmoe"
    olmoe_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Foundation Probing on OLMoE")
    print(f"{'='*60}")
    t0 = time.time()
    olmoe_exp1 = run_experiment_1(olmoe_model, dataset, olmoe_dir)
    print(f"OLMoE probing complete: {time.time() - t0:.1f}s")

    print(f"\n{'='*60}")
    print("EXPERIMENT 5: Geometry Analysis on OLMoE")
    print(f"{'='*60}")
    t0 = time.time()
    olmoe_exp2 = run_experiment_2(olmoe_exp1, olmoe_dir)
    print(f"OLMoE geometry complete: {time.time() - t0:.1f}s")

    del olmoe_model
    _clear_memory()

    # --- Comparison ---
    print(f"\n{'='*60}")
    print("Generating comparison figures...")
    print(f"{'='*60}")

    if olmo_exp1 and olmo_exp2 and olmoe_exp1 and olmoe_exp2:
        generate_comparison_figures(
            olmo_exp1, olmo_exp2, olmoe_exp1, olmoe_exp2, figures_dir,
        )

        # Summary JSON
        summary = {
            "experiment": "exp5_dense_vs_moe_geometry",
            "olmo": {
                "model": args.model,
                "peak_separation_layer": olmo_exp2.get("peak_separation_layer"),
                "mean_cosine_range": {
                    "min": min(olmo_exp2["mean_cosine"].values()),
                    "max": max(olmo_exp2["mean_cosine"].values()),
                },
            },
            "olmoe": {
                "model": OLMOE_REPO,
                "peak_separation_layer": olmoe_exp2.get("peak_separation_layer"),
                "mean_cosine_range": {
                    "min": min(olmoe_exp2["mean_cosine"].values()),
                    "max": max(olmoe_exp2["mean_cosine"].values()),
                },
            },
        }
        with open(output_dir / "exp5_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print comparison
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        for label, exp2 in [("OLMo-2 1B (dense)", olmo_exp2), ("OLMoE-1B-7B (MoE)", olmoe_exp2)]:
            mc = exp2["mean_cosine"]
            ed = exp2["effective_dims"]
            print(f"\n  {label}:")
            print(f"    Mean cos sim range: {min(mc.values()):.4f} – {max(mc.values()):.4f}")
            print(f"    Effective dim range: {min(ed.values())} – {max(ed.values())}")

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
