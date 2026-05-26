#!/usr/bin/env python3
"""Experiment 6: Geometric trajectory during training.

Tracks when framework separation emerges during pre-training by running
foundation-specific probing + geometry analysis across OLMo-2 1B checkpoints.

Key question: does framework separation emerge after binary moral detection
accuracy saturates? If so, this extends the "structure develops after accuracy
saturates" finding from Papers 1 and 2 to a third metric (geometric structure).

Target: allenai/OLMo-2-0425-1B-early-training (37 checkpoints, step 0-36K)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS
Estimated runtime: ~2-3 hours (37 checkpoints × ~3 min each)

Outputs:
    - Per-checkpoint foundation probing results and probe directions
    - Trajectory plots: mean cos sim, effective dim, MFT structure across training
    - Overlay with binary probe accuracy trajectory from Paper 1

Usage:
    # Full run (all 37 checkpoints)
    python papers/3_moral_geometry/scripts/exp6_geometric_trajectory.py

    # Subset of checkpoints (fast test)
    python papers/3_moral_geometry/scripts/exp6_geometric_trajectory.py --max-checkpoints 5

    # Resume from a specific step
    python papers/3_moral_geometry/scripts/exp6_geometric_trajectory.py --resume-from 18000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"

from exp1_2_3_framework_geometry import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING,
    BINDING,
    compute_cosine_similarity_matrix,
    compute_effective_dimensionality,
    permutation_test_mft_groups,
    train_probe_with_direction,
)


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _parse_step(revision: str) -> int | None:
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def _get_all_revisions() -> list[tuple[int, str]]:
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    all_revisions = list_available_revisions(REPO_ID)
    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        step = _parse_step(rev)
        if step is not None:
            step_revisions.append((step, rev))
    step_revisions.sort(key=lambda x: x[0])
    return step_revisions


def run_checkpoint_geometry(
    model,
    dataset,
    n_layers: int,
    *,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> dict:
    """Run foundation probing + geometry for a single checkpoint.

    Returns compact dict with directions, accuracies, and geometry metrics.
    """
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.core.types import MoralFoundation
    from deepsteer.datasets.types import ProbingPair

    train_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    test_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    for pair in dataset.train:
        train_by_foundation[pair.foundation.value].append(pair)
    for pair in dataset.test:
        test_by_foundation[pair.foundation.value].append(pair)

    directions: dict[str, dict[int, np.ndarray]] = {}
    accuracies: dict[str, dict[int, float]] = {}

    for foundation_val in FOUNDATION_ORDER:
        foundation = MoralFoundation(foundation_val)
        train_pairs = train_by_foundation.get(foundation_val, [])
        test_pairs = test_by_foundation.get(foundation_val, [])

        if len(train_pairs) < 5 or len(test_pairs) < 1:
            continue

        all_train = LayerWiseMoralProbe._collect_all_activations(model, train_pairs)
        all_test = LayerWiseMoralProbe._collect_all_activations(model, test_pairs)

        directions[foundation_val] = {}
        accuracies[foundation_val] = {}

        for layer_idx in range(n_layers):
            train_X, train_y = all_train[layer_idx]
            test_X, test_y = all_test[layer_idx]

            acc, loss, w_norm = train_probe_with_direction(
                train_X, train_y, test_X, test_y,
                n_epochs=n_epochs, lr=lr,
            )
            directions[foundation_val][layer_idx] = w_norm
            accuracies[foundation_val][layer_idx] = acc

    # Compute geometry metrics
    foundations_present = [f for f in FOUNDATION_ORDER if f in directions]
    mean_cosine: dict[int, float] = {}
    effective_dims: dict[int, int] = {}
    perm_stats: dict[int, float] = {}

    for layer_idx in range(n_layers):
        cos_sim = compute_cosine_similarity_matrix(
            directions, layer_idx, foundations_present,
        )
        if cos_sim is None:
            continue

        n = len(foundations_present)
        upper = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_cosine[layer_idx] = float(np.mean(upper))

        eff_dim = compute_effective_dimensionality(
            directions, layer_idx, foundations_present,
        )
        if eff_dim is not None:
            effective_dims[layer_idx] = eff_dim

        if len(foundations_present) == 6:
            pt = permutation_test_mft_groups(
                cos_sim, foundations_present, n_permutations=1000,
            )
            perm_stats[layer_idx] = pt["observed_statistic"]

    # Summary: aggregate across layers 5-14 (stable direction range)
    stable_layers = [l for l in range(5, min(15, n_layers))]
    stable_mean_cos = np.mean([mean_cosine.get(l, 0) for l in stable_layers])
    stable_mean_dim = np.mean([effective_dims.get(l, 1) for l in stable_layers])

    # Mean accuracy across all foundations (pooled)
    all_accs = []
    for fv in foundations_present:
        for l in stable_layers:
            if l in accuracies.get(fv, {}):
                all_accs.append(accuracies[fv][l])
    mean_acc = float(np.mean(all_accs)) if all_accs else 0.5

    return {
        "directions": directions,
        "accuracies": accuracies,
        "mean_cosine": mean_cosine,
        "effective_dims": effective_dims,
        "perm_stats": perm_stats,
        "foundations_present": foundations_present,
        "stable_mean_cosine": float(stable_mean_cos),
        "stable_mean_dim": float(stable_mean_dim),
        "mean_accuracy_stable_layers": mean_acc,
    }


def generate_trajectory_figures(
    trajectory: dict[int, dict],
    output_dir: Path,
    figures_dir: Path,
) -> None:
    """Generate trajectory figures across training steps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(trajectory.keys())
    if len(steps) < 2:
        print("  Not enough checkpoints for trajectory plots")
        return

    # -- Figure 6: Three-panel trajectory plot --
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel (a): Mean cosine similarity (stable layers) vs training step
    ax = axes[0]
    mean_cos_values = [trajectory[s]["stable_mean_cosine"] for s in steps]
    ax.plot(steps, mean_cos_values, "o-", color="#1E88E5", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Pairwise Cosine Similarity\n(layers 5-14)", fontsize=10)
    ax.set_title("(a) Framework Collapse/Separation", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    # Panel (b): Effective dimensionality (stable layers) vs training step
    ax = axes[1]
    mean_dim_values = [trajectory[s]["stable_mean_dim"] for s in steps]
    ax.plot(steps, mean_dim_values, "s-", color="#E53935", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Effective Dimensionality\n(layers 5-14, 90% var)", fontsize=10)
    ax.set_title("(b) Framework Direction Diversity", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 6.5)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    # Panel (c): Mean accuracy (stable layers) overlaid with cos sim
    ax = axes[2]
    mean_acc_values = [trajectory[s]["mean_accuracy_stable_layers"] for s in steps]
    ax.plot(steps, mean_acc_values, "o-", color="#43A047", linewidth=2, markersize=4,
            label="Mean probe accuracy")
    ax.set_ylabel("Probe Accuracy", fontsize=10, color="#43A047")
    ax.tick_params(axis="y", labelcolor="#43A047")
    ax.set_ylim(0.4, 1.05)

    ax2 = ax.twinx()
    ax2.plot(steps, mean_cos_values, "s--", color="#1E88E5", linewidth=1.5, markersize=3,
             alpha=0.7, label="Mean cos similarity")
    ax2.set_ylabel("Mean Cosine Similarity", fontsize=10, color="#1E88E5")
    ax2.tick_params(axis="y", labelcolor="#1E88E5")

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_title("(c) Accuracy vs. Geometry", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center right")

    fig.suptitle("Geometric Trajectory During Pre-Training (OLMo-2 1B)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig6_geometric_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig6_geometric_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Trajectory figure: {figures_dir / 'fig6_geometric_trajectory.png'}")

    # -- Per-layer heatmap: mean cosine similarity across training --
    fig, ax = plt.subplots(figsize=(14, 6))

    n_layers = max(
        max(trajectory[s]["mean_cosine"].keys(), default=0)
        for s in steps
    ) + 1

    heatmap_data = np.full((n_layers, len(steps)), np.nan)
    for j, step in enumerate(steps):
        for layer, mc in trajectory[step]["mean_cosine"].items():
            heatmap_data[layer, j] = mc

    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r",
                   vmin=0, vmax=0.6, origin="lower")
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title("Mean Pairwise Cosine Similarity Across Training\n"
                 "(higher = more collapsed framework directions)",
                 fontsize=12, fontweight="bold")

    step_labels = [str(s) if i % max(1, len(steps) // 10) == 0 else ""
                   for i, s in enumerate(steps)]
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(step_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_layers))

    fig.colorbar(im, ax=ax, label="Mean Cosine Similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(figures_dir / "exp6_cosine_heatmap_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp6_cosine_heatmap_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Cosine heatmap trajectory: {figures_dir / 'exp6_cosine_heatmap_trajectory.png'}")

    # -- Per-foundation accuracy trajectory --
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "care_harm": "#E53935",
        "fairness_cheating": "#1E88E5",
        "liberty_oppression": "#43A047",
        "loyalty_betrayal": "#FB8C00",
        "authority_subversion": "#8E24AA",
        "sanctity_degradation": "#00ACC1",
    }

    # Use mean across stable layers for each foundation
    for fv in FOUNDATION_ORDER:
        acc_trajectory = []
        for step in steps:
            accs_at_step = trajectory[step]["accuracies"].get(fv, {})
            stable = [accs_at_step.get(l, 0.5) for l in range(5, 15)
                      if l in accs_at_step]
            acc_trajectory.append(float(np.mean(stable)) if stable else 0.5)

        marker = "o" if fv in INDIVIDUALIZING else "s"
        ax.plot(steps, acc_trajectory, f"{marker}-", color=colors.get(fv, "#666"),
                linewidth=1.5, markersize=3, label=FOUNDATION_SHORT[fv])

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Mean Probe Accuracy (layers 5-14)", fontsize=11)
    ax.set_title("Per-Foundation Accuracy Across Training", fontsize=12, fontweight="bold")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(figures_dir / "exp6_foundation_accuracy_trajectory.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp6_foundation_accuracy_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Foundation accuracy trajectory: {figures_dir / 'exp6_foundation_accuracy_trajectory.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 6: Geometric trajectory during training.",
    )
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/exp6_trajectory")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument("--max-checkpoints", type=int, default=None,
                        help="Limit to N evenly-spaced checkpoints.")
    parser.add_argument("--resume-from", type=int, default=None,
                        help="Skip checkpoints with step < this value.")
    parser.add_argument("--step-interval", type=int, default=None,
                        help="Only process every Nth step (e.g. 3000 for every 3K steps).")
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
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # Get available checkpoints
    print(f"\nQuerying checkpoints for {REPO_ID}...")
    step_revisions = _get_all_revisions()
    print(f"Found {len(step_revisions)} checkpoints")

    # Filter checkpoints
    if args.resume_from is not None:
        step_revisions = [(s, r) for s, r in step_revisions if s >= args.resume_from]
        print(f"  After resume filter: {len(step_revisions)} checkpoints")

    if args.step_interval is not None:
        step_revisions = [(s, r) for s, r in step_revisions if s % args.step_interval == 0]
        print(f"  After interval filter: {len(step_revisions)} checkpoints")

    if args.max_checkpoints is not None and len(step_revisions) > args.max_checkpoints:
        indices = np.linspace(0, len(step_revisions) - 1, args.max_checkpoints, dtype=int)
        step_revisions = [step_revisions[i] for i in indices]
        print(f"  Subsampled to {len(step_revisions)} checkpoints")

    print(f"  Steps: {[s for s, _ in step_revisions]}")

    # Load existing results
    trajectory: dict[int, dict] = {}
    summary_path = output_dir / "exp6_trajectory_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            saved = json.load(f)
        for step_str, data in saved.get("per_step", {}).items():
            step = int(step_str)
            trajectory[step] = {
                "accuracies": {
                    fv: {int(k): v for k, v in accs.items()}
                    for fv, accs in data.get("accuracies", {}).items()
                },
                "mean_cosine": {int(k): v for k, v in data.get("mean_cosine", {}).items()},
                "effective_dims": {int(k): v for k, v in data.get("effective_dims", {}).items()},
                "perm_stats": {int(k): v for k, v in data.get("perm_stats", {}).items()},
                "foundations_present": data.get("foundations_present", []),
                "stable_mean_cosine": data.get("stable_mean_cosine", 0),
                "stable_mean_dim": data.get("stable_mean_dim", 0),
                "mean_accuracy_stable_layers": data.get("mean_accuracy_stable_layers", 0.5),
                "directions": {},  # Not saved to JSON, would need npz
            }
        print(f"  Loaded {len(trajectory)} cached results")

    # Process checkpoints
    total = len(step_revisions)
    t_total_start = time.time()

    for idx, (step, revision) in enumerate(step_revisions):
        if step in trajectory and trajectory[step].get("mean_cosine"):
            print(f"\n[{idx+1}/{total}] Step {step}: cached, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"[{idx+1}/{total}] Step {step} ({revision})")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            model = WhiteBoxModel(
                REPO_ID,
                device=args.device,
                access_tier=AccessTier.CHECKPOINTS,
                revision=revision,
            )
        except Exception as e:
            print(f"  ERROR loading checkpoint: {e}")
            continue

        n_layers = model.info.n_layers
        load_time = time.time() - t0
        print(f"  Loaded in {load_time:.1f}s ({n_layers} layers)")

        t0 = time.time()
        result = run_checkpoint_geometry(model, dataset, n_layers)
        probe_time = time.time() - t0
        print(f"  Probing + geometry: {probe_time:.1f}s")
        print(f"  Stable mean cos sim: {result['stable_mean_cosine']:.4f}")
        print(f"  Stable mean dim: {result['stable_mean_dim']:.1f}")
        print(f"  Mean accuracy (stable): {result['mean_accuracy_stable_layers']:.3f}")

        # Save probe directions for this checkpoint
        direction_arrays = {}
        for fv in FOUNDATION_ORDER:
            if fv not in result["directions"]:
                continue
            for layer_idx, w in result["directions"][fv].items():
                direction_arrays[f"{fv}_layer{layer_idx}"] = w
        np.savez(output_dir / f"directions_step{step:07d}.npz", **direction_arrays)

        trajectory[step] = result

        del model
        _clear_memory()

        # Save incremental summary
        _save_summary(trajectory, output_dir, summary_path)

        elapsed_total = time.time() - t_total_start
        steps_done = idx + 1
        steps_remaining = total - steps_done
        rate = elapsed_total / steps_done
        eta = rate * steps_remaining
        print(f"  Progress: {steps_done}/{total}, "
              f"{elapsed_total:.0f}s elapsed, ~{eta:.0f}s remaining")

    # Final save
    _save_summary(trajectory, output_dir, summary_path)

    # Generate figures
    print(f"\n{'='*60}")
    print("Generating trajectory figures...")
    print(f"{'='*60}")
    generate_trajectory_figures(trajectory, output_dir, figures_dir)

    print(f"\nAll outputs: {output_dir}")
    print(f"Figures: {figures_dir}")


def _save_summary(
    trajectory: dict[int, dict],
    output_dir: Path,
    summary_path: Path,
) -> None:
    """Save trajectory summary JSON (excluding numpy arrays)."""
    per_step = {}
    for step, data in sorted(trajectory.items()):
        per_step[str(step)] = {
            "accuracies": {
                fv: {str(k): round(v, 4) for k, v in accs.items()}
                for fv, accs in data.get("accuracies", {}).items()
            },
            "mean_cosine": {str(k): round(v, 6) for k, v in data.get("mean_cosine", {}).items()},
            "effective_dims": {str(k): v for k, v in data.get("effective_dims", {}).items()},
            "perm_stats": {str(k): round(v, 6) for k, v in data.get("perm_stats", {}).items()},
            "foundations_present": data.get("foundations_present", []),
            "stable_mean_cosine": round(data.get("stable_mean_cosine", 0), 6),
            "stable_mean_dim": round(data.get("stable_mean_dim", 0), 2),
            "mean_accuracy_stable_layers": round(data.get("mean_accuracy_stable_layers", 0.5), 4),
        }

    summary = {
        "experiment": "exp6_geometric_trajectory",
        "model": REPO_ID,
        "n_checkpoints": len(trajectory),
        "steps": sorted(trajectory.keys()),
        "per_step": per_step,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
