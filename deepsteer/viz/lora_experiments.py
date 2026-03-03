"""Visualization functions for LoRA fine-tuning experiments (Phase C Tier 2)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deepsteer.core.types import LoRAExperimentResult

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

# Consistent color palette for conditions
_CONDITION_COLORS = {
    "narrative_moral": "#2196F3",
    "declarative_moral": "#F44336",
    "general_control": "#9E9E9E",
    "moral_curriculum": "#4CAF50",
    "early_lora": "#FF9800",
    "late_lora": "#9C27B0",
}
_DEFAULT_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]


def _get_color(name: str, idx: int) -> str:
    """Get a color for a condition name, falling back to palette index."""
    return _CONDITION_COLORS.get(name, _DEFAULT_COLORS[idx % len(_DEFAULT_COLORS)])


def _make_prefix(experiment: str) -> str:
    """Build a filename prefix for an experiment."""
    return f"lora_{experiment}"


def _save_result_json(data: dict, path: Path) -> None:
    """Write a dict as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON: %s", path)


def plot_lora_fragility_comparison(
    results: dict[str, LoRAExperimentResult],
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
) -> Path:
    """Plot per-layer critical noise comparing multiple LoRA conditions.

    Figure 9: one curve per condition showing fragility profile after training.

    Args:
        results: Dict mapping condition name to LoRAExperimentResult.
        output_dir: Directory for output files.
        filename_prefix: Override auto-generated filename prefix.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or "lora_fragility_comparison"

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, result) in enumerate(results.items()):
        if result.final_fragility is None:
            continue
        layers = [s.layer for s in result.final_fragility.layer_scores]
        criticals = [
            s.critical_noise if s.critical_noise is not None else 0.0
            for s in result.final_fragility.layer_scores
        ]
        color = _get_color(name, i)
        label = name.replace("_", " ").title()
        ax.plot(layers, criticals, "o-", color=color, linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Critical Noise (higher = more robust)")
    ax.set_title("LoRA Experiment: Per-Layer Moral Encoding Robustness")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved fragility comparison: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    json_data = {
        name: result.to_dict() for name, result in results.items()
    }
    _save_result_json(json_data, json_path)

    plt.close(fig)
    return png_path


def plot_lora_acceleration(
    results: dict[str, LoRAExperimentResult],
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
) -> Path:
    """Plot mean probing accuracy vs LoRA training step for each condition.

    Figure 12: annotates 80% accuracy crossing point per condition.

    Args:
        results: Dict mapping condition name to LoRAExperimentResult.
        output_dir: Directory for output files.
        filename_prefix: Override auto-generated filename prefix.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or "lora_acceleration"

    fig, ax = plt.subplots(figsize=(10, 5))
    threshold = 0.8

    for i, (name, result) in enumerate(results.items()):
        if not result.probe_snapshots:
            continue

        steps = [s["step"] for s in result.probe_snapshots]
        accs = [s["probing"]["peak_accuracy"] for s in result.probe_snapshots]
        color = _get_color(name, i)
        label = name.replace("_", " ").title()
        ax.plot(steps, accs, "o-", color=color, linewidth=2, markersize=4, label=label)

        # Find 80% crossing
        for j, acc in enumerate(accs):
            if acc >= threshold:
                crossing_step = steps[j]
                ax.axvline(
                    x=crossing_step, color=color, linestyle=":", linewidth=1, alpha=0.5,
                )
                ax.annotate(
                    f"{crossing_step}",
                    xy=(crossing_step, threshold),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )
                break

    ax.axhline(
        y=threshold, color="#9E9E9E", linestyle="--", linewidth=1,
        label=f"Threshold ({threshold:.0%})",
    )
    ax.set_xlabel("LoRA Training Step")
    ax.set_ylabel("Peak Probing Accuracy")
    ax.set_title("LoRA Experiment: Moral Encoding Acceleration")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved acceleration plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    json_data = {
        name: {
            "steps": [s["step"] for s in result.probe_snapshots],
            "peak_accuracies": [s["probing"]["peak_accuracy"] for s in result.probe_snapshots],
        }
        for name, result in results.items()
        if result.probe_snapshots
    }
    _save_result_json(json_data, json_path)

    plt.close(fig)
    return png_path


def plot_lora_training_loss(
    results: dict[str, LoRAExperimentResult],
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
) -> Path:
    """Plot training loss curves for multiple LoRA conditions.

    Args:
        results: Dict mapping condition name to LoRAExperimentResult.
        output_dir: Directory for output files.
        filename_prefix: Override auto-generated filename prefix.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or "lora_training_loss"

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, result) in enumerate(results.items()):
        if not result.training_steps:
            continue
        steps = [s.step for s in result.training_steps]
        losses = [s.loss for s in result.training_steps]

        # Smooth with moving average for readability
        window = max(1, len(losses) // 50)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            smooth_steps = steps[window - 1:]
        else:
            smoothed = losses
            smooth_steps = steps

        color = _get_color(name, i)
        label = name.replace("_", " ").title()
        ax.plot(
            smooth_steps, smoothed, color=color, linewidth=2, label=label, alpha=0.8,
        )

    ax.set_xlabel("LoRA Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA Experiment: Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved training loss plot: %s", png_path)

    plt.close(fig)
    return png_path


def plot_lora_fragility_trajectory(
    result: LoRAExperimentResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
) -> Path:
    """Plot heatmap of fragility over LoRA training steps (single condition).

    X-axis: LoRA training step (from probe_snapshots)
    Y-axis: layer index
    Color: critical noise at that (step, layer)

    Args:
        result: A single LoRAExperimentResult with fragility snapshots.
        output_dir: Directory for output files.
        filename_prefix: Override auto-generated filename prefix.

    Returns:
        Path to the saved PNG file.
    """
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or f"lora_fragility_trajectory_{result.experiment_id}"

    # Extract fragility data from snapshots
    snapshots_with_frag = [
        s for s in result.probe_snapshots if "fragility" in s
    ]
    if not snapshots_with_frag:
        logger.warning("No fragility snapshots for %s", result.experiment_id)
        # Create empty plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No fragility data", ha="center", va="center",
                transform=ax.transAxes)
        png_path = output_dir / f"{prefix}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        return png_path

    steps = [s["step"] for s in snapshots_with_frag]
    n_layers = len(snapshots_with_frag[0]["fragility"]["layer_critical_noise"])
    n_steps = len(steps)

    matrix = np.zeros((n_layers, n_steps))
    for col, s in enumerate(snapshots_with_frag):
        for row, cn in enumerate(s["fragility"]["layer_critical_noise"]):
            matrix[row, col] = cn if cn is not None else 0.0

    step_labels = [str(s) for s in steps]
    layer_labels = [str(i) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(8, n_steps * 1.2), max(6, n_layers * 0.4)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=step_labels,
        yticklabels=layer_labels,
        cmap="RdYlGn",
        vmin=0.0,
        vmax=10.0,
        annot=n_steps <= 12 and n_layers <= 20,
        fmt=".1f",
        cbar_kws={"label": "Critical Noise"},
    )

    ax.set_xlabel("LoRA Training Step")
    ax.set_ylabel("Layer")
    condition = result.experiment_id.replace("_", " ").title()
    ax.set_title(f"Fragility Trajectory — {condition}")
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved fragility trajectory: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json({
        "experiment_id": result.experiment_id,
        "steps": steps,
        "matrix": matrix.tolist(),
    }, json_path)

    plt.close(fig)
    return png_path
