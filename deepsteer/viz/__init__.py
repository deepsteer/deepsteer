"""Visualization functions for alignment depth results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deepsteer.core.types import CheckpointTrajectoryResult, LayerProbingResult

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _make_prefix(result: LayerProbingResult | CheckpointTrajectoryResult) -> str:
    """Build a filename prefix from the result's model info."""
    name = "unknown"
    if result.model_info is not None:
        name = result.model_info.name.replace("/", "_")
    return f"{result.benchmark_name}_{name}"


def _save_result_json(result: LayerProbingResult | CheckpointTrajectoryResult, path: Path) -> None:
    """Write a result's dict representation as JSON."""
    with open(path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved JSON: %s", path)


def plot_layer_probing(
    result: LayerProbingResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot per-layer probing accuracy and save PNG + companion JSON.

    Args:
        result: Output of ``LayerWiseMoralProbe.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()`` (requires interactive backend).

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    layers = [s.layer for s in result.layer_scores]
    accuracies = [s.accuracy for s in result.layer_scores]
    onset_threshold = result.metadata.get("onset_threshold", 0.6)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, accuracies, "o-", color="#2196F3", linewidth=2, markersize=5, label="Accuracy")

    # Onset threshold line
    ax.axhline(y=onset_threshold, color="#9E9E9E", linestyle="--", linewidth=1,
               label=f"Onset threshold ({onset_threshold:.0%})")

    # Mark onset layer
    if result.onset_layer is not None:
        onset_acc = accuracies[result.onset_layer]
        ax.plot(result.onset_layer, onset_acc, "D", color="#FF9800", markersize=10,
                zorder=5, label=f"Onset (layer {result.onset_layer})")

    # Mark peak layer
    if result.peak_layer is not None:
        ax.plot(result.peak_layer, result.peak_accuracy, "*", color="#4CAF50", markersize=15,
                zorder=5, label=f"Peak (layer {result.peak_layer}, {result.peak_accuracy:.1%})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probing Accuracy")
    model_name = result.model_info.name if result.model_info else "Unknown"
    ax.set_title(f"Layer-Wise Moral Probing — {model_name}")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_checkpoint_trajectory(
    result: CheckpointTrajectoryResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot a heatmap of probing accuracy across layers and checkpoints.

    Args:
        result: Output of ``CheckpointTrajectoryProbe.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()`` (requires interactive backend).

    Returns:
        Path to the saved PNG file.
    """
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    # Build accuracy matrix: rows = layers, columns = checkpoints
    n_checkpoints = len(result.trajectory)
    if n_checkpoints == 0:
        raise ValueError("CheckpointTrajectoryResult has no trajectory entries")

    n_layers = len(result.trajectory[0].layer_scores)
    matrix = np.zeros((n_layers, n_checkpoints))
    for col, probing_result in enumerate(result.trajectory):
        for score in probing_result.layer_scores:
            matrix[score.layer, col] = score.accuracy

    step_labels = [str(s) for s in result.checkpoint_steps]
    layer_labels = [str(i) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(8, n_checkpoints * 1.2), max(6, n_layers * 0.4)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=step_labels,
        yticklabels=layer_labels,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        annot=n_checkpoints <= 10 and n_layers <= 20,
        fmt=".2f",
        cbar_kws={"label": "Probing Accuracy"},
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Layer")
    model_name = result.model_info.name if result.model_info else "Unknown"
    ax.set_title(f"Moral Encoding Trajectory — {model_name}")
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path
