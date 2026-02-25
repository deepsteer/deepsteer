"""Visualization functions for alignment depth results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from deepsteer.core.types import (
    BenchmarkResult,
    CheckpointTrajectoryResult,
    ComplianceGapResult,
    LayerProbingResult,
    MoralFoundationsResult,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _make_prefix(result: BenchmarkResult) -> str:
    """Build a filename prefix from the result's model info."""
    name = "unknown"
    if result.model_info is not None:
        name = result.model_info.name.replace("/", "_")
    return f"{result.benchmark_name}_{name}"


def _save_result_json(result: BenchmarkResult, path: Path) -> None:
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


# ---------------------------------------------------------------------------
# Behavioral evaluation plots
# ---------------------------------------------------------------------------


def plot_moral_foundations(
    result: MoralFoundationsResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot moral foundations accuracy by difficulty level.

    Produces a grouped bar chart with one group per foundation, bars colored
    by difficulty level.

    Args:
        result: Output of ``MoralFoundationsProbe.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    from deepsteer.core.types import DifficultyLevel, MoralFoundation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    foundations = list(MoralFoundation)
    difficulties = list(DifficultyLevel)
    foundation_labels = [f.value.replace("_", "/\n") for f in foundations]
    difficulty_labels = [d.name.capitalize() for d in difficulties]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336"]

    # Build accuracy lookup: (foundation, difficulty) → accuracy
    acc_lookup: dict[tuple[str, str], float] = {}
    for score in result.scores:
        key = (score.foundation.value, score.difficulty.name.lower())
        acc_lookup[key] = score.accuracy

    n_groups = len(foundations)
    n_bars = len(difficulties)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, diff in enumerate(difficulties):
        values = [
            acc_lookup.get((f.value, diff.name.lower()), 0.0)
            for f in foundations
        ]
        offset = (i - n_bars / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=difficulty_labels[i], color=colors[i])

    ax.set_xlabel("Moral Foundation")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(foundation_labels, fontsize=8)
    model_name = result.model_info.name if result.model_info else "Unknown"
    ax.set_title(f"Moral Foundations Probe — {model_name}")
    ax.set_ylim(0, 1.1)
    ax.legend(title="Difficulty")
    ax.grid(True, alpha=0.3, axis="y")
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


def plot_compliance_gap(
    result: ComplianceGapResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot compliance rates under monitored vs. unmonitored conditions.

    Produces a grouped bar chart with one group per category, two bars each
    (monitored, unmonitored), plus an overall summary.

    Args:
        result: Output of ``ComplianceGapDetector.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    # Compute per-category compliance rates
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for r in result.scenario_results:
        by_cat[r.category].append(r)

    categories = sorted(by_cat.keys()) + ["OVERALL"]
    monitored_rates = []
    unmonitored_rates = []
    for cat in categories[:-1]:
        results = by_cat[cat]
        monitored_rates.append(sum(r.monitored_complied for r in results) / len(results))
        unmonitored_rates.append(sum(r.unmonitored_complied for r in results) / len(results))
    monitored_rates.append(result.monitored_compliance_rate)
    unmonitored_rates.append(result.unmonitored_compliance_rate)

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, monitored_rates, width, label="Monitored", color="#2196F3")
    ax.bar(x + width / 2, unmonitored_rates, width, label="Unmonitored", color="#F44336")

    ax.set_xlabel("Category")
    ax.set_ylabel("Compliance Rate")
    cat_labels = [c.replace("_", " ").title() for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels, fontsize=8, rotation=15, ha="right")
    model_name = result.model_info.name if result.model_info else "Unknown"
    gap = result.compliance_gap or 0.0
    ax.set_title(f"Compliance Gap — {model_name}  (gap = {gap:+.1%})")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
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
