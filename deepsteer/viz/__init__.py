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
    CausalTracingResult,
    CheckpointTrajectoryResult,
    ComplianceGapResult,
    CurriculumSchedule,
    FoundationProbingResult,
    FragilityResult,
    LayerProbingResult,
    LoRAExperimentResult,
    MixingResult,
    MonitoringSession,
    MoralFoundationsResult,
    PersonaShiftResult,
)
from deepsteer.viz.lora_experiments import (
    plot_lora_acceleration,
    plot_lora_fragility_comparison,
    plot_lora_fragility_trajectory,
    plot_lora_training_loss,
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


def plot_model_comparison(
    results: list[LayerProbingResult],
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str = "model_comparison",
    show: bool = False,
) -> Path:
    """Overlay layer probing curves from multiple models.

    Useful for comparing moral encoding depth across model families
    (e.g. OLMo vs. Llama).

    Args:
        results: List of ``LayerProbingResult`` from different models.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Output filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    if not results:
        raise ValueError("results list is empty")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    onset_threshold = results[0].metadata.get("onset_threshold", 0.6)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, result in enumerate(results):
        color = colors[i % len(colors)]
        model_name = result.model_info.name if result.model_info else f"Model {i}"
        layers = [s.layer for s in result.layer_scores]
        accuracies = [s.accuracy for s in result.layer_scores]

        # Normalize x-axis to [0, 1] so models with different layer counts
        # can be compared on the same relative depth scale
        n_layers = len(layers)
        x_norm = [l / (n_layers - 1) if n_layers > 1 else 0.5 for l in layers]

        ax.plot(x_norm, accuracies, "o-", color=color, linewidth=2, markersize=4,
                label=f"{model_name} (peak={result.peak_accuracy:.0%})")

        # Mark peak
        if result.peak_layer is not None:
            peak_x = result.peak_layer / (n_layers - 1) if n_layers > 1 else 0.5
            ax.plot(peak_x, result.peak_accuracy, "*", color=color,
                    markersize=12, zorder=5)

    ax.axhline(y=onset_threshold, color="#9E9E9E", linestyle="--", linewidth=1,
               label=f"Onset threshold ({onset_threshold:.0%})")

    ax.set_xlabel("Relative Depth (layer / total layers)")
    ax.set_ylabel("Probing Accuracy")
    ax.set_title("Moral Encoding Depth — Model Comparison")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved comparison plot: %s", png_path)

    # Save all results as companion JSON
    json_path = output_dir / f"{filename_prefix}.json"
    comparison_data = {
        "models": [
            {
                "name": r.model_info.name if r.model_info else f"model_{i}",
                "result": r.to_dict(),
            }
            for i, r in enumerate(results)
        ],
    }
    with open(json_path, "w") as f:
        json.dump(comparison_data, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

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


# ---------------------------------------------------------------------------
# Phase 5: Expanded benchmark plots
# ---------------------------------------------------------------------------


def plot_persona_shift(
    result: PersonaShiftResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot baseline vs. persona compliance rates as a grouped bar chart.

    Args:
        result: Output of ``PersonaShiftDetector.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    from collections import defaultdict

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    # Per-persona rates
    by_persona: dict[str, list] = defaultdict(list)
    for r in result.scenario_results:
        by_persona[r.persona_name].append(r)

    labels = sorted(by_persona.keys()) + ["OVERALL"]
    baseline_rates: list[float] = []
    persona_rates: list[float] = []
    for pname in labels[:-1]:
        items = by_persona[pname]
        baseline_rates.append(sum(r.baseline_complied for r in items) / len(items))
        persona_rates.append(sum(r.persona_complied for r in items) / len(items))
    baseline_rates.append(result.baseline_compliance_rate or 0.0)
    persona_rates.append(result.persona_compliance_rate or 0.0)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, baseline_rates, width, label="Baseline", color="#2196F3")
    ax.bar(x + width / 2, persona_rates, width, label="Persona", color="#F44336")

    ax.set_xlabel("Persona")
    ax.set_ylabel("Compliance Rate")
    display_labels = [l.replace("_", " ").title() for l in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=8, rotation=15, ha="right")
    model_name = result.model_info.name if result.model_info else "Unknown"
    gap = result.persona_shift_gap or 0.0
    ax.set_title(f"Persona Shift — {model_name}  (gap = {gap:+.1%})")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved persona shift plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_foundation_probes(
    result: FoundationProbingResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot per-foundation probing accuracy across layers (multi-line).

    Args:
        result: Output of ``FoundationSpecificProbe.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    from deepsteer.core.types import MoralFoundation

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    # Group scores by foundation
    from collections import defaultdict
    by_foundation: dict[str, list] = defaultdict(list)
    for s in result.foundation_layer_scores:
        by_foundation[s.foundation.value].append(s)

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    onset_threshold = result.metadata.get("onset_threshold", 0.6)

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, foundation in enumerate(MoralFoundation):
        scores = by_foundation.get(foundation.value, [])
        if not scores:
            continue
        scores_sorted = sorted(scores, key=lambda s: s.layer)
        layers = [s.layer for s in scores_sorted]
        accs = [s.accuracy for s in scores_sorted]
        color = colors[i % len(colors)]
        label = foundation.value.replace("_", "/")
        ax.plot(layers, accs, "o-", color=color, linewidth=2, markersize=4, label=label)

    ax.axhline(
        y=onset_threshold, color="#9E9E9E", linestyle="--", linewidth=1,
        label=f"Onset threshold ({onset_threshold:.0%})",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Probing Accuracy")
    model_name = result.model_info.name if result.model_info else "Unknown"
    ax.set_title(f"Foundation-Specific Probing — {model_name}")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved foundation probes plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_causal_tracing(
    result: CausalTracingResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot mean indirect causal effect by layer as a bar chart.

    Args:
        result: Output of ``MoralCausalTracer.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    layers = sorted(result.mean_indirect_effect_by_layer.keys())
    effects = [result.mean_indirect_effect_by_layer[l] for l in layers]

    # Highlight peak layer
    peak = result.peak_causal_layer
    bar_colors = ["#F44336" if l == peak else "#2196F3" for l in layers]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(layers, effects, color=bar_colors)

    if peak is not None and peak in result.mean_indirect_effect_by_layer:
        ax.bar(
            [peak], [result.mean_indirect_effect_by_layer[peak]],
            color="#F44336", label=f"Peak (layer {peak})",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Indirect Effect")
    model_name = result.model_info.name if result.model_info else "Unknown"
    depth = result.causal_depth or 0.0
    ax.set_title(f"Causal Tracing — {model_name}  (depth = {depth:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved causal tracing plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_fragility(
    result: FragilityResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str | None = None,
    show: bool = False,
) -> Path:
    """Plot a heatmap of probing accuracy across layers and noise levels.

    Args:
        result: Output of ``MoralFragilityTest.run()``.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Override the auto-generated filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    import seaborn as sns

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = filename_prefix or _make_prefix(result)

    n_layers = len(result.layer_scores)
    noise_levels = sorted(result.noise_levels)
    n_noise = len(noise_levels)

    # Build accuracy matrix: rows = layers, columns = noise levels
    matrix = np.zeros((n_layers, n_noise))
    layer_labels: list[str] = []
    for row, ls in enumerate(result.layer_scores):
        layer_labels.append(str(ls.layer))
        for col, nl in enumerate(noise_levels):
            matrix[row, col] = ls.accuracy_by_noise.get(nl, 0.0)

    noise_labels = [str(nl) for nl in noise_levels]

    fig, ax = plt.subplots(figsize=(max(8, n_noise * 1.5), max(6, n_layers * 0.4)))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=noise_labels,
        yticklabels=layer_labels,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        annot=n_noise <= 10 and n_layers <= 20,
        fmt=".2f",
        cbar_kws={"label": "Probing Accuracy"},
    )

    ax.set_xlabel("Noise Level (std)")
    ax.set_ylabel("Layer")
    model_name = result.model_info.name if result.model_info else "Unknown"
    ax.set_title(f"Moral Encoding Fragility — {model_name}")
    fig.tight_layout()

    png_path = output_dir / f"{prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved fragility plot: %s", png_path)

    json_path = output_dir / f"{prefix}.json"
    _save_result_json(result, json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


# ---------------------------------------------------------------------------
# Phase 6: Steering tool plots
# ---------------------------------------------------------------------------


def plot_curriculum_schedule(
    schedule: CurriculumSchedule,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str = "curriculum_schedule",
    show: bool = False,
) -> Path:
    """Plot moral content ratio over training steps.

    Args:
        schedule: A CurriculumSchedule to visualize.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Output filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))

    for phase in schedule.phases:
        mid = (phase.start_step + phase.end_step) / 2.0
        width = phase.end_step - phase.start_step
        ax.bar(mid, phase.moral_ratio, width=width, align="center",
               color="#2196F3", alpha=0.7, edgecolor="#1565C0", linewidth=0.5)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Moral Content Ratio")
    ax.set_title(f"Curriculum Schedule — {schedule.method}")
    ax.set_xlim(0, schedule.total_steps)
    ax.set_ylim(0, max(p.moral_ratio for p in schedule.phases) * 1.2 if schedule.phases else 0.2)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    png_path = output_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved curriculum plot: %s", png_path)

    json_path = output_dir / f"{filename_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(schedule.to_dict(), f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_mixing_distribution(
    result: MixingResult,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str = "mixing_distribution",
    show: bool = False,
) -> Path:
    """Plot foundation distribution in a mixed batch.

    Args:
        result: A MixingResult from DataMixer.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Output filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: moral vs general pie
    ax_pie = axes[0]
    sizes = [result.moral_samples, result.general_samples]
    labels = ["Moral", "General"]
    colors_pie = ["#F44336", "#2196F3"]
    ax_pie.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%", startangle=90)
    ax_pie.set_title(f"Moral vs General  (n={result.total_samples})")

    # Right: foundation breakdown bar chart
    ax_bar = axes[1]
    if result.foundation_counts:
        foundations = sorted(result.foundation_counts.keys())
        counts = [result.foundation_counts[f] for f in foundations]
        bar_labels = [f.replace("_", "/") for f in foundations]
        colors_bar = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
        bar_colors = [colors_bar[i % len(colors_bar)] for i in range(len(foundations))]
        ax_bar.bar(range(len(foundations)), counts, color=bar_colors)
        ax_bar.set_xticks(range(len(foundations)))
        ax_bar.set_xticklabels(bar_labels, fontsize=7, rotation=30, ha="right")
        ax_bar.set_ylabel("Count")
        ax_bar.set_title("Moral Content by Foundation")
        ax_bar.grid(True, alpha=0.3, axis="y")
    else:
        ax_bar.text(0.5, 0.5, "No moral samples", ha="center", va="center",
                    transform=ax_bar.transAxes)
        ax_bar.set_title("Foundation Breakdown")

    fig.tight_layout()

    png_path = output_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved mixing distribution plot: %s", png_path)

    json_path = output_dir / f"{filename_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path


def plot_training_monitoring(
    session: MonitoringSession,
    output_dir: str | Path = "outputs",
    *,
    filename_prefix: str = "training_monitoring",
    show: bool = False,
) -> Path:
    """Plot moral probing metrics over training steps.

    Produces a two-panel plot: top panel shows peak accuracy and encoding
    breadth over training, bottom panel shows onset layer and peak layer.

    Args:
        session: A MonitoringSession from ProbeMonitor.
        output_dir: Directory for output files (created if needed).
        filename_prefix: Output filename prefix.
        show: If ``True``, call ``plt.show()``.

    Returns:
        Path to the saved PNG file.
    """
    if not session.snapshots:
        raise ValueError("MonitoringSession has no snapshots")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [s.step for s in session.snapshots]
    peak_accs = [s.peak_accuracy or 0.0 for s in session.snapshots]
    breadths = [s.moral_encoding_breadth or 0.0 for s in session.snapshots]
    depths = [s.moral_encoding_depth or 1.0 for s in session.snapshots]
    onset_layers = [s.onset_layer for s in session.snapshots]
    peak_layers = [s.peak_layer for s in session.snapshots]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: accuracy and breadth
    ax_top.plot(steps, peak_accs, "o-", color="#2196F3", linewidth=2,
                markersize=4, label="Peak Accuracy")
    ax_top.plot(steps, breadths, "s-", color="#4CAF50", linewidth=2,
                markersize=4, label="Encoding Breadth")
    ax_top.set_ylabel("Value")
    ax_top.set_ylim(0, 1.05)
    ax_top.legend(loc="lower right")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(f"Training Monitoring — {session.model_name}")

    # Bottom: layer positions
    onset_valid = [(s, l) for s, l in zip(steps, onset_layers) if l is not None]
    peak_valid = [(s, l) for s, l in zip(steps, peak_layers) if l is not None]
    if onset_valid:
        ax_bot.plot(*zip(*onset_valid), "D-", color="#FF9800", linewidth=2,
                    markersize=5, label="Onset Layer")
    if peak_valid:
        ax_bot.plot(*zip(*peak_valid), "*-", color="#F44336", linewidth=2,
                    markersize=7, label="Peak Layer")
    ax_bot.set_xlabel("Training Step")
    ax_bot.set_ylabel("Layer")
    ax_bot.legend(loc="upper right")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()

    png_path = output_dir / f"{filename_prefix}.png"
    fig.savefig(png_path, dpi=150)
    logger.info("Saved training monitoring plot: %s", png_path)

    json_path = output_dir / f"{filename_prefix}.json"
    with open(json_path, "w") as f:
        json.dump(session.to_dict(), f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    if show:
        plt.show()
    plt.close(fig)

    return png_path
