#!/usr/bin/env python3
"""Phase C1: Dense phase-transition mapping — all 37 OLMo-2 1B checkpoints.

Tests H7 (The Phase Transition Has Internal Structure) by running
LayerWiseMoralProbe, FoundationSpecificProbe, and MoralFragilityTest on all
37 early-training checkpoints at 1K-step intervals.

Target model: allenai/OLMo-2-0425-1B-early-training (1B params, 16 layers)
Checkpoints: step 0 to step 36000 at 1000-step intervals (37 total)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory
Estimated runtime: ~15-20 minutes total

Outputs (in papers/accuracy_vs_fragility/outputs/phase_c1/):
    - Per-step directories with probe results (JSON + PNG)
    - Figure 7: High-resolution phase transition heatmap
    - Foundation emergence across all 37 checkpoints
    - Fragility evolution across all 37 checkpoints
    - phase_c1_summary.json

Usage:
    # Run full experiment
    python papers/accuracy_vs_fragility/scripts/phase_c1.py

    # Quick test with reduced dataset
    python papers/accuracy_vs_fragility/scripts/phase_c1.py --dataset-target 10

    # Resume from a specific checkpoint
    python papers/accuracy_vs_fragility/scripts/phase_c1.py --resume-from stage1-step18000

    # Run only a subset of probes
    python papers/accuracy_vs_fragility/scripts/phase_c1.py --probes layer foundation
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"
ALL_PROBES = ["layer", "foundation", "fragility"]


def _clear_memory() -> None:
    """Free GPU/MPS memory."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _parse_step(revision: str) -> int | None:
    """Extract step number from a revision string like 'stage1-step9000-tokens19B'."""
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def _get_all_revisions() -> list[tuple[int, str]]:
    """List all step-based revisions for the early-training model, sorted by step."""
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    all_revisions = list_available_revisions(REPO_ID)
    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        step = _parse_step(rev)
        if step is not None:
            step_revisions.append((step, rev))
    step_revisions.sort(key=lambda x: x[0])
    return step_revisions


def run_probes_on_checkpoint(
    model,
    dataset,
    step: int,
    probes_to_run: list[str],
    output_dir: Path,
) -> dict[str, dict]:
    """Run all applicable probes on a single loaded model checkpoint."""
    results: dict[str, dict] = {}
    step_dir = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    if "layer" in probes_to_run:
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.viz import plot_layer_probing

        probe = LayerWiseMoralProbe(dataset=dataset)
        t0 = time.time()
        result = probe.run(model)
        elapsed = time.time() - t0

        plot_layer_probing(result, output_dir=step_dir)
        logger.info(
            "  LayerWiseMoralProbe: %.1fs, peak=%.1f%% @ layer %d",
            elapsed, result.peak_accuracy * 100, result.peak_layer,
        )
        results["layer_probing"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    if "foundation" in probes_to_run:
        from deepsteer.benchmarks.representational.foundation_probes import (
            FoundationSpecificProbe,
        )
        from deepsteer.viz import plot_foundation_probes

        probe = FoundationSpecificProbe(dataset=dataset)
        t0 = time.time()
        result = probe.run(model)
        elapsed = time.time() - t0

        plot_foundation_probes(result, output_dir=step_dir)
        logger.info("  FoundationSpecificProbe: %.1fs", elapsed)
        results["foundation_probes"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    if "fragility" in probes_to_run:
        from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
        from deepsteer.viz import plot_fragility

        test = MoralFragilityTest(dataset=dataset)
        t0 = time.time()
        result = test.run(model)
        elapsed = time.time() - t0

        plot_fragility(result, output_dir=step_dir)
        logger.info(
            "  MoralFragilityTest: %.1fs, mean_critical=%.2f",
            elapsed, result.mean_critical_noise or 0.0,
        )
        results["fragility"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    return results


def _reload_step_results(step: int, output_dir: Path) -> dict[str, dict]:
    """Reload previously saved probe results for a checkpoint step."""
    from deepsteer.core.types import (
        AccessTier,
        FoundationLayerProbeScore,
        FoundationProbingResult,
        FragilityLayerScore,
        FragilityResult,
        LayerProbeScore,
        LayerProbingResult,
        ModelInfo,
        MoralFoundation,
    )

    step_dir = output_dir / f"step_{step:07d}"
    results: dict[str, dict] = {}

    for json_file in step_dir.glob("*.json"):
        if json_file.name == "step_summary.json":
            continue

        with open(json_file) as f:
            data = json.load(f)

        bname = data.get("benchmark_name", "")
        mi = data.get("model_info", {})
        model_info = ModelInfo(
            name=mi.get("name", ""),
            provider=mi.get("provider", ""),
            access_tier=AccessTier(mi.get("access_tier", 3)),
            n_layers=mi.get("n_layers"),
            n_params=mi.get("n_params"),
            checkpoint_step=mi.get("checkpoint_step"),
        )

        if bname == "layer_wise_moral_probe":
            layer_scores = [
                LayerProbeScore(layer=ls["layer"], accuracy=ls["accuracy"], loss=ls["loss"])
                for ls in data.get("layer_scores", [])
            ]
            result = LayerProbingResult(
                benchmark_name=bname,
                model_info=model_info,
                layer_scores=layer_scores,
                onset_layer=data.get("onset_layer"),
                peak_layer=data.get("peak_layer"),
                peak_accuracy=data.get("peak_accuracy"),
                moral_encoding_depth=data.get("moral_encoding_depth"),
                moral_encoding_breadth=data.get("moral_encoding_breadth"),
                checkpoint_step=data.get("checkpoint_step"),
            )
            results["layer_probing"] = {"result": result, "elapsed_s": 0}

        elif bname == "foundation_specific_probe":
            fls = [
                FoundationLayerProbeScore(
                    foundation=MoralFoundation(s["foundation"]),
                    layer=s["layer"],
                    accuracy=s["accuracy"],
                    loss=s["loss"],
                    n_pairs=s["n_pairs"],
                )
                for s in data.get("foundation_layer_scores", [])
            ]
            result = FoundationProbingResult(
                benchmark_name=bname,
                model_info=model_info,
                foundation_layer_scores=fls,
                per_foundation_summary=data.get("per_foundation_summary", {}),
            )
            results["foundation_probes"] = {"result": result, "elapsed_s": 0}

        elif bname == "moral_fragility_test":
            layer_scores = [
                FragilityLayerScore(
                    layer=ls["layer"],
                    baseline_accuracy=ls["baseline_accuracy"],
                    accuracy_by_noise={float(k): v for k, v in ls["accuracy_by_noise"].items()},
                    critical_noise=ls["critical_noise"],
                )
                for ls in data.get("layer_scores", [])
            ]
            result = FragilityResult(
                benchmark_name=bname,
                model_info=model_info,
                layer_scores=layer_scores,
                noise_levels=data.get("noise_levels", []),
                mean_critical_noise=data.get("mean_critical_noise"),
                most_fragile_layer=data.get("most_fragile_layer"),
                most_robust_layer=data.get("most_robust_layer"),
            )
            results["fragility"] = {"result": result, "elapsed_s": 0}

    return results


def generate_aggregate_plots(
    all_results: dict[int, dict],
    output_dir: Path,
    probes: list[str],
) -> None:
    """Generate cross-checkpoint aggregate plots for C1."""
    import matplotlib
    matplotlib.use("Agg")

    if "layer" in probes:
        _plot_phase_transition_heatmap(all_results, output_dir)
        _plot_depth_breadth_curves(all_results, output_dir)

    if "foundation" in probes:
        _plot_foundation_emergence(all_results, output_dir)

    if "fragility" in probes:
        _plot_fragility_evolution(all_results, output_dir)


def _plot_phase_transition_heatmap(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Figure 7: High-resolution phase transition heatmap.

    X-axis: training step (0-36K at 1K intervals)
    Y-axis: layer index (0-15)
    Color: probing accuracy
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    from deepsteer.core.types import CheckpointTrajectoryResult, LayerProbingResult

    sorted_steps = sorted(s for s in all_results if "layer_probing" in all_results[s])
    if not sorted_steps:
        logger.warning("No layer probing results for phase transition heatmap")
        return

    trajectory: list[LayerProbingResult] = []
    steps_used: list[int] = []
    for step in sorted_steps:
        result = all_results[step]["layer_probing"]["result"]
        trajectory.append(result)
        steps_used.append(step)

    n_layers = len(trajectory[0].layer_scores)
    n_checkpoints = len(trajectory)
    matrix = np.zeros((n_layers, n_checkpoints))
    for col, probing_result in enumerate(trajectory):
        for score in probing_result.layer_scores:
            matrix[score.layer, col] = score.accuracy

    # Find inflection point: step with largest accuracy jump (mean across layers)
    mean_acc_per_step = matrix.mean(axis=0)
    if len(mean_acc_per_step) > 1:
        diffs = np.diff(mean_acc_per_step)
        inflection_idx = int(np.argmax(diffs)) + 1
        inflection_step = steps_used[inflection_idx]
        inflection_acc = mean_acc_per_step[inflection_idx]
    else:
        inflection_idx = None
        inflection_step = None
        inflection_acc = None

    step_labels = [f"{s//1000}K" if s > 0 else "0" for s in steps_used]
    layer_labels = [str(i) for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(14, n_checkpoints * 0.5), 7))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=step_labels,
        yticklabels=layer_labels,
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        annot=False,
        cbar_kws={"label": "Probing Accuracy"},
    )

    # Annotate inflection point
    if inflection_idx is not None:
        ax.axvline(x=inflection_idx + 0.5, color="white", linestyle="--", linewidth=2, alpha=0.8)
        ax.text(
            inflection_idx + 0.5, -0.8,
            f"Inflection\n(step {inflection_step})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="black",
        )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)
    ax.set_title(
        f"Figure 7: High-Resolution Moral Phase Transition — {REPO_ID}\n"
        f"(37 checkpoints, 1K-step intervals)",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()

    png_path = output_dir / "c1_phase_transition_heatmap.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Figure 7 (phase transition heatmap): %s", png_path)

    # Also save via the standard trajectory viz for consistency
    from deepsteer.viz import plot_checkpoint_trajectory

    traj_result = CheckpointTrajectoryResult(
        benchmark_name="checkpoint_trajectory_probe",
        model_info=trajectory[0].model_info,
        trajectory=trajectory,
        checkpoint_steps=steps_used,
        metadata={"n_checkpoints": len(steps_used), "experiment": "C1"},
    )
    plot_checkpoint_trajectory(traj_result, output_dir=output_dir)

    # Save inflection data
    inflection_data = {
        "inflection_step": inflection_step,
        "inflection_mean_accuracy": round(float(inflection_acc), 4) if inflection_acc else None,
        "mean_accuracy_by_step": {
            str(s): round(float(a), 4) for s, a in zip(steps_used, mean_acc_per_step)
        },
    }
    with open(output_dir / "c1_inflection_analysis.json", "w") as f:
        json.dump(inflection_data, f, indent=2)


def _plot_depth_breadth_curves(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Plot moral encoding depth and breadth over all 37 checkpoints."""
    import matplotlib.pyplot as plt

    sorted_steps = sorted(s for s in all_results if "layer_probing" in all_results[s])
    if not sorted_steps:
        return

    steps = []
    depths = []
    breadths = []
    peak_accs = []
    for step in sorted_steps:
        r = all_results[step]["layer_probing"]["result"]
        steps.append(step)
        depths.append(r.moral_encoding_depth)
        breadths.append(r.moral_encoding_breadth)
        peak_accs.append(r.peak_accuracy)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Top: peak accuracy and breadth
    ax1.plot(steps, peak_accs, "o-", color="#2196F3", linewidth=2, markersize=4,
             label="Peak Accuracy")
    ax1.plot(steps, breadths, "s-", color="#4CAF50", linewidth=2, markersize=4,
             label="Encoding Breadth")
    ax1.set_ylabel("Value")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"Moral Encoding Metrics Over Training — {REPO_ID}")

    # Bottom: depth (onset layer / n_layers)
    ax2.plot(steps, depths, "D-", color="#FF9800", linewidth=2, markersize=4,
             label="Encoding Depth (lower = earlier onset)")
    ax2.set_xlabel("Training Step")
    ax2.set_ylabel("Depth (onset_layer / n_layers)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = output_dir / "c1_depth_breadth_curves.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Depth/breadth curves: %s", png_path)


def _plot_foundation_emergence(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Plot foundation onset layer vs training step across all 37 checkpoints."""
    import matplotlib.pyplot as plt

    from deepsteer.core.types import MoralFoundation

    sorted_steps = sorted(s for s in all_results if "foundation_probes" in all_results[s])
    if not sorted_steps:
        return

    foundation_onsets: dict[str, list[tuple[int, int | None]]] = {
        f.value: [] for f in MoralFoundation
    }
    foundation_peaks: dict[str, list[tuple[int, float]]] = {
        f.value: [] for f in MoralFoundation
    }

    for step in sorted_steps:
        result = all_results[step]["foundation_probes"]["result"]
        for fname, summary in result.per_foundation_summary.items():
            foundation_onsets[fname].append((step, summary.get("onset_layer")))
            foundation_peaks[fname].append((step, summary.get("peak_accuracy", 0.0)))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]

    # Plot 1: Onset layer over training
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (fname, data) in enumerate(foundation_onsets.items()):
        if not data:
            continue
        steps = [s for s, o in data if o is not None]
        onsets = [o for _, o in data if o is not None]
        if steps:
            label = fname.replace("_", "/")
            ax.plot(steps, onsets, "o-", color=colors[i % len(colors)],
                    linewidth=2, markersize=4, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Onset Layer")
    ax.set_title(f"Foundation Emergence — Onset Layer Over Training — {REPO_ID}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()

    png_path = output_dir / "c1_foundation_emergence.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Foundation emergence: %s", png_path)

    # Plot 2: Peak accuracy over training per foundation
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (fname, data) in enumerate(foundation_peaks.items()):
        if not data:
            continue
        steps = [s for s, _ in data]
        accs = [a for _, a in data]
        label = fname.replace("_", "/")
        ax.plot(steps, accs, "o-", color=colors[i % len(colors)],
                linewidth=2, markersize=4, label=label)

    ax.axhline(y=0.6, color="#9E9E9E", linestyle="--", linewidth=1, label="Onset threshold (60%)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Peak Probing Accuracy")
    ax.set_title(f"Foundation Peak Accuracy Over Training — {REPO_ID}")
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / "c1_foundation_peak_accuracy.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Foundation peak accuracy: %s", png_path)


def _plot_fragility_evolution(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Plot fragility evolution across all 37 checkpoints.

    Groups layers into early (0-5), mid (6-10), late (11-15) for the 16-layer model.
    """
    import matplotlib.pyplot as plt

    sorted_steps = sorted(s for s in all_results if "fragility" in all_results[s])
    if not sorted_steps:
        return

    # Layer groups for 16-layer model
    groups = {
        "early (0-5)": range(0, 6),
        "mid (6-10)": range(6, 11),
        "late (11-15)": range(11, 16),
    }
    group_critical: dict[str, list[tuple[int, float]]] = {g: [] for g in groups}

    for step in sorted_steps:
        result = all_results[step]["fragility"]["result"]
        for group_name, layer_range in groups.items():
            criticals = []
            for ls in result.layer_scores:
                if ls.layer in layer_range and ls.critical_noise is not None:
                    criticals.append(ls.critical_noise)
            if criticals:
                group_critical[group_name].append((step, float(np.mean(criticals))))

    colors = {"early (0-5)": "#F44336", "mid (6-10)": "#FF9800", "late (11-15)": "#4CAF50"}

    fig, ax = plt.subplots(figsize=(12, 5))
    for group_name, data in group_critical.items():
        if not data:
            continue
        steps = [s for s, _ in data]
        criticals = [c for _, c in data]
        ax.plot(steps, criticals, "o-", color=colors[group_name],
                linewidth=2, markersize=4, label=group_name)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Critical Noise (higher = more robust)")
    ax.set_title(f"Moral Encoding Robustness Over Training — {REPO_ID}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / "c1_fragility_evolution.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Fragility evolution: %s", png_path)

    # Also plot mean critical noise (all layers) over training
    fig, ax = plt.subplots(figsize=(12, 5))
    steps = []
    mean_criticals = []
    for step in sorted_steps:
        r = all_results[step]["fragility"]["result"]
        steps.append(step)
        mean_criticals.append(r.mean_critical_noise or 0.0)

    ax.plot(steps, mean_criticals, "o-", color="#2196F3", linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Critical Noise (all layers)")
    ax.set_title(f"Overall Moral Robustness Over Training — {REPO_ID}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / "c1_mean_critical_noise.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Mean critical noise curve: %s", png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C1: Dense phase-transition mapping on OLMo-2 1B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="papers/accuracy_vs_fragility/outputs/phase_c1",
        help="Directory for output plots and JSON "
             "(default: papers/accuracy_vs_fragility/outputs/phase_c1).",
    )
    parser.add_argument(
        "--dataset-target", type=int, default=40,
        help="Target pairs per moral foundation (default: 40).",
    )
    parser.add_argument(
        "--probes", nargs="+", default=ALL_PROBES, choices=ALL_PROBES,
        help="Which probes to run (default: all).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Skip checkpoints before this revision (e.g., stage1-step18000).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # -- Build dataset --
    from deepsteer.datasets.pipeline import build_probing_dataset

    print(f"Building probing dataset (target={args.dataset_target} per foundation)...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # -- List all checkpoints --
    print(f"\nListing revisions for {REPO_ID}...")
    all_revisions = _get_all_revisions()
    print(f"Found {len(all_revisions)} checkpoints (step {all_revisions[0][0]} to "
          f"{all_revisions[-1][0]})")

    step_to_rev = {s: r for s, r in all_revisions}
    sorted_steps = [s for s, _ in all_revisions]

    # Handle --resume-from
    resume_step = None
    if args.resume_from:
        resume_step = _parse_step(args.resume_from)
        if resume_step is not None:
            sorted_steps = [s for s in sorted_steps if s >= resume_step]
            print(f"Resuming from step {resume_step}: {len(sorted_steps)} checkpoints remaining")

    print(f"\nPhase C1 plan:")
    print(f"  Model: {REPO_ID}")
    print(f"  Checkpoints: {len(sorted_steps)}")
    print(f"  Probes per checkpoint: {', '.join(args.probes)}")
    print(f"  Steps: {sorted_steps[0]}–{sorted_steps[-1]}")

    # Time estimate: ~20s per checkpoint for all 3 probes on 1B model
    est_per_checkpoint = 5 * len(args.probes)  # ~5s per probe
    est_total = est_per_checkpoint * len(sorted_steps) + len(sorted_steps) * 10  # +10s load
    print(f"  Estimated time: ~{est_total // 60} minutes")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plan
    plan_data = {
        "repo_id": REPO_ID,
        "experiment": "C1",
        "hypothesis": "H7 — The Phase Transition Has Internal Structure",
        "probes": args.probes,
        "total_checkpoints": len(sorted_steps),
        "steps": sorted_steps,
        "dataset_target": args.dataset_target,
    }
    with open(output_dir / "phase_c1_plan.json", "w") as f:
        json.dump(plan_data, f, indent=2)

    # -- Load results from previously completed steps (for resume) --
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    all_results: dict[int, dict] = {}

    if resume_step is not None:
        all_plan_steps = [s for s, _ in all_revisions]
        skipped_steps = [s for s in all_plan_steps if s < resume_step]
        for s in skipped_steps:
            step_json = output_dir / f"step_{s:07d}" / "step_summary.json"
            if step_json.exists():
                all_results[s] = _reload_step_results(s, output_dir)
                logger.info("Reloaded results from step %d", s)
        print(f"Reloaded {len(all_results)} previously completed checkpoints")

    # -- Iterate over checkpoints --
    total_t0 = time.time()

    for i, step in enumerate(sorted_steps):
        revision = step_to_rev[step]

        print(f"\n{'='*60}")
        print(f"Checkpoint {i+1}/{len(sorted_steps)}: {revision} (step {step})")
        print(f"{'='*60}")

        load_t0 = time.time()
        model = WhiteBoxModel(
            REPO_ID,
            revision=revision,
            device=args.device,
            access_tier=AccessTier.CHECKPOINTS,
            checkpoint_step=step,
        )
        load_elapsed = time.time() - load_t0
        print(f"  Model loaded in {load_elapsed:.1f}s")

        step_results = run_probes_on_checkpoint(
            model, dataset, step, args.probes, output_dir,
        )
        all_results[step] = step_results

        # Save incremental step summary
        step_summary: dict = {
            "step": step,
            "revision": revision,
            "probes_run": args.probes,
        }
        for probe_name, probe_data in step_results.items():
            step_summary[probe_name] = {"elapsed_s": probe_data["elapsed_s"]}
            r = probe_data["result"]
            if hasattr(r, "peak_accuracy"):
                step_summary[probe_name]["peak_accuracy"] = round(r.peak_accuracy, 4)
                step_summary[probe_name]["onset_layer"] = r.onset_layer
                step_summary[probe_name]["moral_encoding_depth"] = round(
                    r.moral_encoding_depth, 4
                )
                step_summary[probe_name]["moral_encoding_breadth"] = round(
                    r.moral_encoding_breadth, 4
                )
            if hasattr(r, "mean_critical_noise"):
                step_summary[probe_name]["mean_critical_noise"] = r.mean_critical_noise

        step_json = output_dir / f"step_{step:07d}" / "step_summary.json"
        with open(step_json, "w") as f:
            json.dump(step_summary, f, indent=2)

        # Free memory
        del model
        _clear_memory()

        elapsed_so_far = time.time() - total_t0
        remaining = len(sorted_steps) - (i + 1)
        avg_per_step = elapsed_so_far / (i + 1)
        eta = avg_per_step * remaining
        print(f"  Elapsed: {elapsed_so_far/60:.1f}min, ETA: {eta/60:.1f}min "
              f"({remaining} checkpoints left)")

    total_elapsed = time.time() - total_t0

    # -- Generate aggregate plots --
    print(f"\n{'='*60}")
    print("Generating aggregate plots...")
    print(f"{'='*60}")
    generate_aggregate_plots(all_results, output_dir, args.probes)

    # -- Summary --
    print(f"\n{'='*60}")
    print("PHASE C1 SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {REPO_ID}")
    print(f"Total time: {total_elapsed/60:.1f} min")
    print(f"Checkpoints processed: {len(sorted_steps)}")
    print(f"Output: {output_dir}")

    # Key metrics trajectory
    if "layer" in args.probes:
        print(f"\nLayer Probing Trajectory (H7):")
        for step in sorted(all_results.keys()):
            if "layer_probing" in all_results[step]:
                r = all_results[step]["layer_probing"]["result"]
                print(f"  step {step:>6d}: peak={r.peak_accuracy:.1%}, "
                      f"onset={r.onset_layer}, depth={r.moral_encoding_depth:.3f}, "
                      f"breadth={r.moral_encoding_breadth:.3f}")

    if "fragility" in args.probes:
        print(f"\nFragility Trajectory:")
        for step in sorted(all_results.keys()):
            if "fragility" in all_results[step]:
                r = all_results[step]["fragility"]["result"]
                print(f"  step {step:>6d}: mean_critical={r.mean_critical_noise:.3f}")

    # Save summary
    summary_data = {
        "repo_id": REPO_ID,
        "experiment": "C1",
        "hypothesis": "H7",
        "total_elapsed_s": round(total_elapsed, 1),
        "checkpoints_processed": len(sorted_steps),
        "probes": args.probes,
        "steps_processed": sorted(all_results.keys()),
    }
    summary_path = output_dir / "phase_c1_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
