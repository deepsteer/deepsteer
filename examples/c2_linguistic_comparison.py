#!/usr/bin/env python3
"""Phase C2: Moral vs. Linguistic Emergence Timing.

Compares emergence timing of moral encoding against sentiment and syntax
probes across 37 early-training checkpoints of OLMo-2 1B.

Key question: Does moral encoding emerge at the same training step as
general linguistic competence, or does it lag behind?

If moral and sentiment/syntax probes all rise at the same step, then
"moral emergence" is just a byproduct of learning language.  If moral
lags, it suggests moral concepts require specific data exposure —
validating pre-training data curation as an alignment lever.

Target model: allenai/OLMo-2-0425-1B-early-training (1B params, 16 layers)
Checkpoints: step 0 to step 36000 at 1000-step intervals (37 total)
Probes per checkpoint: moral, sentiment, syntax (shared activations)
Hardware: M4 Pro Mac, MPS
Estimated runtime: ~2-3 hours

Outputs (in outputs/phase_c2/):
    - Per-step JSON with all probe results
    - Figure 8: Emergence timing curves (mean accuracy vs. step)
    - Figure 8b: Per-layer heatmaps (1x3 subplot)
    - phase_c2_summary.json

Usage:
    # Full run
    python examples/c2_linguistic_comparison.py

    # Quick test with reduced dataset
    python examples/c2_linguistic_comparison.py --dataset-target 10

    # Resume from a specific checkpoint
    python examples/c2_linguistic_comparison.py --resume-from step18000
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
import torch

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"
PROBE_NAMES = ["moral", "sentiment", "syntax"]

# Onset threshold: first step where mean accuracy exceeds this value.
ONSET_ACCURACY = 0.70


def _clear_memory() -> None:
    """Free GPU/MPS memory."""
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
    """List all step-based revisions, sorted by step number."""
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    all_revisions = list_available_revisions(REPO_ID)
    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        step = _parse_step(rev)
        if step is not None:
            step_revisions.append((step, rev))
    step_revisions.sort(key=lambda x: x[0])
    return step_revisions


def build_all_datasets(
    target_per_foundation: int = 40,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]]:
    """Build moral, sentiment, and syntax datasets.

    Returns:
        Dict mapping probe name to (train_pairs, test_pairs), where each
        pair is a (positive_text, negative_text) tuple.
    """
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.datasets.sentiment_pairs import get_sentiment_dataset
    from deepsteer.datasets.syntax_pairs import get_syntax_dataset

    # Moral dataset: ProbingPair.moral = class 1, ProbingPair.neutral = class 0
    moral_ds = build_probing_dataset(
        target_per_foundation=target_per_foundation, seed=seed,
    )
    moral_train = [(p.moral, p.neutral) for p in moral_ds.train]
    moral_test = [(p.moral, p.neutral) for p in moral_ds.test]

    # Sentiment dataset: positive = class 1, negative = class 0
    sent_train, sent_test = get_sentiment_dataset(
        test_fraction=test_fraction, seed=seed,
    )

    # Syntax dataset: grammatical = class 1, ungrammatical = class 0
    syn_train, syn_test = get_syntax_dataset(
        test_fraction=test_fraction, seed=seed,
    )

    return {
        "moral": (moral_train, moral_test),
        "sentiment": (sent_train, sent_test),
        "syntax": (syn_train, syn_test),
    }


def collect_all_texts(
    datasets: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]],
) -> list[str]:
    """Collect all unique texts across all datasets, sorted for determinism."""
    texts: set[str] = set()
    for _name, (train, test) in datasets.items():
        for pos, neg in train + test:
            texts.add(pos)
            texts.add(neg)
    return sorted(texts)


def run_probes_on_checkpoint(
    model,
    datasets: dict[str, tuple[list[tuple[str, str]], list[tuple[str, str]]]],
    step: int,
    output_dir: Path,
) -> dict[str, dict]:
    """Run all 3 probes on a loaded model, sharing activations.

    Collects activations for all texts from all datasets in a single pass,
    then trains separate linear classifiers for each probe type.
    """
    from deepsteer.benchmarks.representational.general_probe import (
        GeneralLinearProbe,
        collect_activations_batch,
    )

    n_layers = model.info.n_layers
    assert n_layers is not None

    # Collect activations for ALL texts across ALL datasets — single pass
    all_texts = collect_all_texts(datasets)

    t0 = time.time()
    cache = collect_activations_batch(model, all_texts)
    act_time = time.time() - t0
    logger.info(
        "  Activation collection: %.1fs for %d unique texts", act_time, len(all_texts),
    )

    # Run each probe using the shared cache
    results: dict[str, dict] = {}
    for probe_name in PROBE_NAMES:
        train_pairs, test_pairs = datasets[probe_name]
        probe = GeneralLinearProbe(probe_name=f"{probe_name}_probe")

        t0 = time.time()
        result = probe.run_on_cached_activations(
            cache, train_pairs, test_pairs, model.info, n_layers,
            checkpoint_step=step,
        )
        probe_time = time.time() - t0

        results[probe_name] = {
            "result": result,
            "elapsed_s": round(probe_time, 1),
        }

        mean_acc = float(np.mean([s.accuracy for s in result.layer_scores]))
        logger.info(
            "  %s probe: %.1fs, peak=%.1f%%, mean=%.1f%%",
            probe_name, probe_time, result.peak_accuracy * 100, mean_acc * 100,
        )

    results["_activation_time_s"] = round(act_time, 1)  # type: ignore[assignment]

    # Save per-step results as JSON
    step_dir = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    step_data: dict = {"step": step}
    for probe_name in PROBE_NAMES:
        r = results[probe_name]["result"]
        step_data[probe_name] = {
            "layer_scores": [
                {
                    "layer": s.layer,
                    "accuracy": round(s.accuracy, 4),
                    "loss": round(s.loss, 4),
                }
                for s in r.layer_scores
            ],
            "peak_accuracy": round(r.peak_accuracy, 4),
            "peak_layer": r.peak_layer,
            "onset_layer": r.onset_layer,
            "mean_accuracy": round(
                float(np.mean([s.accuracy for s in r.layer_scores])), 4,
            ),
        }

    with open(step_dir / "c2_probes.json", "w") as f:
        json.dump(step_data, f, indent=2)

    return results


def _reload_step_results(step: int, output_dir: Path) -> dict[str, dict] | None:
    """Reload previously saved C2 results for a checkpoint step."""
    from deepsteer.core.types import AccessTier, LayerProbeScore, LayerProbingResult, ModelInfo

    step_dir = output_dir / f"step_{step:07d}"
    json_path = step_dir / "c2_probes.json"
    if not json_path.exists():
        return None

    with open(json_path) as f:
        data = json.load(f)

    results: dict[str, dict] = {}
    for probe_name in PROBE_NAMES:
        if probe_name not in data:
            continue
        pdata = data[probe_name]
        layer_scores = [
            LayerProbeScore(
                layer=ls["layer"], accuracy=ls["accuracy"], loss=ls["loss"],
            )
            for ls in pdata["layer_scores"]
        ]

        n_layers = len(layer_scores)
        onset_layer = pdata.get("onset_layer")
        peak_layer = pdata.get("peak_layer", 0)
        peak_accuracy = pdata.get("peak_accuracy", 0.0)
        depth = (onset_layer / n_layers) if onset_layer is not None else 1.0
        breadth = sum(1 for s in layer_scores if s.accuracy >= 0.6) / n_layers

        result = LayerProbingResult(
            benchmark_name=f"{probe_name}_probe",
            model_info=ModelInfo(
                name=REPO_ID,
                provider="allenai",
                access_tier=AccessTier.CHECKPOINTS,
                n_layers=n_layers,
                checkpoint_step=step,
            ),
            layer_scores=layer_scores,
            onset_layer=onset_layer,
            peak_layer=peak_layer,
            peak_accuracy=peak_accuracy,
            moral_encoding_depth=depth,
            moral_encoding_breadth=breadth,
            checkpoint_step=step,
        )
        results[probe_name] = {"result": result, "elapsed_s": 0}

    return results if len(results) == len(PROBE_NAMES) else None


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def generate_figure_8(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Figure 8: Moral vs. Linguistic Emergence Timing.

    X-axis: training step (0-36K)
    Y-axis: mean probing accuracy across all layers
    Three curves: moral, sentiment, syntax
    Vertical dashed lines at the onset step for each probe type.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sorted_steps = sorted(all_results.keys())

    # Compute mean accuracy per probe per step
    curves: dict[str, tuple[list[int], list[float]]] = {}
    for probe_name in PROBE_NAMES:
        steps_with_data: list[int] = []
        mean_accs: list[float] = []
        for step in sorted_steps:
            if probe_name in all_results[step]:
                r = all_results[step][probe_name]["result"]
                mean_acc = float(np.mean([s.accuracy for s in r.layer_scores]))
                steps_with_data.append(step)
                mean_accs.append(mean_acc)
        curves[probe_name] = (steps_with_data, mean_accs)

    colors = {"moral": "#F44336", "sentiment": "#2196F3", "syntax": "#4CAF50"}
    labels = {"moral": "Moral", "sentiment": "Sentiment", "syntax": "Syntax"}

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot curves
    onset_steps: dict[str, int] = {}
    for probe_name in PROBE_NAMES:
        steps, accs = curves[probe_name]
        if not steps:
            continue
        ax.plot(
            steps, accs, "o-", color=colors[probe_name], linewidth=2,
            markersize=4, label=labels[probe_name],
        )

        # Find onset step (first exceeding ONSET_ACCURACY)
        for s, a in zip(steps, accs):
            if a >= ONSET_ACCURACY:
                onset_steps[probe_name] = s
                break

    # Vertical dashed lines at onset steps
    for probe_name, onset_step in onset_steps.items():
        ax.axvline(
            x=onset_step, color=colors[probe_name], linestyle="--",
            linewidth=1.5, alpha=0.7,
        )
        ax.text(
            onset_step, 0.42, f"{labels[probe_name]}\nonset\n(step {onset_step})",
            ha="center", va="bottom", fontsize=8, color=colors[probe_name],
            fontweight="bold",
        )

    # Annotate gap between earliest and latest onset
    if len(onset_steps) >= 2:
        earliest = min(onset_steps.values())
        latest = max(onset_steps.values())
        gap = latest - earliest
        if gap > 0:
            mid_y = 0.97
            ax.annotate(
                "", xy=(latest, mid_y), xytext=(earliest, mid_y),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.5),
            )
            ax.text(
                (earliest + latest) / 2, mid_y + 0.02,
                f"Gap: {gap} steps",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
            )

    ax.axhline(
        y=ONSET_ACCURACY, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.5,
        label=f"Onset threshold ({ONSET_ACCURACY:.0%})",
    )
    ax.axhline(
        y=0.5, color="#BDBDBD", linestyle=":", linewidth=1, alpha=0.3,
        label="Chance (50%)",
    )

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Mean Probing Accuracy (all layers)", fontsize=12)
    ax.set_title(
        "Figure 8: Emergence Timing — Moral vs. Sentiment vs. Syntax Probes\n"
        f"{REPO_ID}",
        fontsize=13,
    )
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        path = output_dir / f"c2_emergence_timing.{fmt}"
        fig.savefig(path, dpi=150)
        logger.info("Figure 8 (%s): %s", fmt, path)
    plt.close(fig)

    # Save onset data as companion JSON
    onset_data = {
        "onset_steps": onset_steps,
        "onset_threshold": ONSET_ACCURACY,
        "curves": {
            name: {
                "steps": steps,
                "mean_accuracies": [round(a, 4) for a in accs],
            }
            for name, (steps, accs) in curves.items()
        },
    }
    with open(output_dir / "c2_emergence_timing.json", "w") as f:
        json.dump(onset_data, f, indent=2)


def generate_heatmaps(
    all_results: dict[int, dict],
    output_dir: Path,
) -> None:
    """Figure 8b: Per-layer accuracy heatmaps, 1x3 subplot.

    Shows layer x step accuracy heatmaps for all three probe types
    side-by-side, enabling visual comparison of emergence patterns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sorted_steps = sorted(all_results.keys())

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    probe_labels = {"moral": "Moral", "sentiment": "Sentiment", "syntax": "Syntax"}

    for ax_idx, probe_name in enumerate(PROBE_NAMES):
        ax = axes[ax_idx]

        # Collect data for this probe
        steps_with_data: list[int] = []
        for step in sorted_steps:
            if probe_name in all_results[step]:
                steps_with_data.append(step)

        if not steps_with_data:
            ax.text(
                0.5, 0.5, f"No {probe_name} data", ha="center", va="center",
                transform=ax.transAxes,
            )
            continue

        # Get n_layers from first result
        first_result = all_results[steps_with_data[0]][probe_name]["result"]
        n_layers = len(first_result.layer_scores)

        # Build accuracy matrix: rows = layers, columns = checkpoints
        matrix = np.zeros((n_layers, len(steps_with_data)))
        for col, step in enumerate(steps_with_data):
            r = all_results[step][probe_name]["result"]
            for score in r.layer_scores:
                matrix[score.layer, col] = score.accuracy

        step_labels = [f"{s // 1000}K" if s > 0 else "0" for s in steps_with_data]
        layer_labels = [str(i) for i in range(n_layers)]

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=step_labels, yticklabels=layer_labels if ax_idx == 0 else False,
            cmap="RdYlGn", vmin=0.4, vmax=1.0, annot=False,
            cbar=ax_idx == 2,  # only show colorbar on rightmost plot
            cbar_kws={"label": "Probing Accuracy"} if ax_idx == 2 else {},
        )

        ax.set_xlabel("Training Step")
        if ax_idx == 0:
            ax.set_ylabel("Layer")
        ax.set_title(f"{probe_labels[probe_name]} Probe")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle(
        f"Figure 8b: Per-Layer Emergence Patterns — {REPO_ID}",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    for fmt in ["png", "pdf"]:
        path = output_dir / f"c2_layer_heatmaps.{fmt}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Figure 8b (%s): %s", fmt, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase C2: Moral vs. Linguistic Emergence Timing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--output-dir", default="outputs/phase_c2",
        help="Directory for output plots and JSON (default: outputs/phase_c2).",
    )
    parser.add_argument(
        "--dataset-target", type=int, default=40,
        help="Target pairs per moral foundation (default: 40).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda, mps, cpu).  Auto-detected if omitted.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Skip checkpoints before this revision (e.g., step18000).",
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

    # -- Build all three datasets --
    print("Building probing datasets (moral + sentiment + syntax)...")
    datasets = build_all_datasets(target_per_foundation=args.dataset_target)
    for name, (train, test) in datasets.items():
        print(f"  {name}: {len(train)} train, {len(test)} test pairs")

    # -- List all checkpoints --
    print(f"\nListing revisions for {REPO_ID}...")
    all_revisions = _get_all_revisions()
    print(
        f"Found {len(all_revisions)} checkpoints "
        f"(step {all_revisions[0][0]} to {all_revisions[-1][0]})",
    )

    step_to_rev = {s: r for s, r in all_revisions}
    sorted_steps = [s for s, _ in all_revisions]

    # Handle --resume-from
    resume_step = None
    if args.resume_from:
        resume_step = _parse_step(args.resume_from)
        if resume_step is not None:
            sorted_steps = [s for s in sorted_steps if s >= resume_step]
            print(
                f"Resuming from step {resume_step}: "
                f"{len(sorted_steps)} checkpoints remaining",
            )

    # Estimate runtime
    total_texts = len(collect_all_texts(datasets))
    est_per_ckpt = total_texts * 0.05 + 3 * 3  # ~0.05s/text + ~3s/probe
    est_total = (est_per_ckpt + 15) * len(sorted_steps)  # +15s model load

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPhase C2 plan:")
    print(f"  Model: {REPO_ID}")
    print(f"  Checkpoints: {len(sorted_steps)}")
    print(f"  Total unique texts per checkpoint: {total_texts}")
    print(f"  Probes: {', '.join(PROBE_NAMES)}")
    print(f"  Estimated time: ~{est_total / 60:.0f} minutes")

    # Save plan
    plan = {
        "repo_id": REPO_ID,
        "experiment": "C2",
        "hypothesis": "Moral encoding emerges later than general linguistic competence",
        "probes": PROBE_NAMES,
        "total_checkpoints": len(sorted_steps),
        "steps": sorted_steps,
        "dataset_sizes": {
            name: {"train": len(t), "test": len(e)}
            for name, (t, e) in datasets.items()
        },
        "total_unique_texts_per_checkpoint": total_texts,
        "onset_threshold": ONSET_ACCURACY,
    }
    with open(output_dir / "phase_c2_plan.json", "w") as f:
        json.dump(plan, f, indent=2)

    # -- Load results from previously completed steps (for resume) --
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    all_results: dict[int, dict] = {}

    if resume_step is not None:
        all_plan_steps = [s for s, _ in all_revisions]
        for s in all_plan_steps:
            if s < resume_step:
                prev = _reload_step_results(s, output_dir)
                if prev:
                    all_results[s] = prev
        print(f"Reloaded {len(all_results)} previously completed checkpoints")

    # -- Iterate over checkpoints --
    total_t0 = time.time()

    for i, step in enumerate(sorted_steps):
        revision = step_to_rev[step]

        print(f"\n{'=' * 60}")
        print(f"Checkpoint {i + 1}/{len(sorted_steps)}: {revision} (step {step})")
        print(f"{'=' * 60}")

        # Skip if results already exist (allows crash-resume without --resume-from)
        existing = _reload_step_results(step, output_dir)
        if existing:
            print("  Results already exist, skipping")
            all_results[step] = existing
            continue

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
            model, datasets, step, output_dir,
        )
        all_results[step] = step_results

        # Free memory
        del model
        _clear_memory()

        elapsed_so_far = time.time() - total_t0
        remaining = len(sorted_steps) - (i + 1)
        avg_per_step = elapsed_so_far / (i + 1)
        eta = avg_per_step * remaining
        print(
            f"  Elapsed: {elapsed_so_far / 60:.1f}min, "
            f"ETA: {eta / 60:.1f}min ({remaining} checkpoints left)",
        )

    total_elapsed = time.time() - total_t0

    # -- Generate figures --
    print(f"\n{'=' * 60}")
    print("Generating figures...")
    print(f"{'=' * 60}")

    generate_figure_8(all_results, output_dir)
    generate_heatmaps(all_results, output_dir)

    # -- Print summary --
    print(f"\n{'=' * 60}")
    print("PHASE C2 SUMMARY")
    print(f"{'=' * 60}")
    print(f"Model: {REPO_ID}")
    print(f"Total time: {total_elapsed / 60:.1f} min")
    print(f"Checkpoints processed: {len(sorted_steps)}")
    print(f"Output: {output_dir}")

    # Emergence timing comparison
    print(f"\nEmergence Timing (onset threshold: {ONSET_ACCURACY:.0%}):")
    onset_steps: dict[str, int | None] = {}
    for probe_name in PROBE_NAMES:
        onset_step = None
        for step in sorted(all_results.keys()):
            if probe_name in all_results[step]:
                r = all_results[step][probe_name]["result"]
                mean_acc = float(np.mean([s.accuracy for s in r.layer_scores]))
                if mean_acc >= ONSET_ACCURACY:
                    onset_step = step
                    break
        onset_steps[probe_name] = onset_step
        print(f"  {probe_name:>10s}: onset at step {onset_step}")

    # Key result: the gap
    valid_onsets = [s for s in onset_steps.values() if s is not None]
    if len(valid_onsets) >= 2:
        gap = max(valid_onsets) - min(valid_onsets)
        earliest_name = min(
            (n for n, s in onset_steps.items() if s is not None),
            key=lambda n: onset_steps[n],
        )
        latest_name = max(
            (n for n, s in onset_steps.items() if s is not None),
            key=lambda n: onset_steps[n],
        )
        print(f"\n  Emergence gap: {gap} steps ({earliest_name} -> {latest_name})")
        if gap == 0:
            print("  Interpretation: All probes emerge simultaneously — moral "
                  "encoding may be a byproduct of general language learning.")
        else:
            print(f"  Interpretation: {latest_name} lags by {gap} steps — "
                  "suggests specific data exposure is required.")

    # Per-checkpoint trajectory
    print(f"\nTrajectory (mean accuracy):")
    print(f"  {'step':>7s}  {'moral':>8s}  {'sentiment':>10s}  {'syntax':>8s}")
    for step in sorted(all_results.keys()):
        row = f"  {step:>7d}"
        for probe_name in PROBE_NAMES:
            if probe_name in all_results[step]:
                r = all_results[step][probe_name]["result"]
                mean_acc = float(np.mean([s.accuracy for s in r.layer_scores]))
                row += f"  {mean_acc:>8.1%}"
            else:
                row += f"  {'N/A':>8s}"
        print(row)

    # Save summary JSON
    summary = {
        "repo_id": REPO_ID,
        "experiment": "C2",
        "hypothesis": "Moral encoding emerges later than general linguistic competence",
        "total_elapsed_s": round(total_elapsed, 1),
        "checkpoints_processed": len(sorted_steps),
        "probes": PROBE_NAMES,
        "onset_threshold": ONSET_ACCURACY,
        "onset_steps": {k: v for k, v in onset_steps.items()},
        "dataset_sizes": {
            name: {"train": len(t), "test": len(e)}
            for name, (t, e) in datasets.items()
        },
    }
    summary_path = output_dir / "phase_c2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
