#!/usr/bin/env python3
"""Phase A: Pipeline validation — Moral Emergence Curve on OLMo-2 1B.

Runs all 5 Phase A experiments from RESEARCH_PLAN.md end-to-end on OLMo-2's
early-training checkpoints to validate the pipeline before committing GPU time
to the full OLMo-3 7B sweep.

Experiments:
    A1: LayerWiseMoralProbe — layer accuracy curve (final checkpoint)
    A2: CheckpointTrajectoryProbe — small heatmap (5 evenly spaced checkpoints)
    A3: FoundationSpecificProbe — per-foundation curves (final checkpoint)
    A4: MoralCausalTracer — causal effect heatmap (final checkpoint)
    A5: MoralFragilityTest — noise robustness curves (final checkpoint)

Target model: allenai/OLMo-2-0425-1B-early-training (1B params, 16 layers)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory

Usage:
    # Run all experiments
    python examples/moral_emergence.py --output-dir outputs/phase_a

    # Quick test with reduced dataset and fewer causal prompts
    python examples/moral_emergence.py --output-dir outputs/phase_a \\
        --dataset-target 10 --causal-prompts 10

    # Run specific experiments only
    python examples/moral_emergence.py --experiments A1 A3 A5

    # Skip trajectory (A2) for faster iteration
    python examples/moral_emergence.py --experiments A1 A3 A4 A5

    # Use a different model
    python examples/moral_emergence.py --weights allenai/OLMo-1B-hf
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "allenai/OLMo-2-0425-1B-early-training"
ALL_EXPERIMENTS = ["A1", "A2", "A3", "A4", "A5"]


def _clear_memory() -> None:
    """Free GPU/MPS memory between experiments."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _select_trajectory_revisions(
    all_revisions: list[str],
    n_points: int,
) -> list[str]:
    """Select evenly spaced revisions from the available list.

    Filters to step-based revisions (those matching 'stepNNN'), sorts by step
    number, and picks n_points evenly spaced across the range.
    """
    import re

    step_revisions: list[tuple[int, str]] = []
    for rev in all_revisions:
        match = re.search(r"step(\d+)", rev)
        if match:
            step_revisions.append((int(match.group(1)), rev))

    step_revisions.sort(key=lambda x: x[0])

    if len(step_revisions) <= n_points:
        return [rev for _, rev in step_revisions]

    # Select evenly spaced indices including first and last
    indices = [
        round(i * (len(step_revisions) - 1) / (n_points - 1))
        for i in range(n_points)
    ]
    return [step_revisions[i][1] for i in indices]


def run_a1(model, dataset, output_dir: Path) -> dict:
    """A1: Smoke test — LayerWiseMoralProbe on single checkpoint."""
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.viz import plot_layer_probing

    print("\n" + "=" * 60)
    print("A1: LayerWiseMoralProbe — layer accuracy curve")
    print("=" * 60)

    probe = LayerWiseMoralProbe(dataset=dataset)
    t0 = time.time()
    result = probe.run(model)
    elapsed = time.time() - t0

    png_path = plot_layer_probing(result, output_dir=output_dir)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  onset_layer = {result.onset_layer}")
    print(f"  peak_layer = {result.peak_layer}")
    print(f"  peak_accuracy = {result.peak_accuracy:.1%}")
    print(f"  moral_encoding_depth = {result.moral_encoding_depth:.3f}")
    print(f"  moral_encoding_breadth = {result.moral_encoding_breadth:.3f}")
    print(f"  Plot: {png_path}")

    return {
        "experiment": "A1",
        "probe": "LayerWiseMoralProbe",
        "elapsed_s": round(elapsed, 1),
        "onset_layer": result.onset_layer,
        "peak_layer": result.peak_layer,
        "peak_accuracy": round(result.peak_accuracy, 4),
        "moral_encoding_depth": round(result.moral_encoding_depth, 4),
        "moral_encoding_breadth": round(result.moral_encoding_breadth, 4),
        "plot": str(png_path),
    }


def run_a2(
    model,
    dataset,
    output_dir: Path,
    *,
    n_trajectory_points: int = 5,
    device: str | None = None,
) -> dict:
    """A2: Mini trajectory — CheckpointTrajectoryProbe across checkpoints."""
    import torch

    from deepsteer.benchmarks.representational.trajectory import (
        CheckpointTrajectoryProbe,
        list_available_revisions,
    )
    from deepsteer.viz import plot_checkpoint_trajectory

    print("\n" + "=" * 60)
    print("A2: CheckpointTrajectoryProbe — mini trajectory heatmap")
    print("=" * 60)

    repo_id = model.info.name
    print(f"  Listing revisions for {repo_id}...")
    all_revisions = list_available_revisions(repo_id)
    print(f"  Found {len(all_revisions)} revisions")

    selected = _select_trajectory_revisions(all_revisions, n_trajectory_points)
    print(f"  Selected {len(selected)} checkpoints: {selected}")

    if not selected:
        print("  WARNING: No step-based revisions found. Skipping A2.")
        return {"experiment": "A2", "skipped": True, "reason": "no_step_revisions"}

    # Free the current model before loading checkpoint models
    dtype = model._dtype if hasattr(model, "_dtype") else None

    traj_probe = CheckpointTrajectoryProbe(
        checkpoint_revisions=selected,
        dataset=dataset,
        device=device,
        torch_dtype=dtype,
    )

    t0 = time.time()
    traj_result = traj_probe.run(model)
    elapsed = time.time() - t0

    png_path = plot_checkpoint_trajectory(traj_result, output_dir=output_dir)

    # Summarize
    depth_trajectory = []
    breadth_trajectory = []
    for lr in traj_result.trajectory:
        depth_trajectory.append(round(lr.moral_encoding_depth, 4))
        breadth_trajectory.append(round(lr.moral_encoding_breadth, 4))

    print(f"  Time: {elapsed:.1f}s")
    print(f"  Checkpoints: {traj_result.checkpoint_steps}")
    print(f"  Depth trajectory: {depth_trajectory}")
    print(f"  Breadth trajectory: {breadth_trajectory}")
    print(f"  Plot: {png_path}")

    return {
        "experiment": "A2",
        "probe": "CheckpointTrajectoryProbe",
        "elapsed_s": round(elapsed, 1),
        "n_checkpoints": len(selected),
        "revisions": selected,
        "checkpoint_steps": traj_result.checkpoint_steps,
        "depth_trajectory": depth_trajectory,
        "breadth_trajectory": breadth_trajectory,
        "plot": str(png_path),
    }


def run_a3(model, dataset, output_dir: Path) -> dict:
    """A3: Foundation check — FoundationSpecificProbe per-foundation curves."""
    from deepsteer.benchmarks.representational.foundation_probes import (
        FoundationSpecificProbe,
    )
    from deepsteer.viz import plot_foundation_probes

    print("\n" + "=" * 60)
    print("A3: FoundationSpecificProbe — per-foundation curves")
    print("=" * 60)

    probe = FoundationSpecificProbe(dataset=dataset)
    t0 = time.time()
    result = probe.run(model)
    elapsed = time.time() - t0

    png_path = plot_foundation_probes(result, output_dir=output_dir)

    print(f"  Time: {elapsed:.1f}s")
    for fname, summary in result.per_foundation_summary.items():
        print(
            f"  {fname}: onset={summary['onset_layer']}, "
            f"peak={summary['peak_layer']} ({summary['peak_accuracy']:.1%}), "
            f"depth={summary['depth']:.3f}"
        )
    print(f"  Plot: {png_path}")

    return {
        "experiment": "A3",
        "probe": "FoundationSpecificProbe",
        "elapsed_s": round(elapsed, 1),
        "per_foundation_summary": result.per_foundation_summary,
        "plot": str(png_path),
    }


def run_a4(model, dataset, output_dir: Path, *, max_prompts: int = 40) -> dict:
    """A4: Causal check — MoralCausalTracer causal effect by layer."""
    from deepsteer.benchmarks.representational.causal_tracing import MoralCausalTracer
    from deepsteer.viz import plot_causal_tracing

    print("\n" + "=" * 60)
    print("A4: MoralCausalTracer — causal effect heatmap")
    print("=" * 60)

    tracer = MoralCausalTracer(dataset=dataset, max_prompts=max_prompts)
    t0 = time.time()
    result = tracer.run(model)
    elapsed = time.time() - t0

    png_path = plot_causal_tracing(result, output_dir=output_dir)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  peak_causal_layer = {result.peak_causal_layer}")
    print(f"  peak_mean_indirect_effect = {result.peak_mean_indirect_effect:.4f}")
    print(f"  causal_depth = {result.causal_depth:.3f}")
    print(f"  Plot: {png_path}")

    return {
        "experiment": "A4",
        "probe": "MoralCausalTracer",
        "elapsed_s": round(elapsed, 1),
        "peak_causal_layer": result.peak_causal_layer,
        "peak_mean_indirect_effect": round(result.peak_mean_indirect_effect, 4),
        "causal_depth": round(result.causal_depth, 4),
        "plot": str(png_path),
    }


def run_a5(model, dataset, output_dir: Path) -> dict:
    """A5: Fragility check — MoralFragilityTest noise robustness."""
    from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
    from deepsteer.viz import plot_fragility

    print("\n" + "=" * 60)
    print("A5: MoralFragilityTest — noise robustness curves")
    print("=" * 60)

    test = MoralFragilityTest(dataset=dataset)
    t0 = time.time()
    result = test.run(model)
    elapsed = time.time() - t0

    png_path = plot_fragility(result, output_dir=output_dir)

    print(f"  Time: {elapsed:.1f}s")
    print(f"  mean_critical_noise = {result.mean_critical_noise}")
    print(f"  most_fragile_layer = {result.most_fragile_layer}")
    print(f"  most_robust_layer = {result.most_robust_layer}")
    print(f"  Plot: {png_path}")

    return {
        "experiment": "A5",
        "probe": "MoralFragilityTest",
        "elapsed_s": round(elapsed, 1),
        "mean_critical_noise": result.mean_critical_noise,
        "most_fragile_layer": result.most_fragile_layer,
        "most_robust_layer": result.most_robust_layer,
        "plot": str(png_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A: Pipeline validation for Moral Emergence Curve study.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--weights", default=DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-dir", default="outputs/phase_a",
        help="Directory for output plots and JSON (default: outputs/phase_a).",
    )
    parser.add_argument(
        "--experiments", nargs="+", default=ALL_EXPERIMENTS,
        choices=ALL_EXPERIMENTS,
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--dataset-target", type=int, default=40,
        help="Target pairs per moral foundation (default: 40).",
    )
    parser.add_argument(
        "--trajectory-points", type=int, default=5,
        help="Number of checkpoints for A2 trajectory (default: 5).",
    )
    parser.add_argument(
        "--causal-prompts", type=int, default=40,
        help="Max prompts for A4 causal tracing (default: 40).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device to use (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--list-checkpoints", action="store_true",
        help="List available checkpoint revisions and exit.",
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

    # -- List checkpoints and exit --
    if args.list_checkpoints:
        from deepsteer.benchmarks.representational.trajectory import (
            list_available_revisions,
        )

        revisions = list_available_revisions(args.weights)
        print(f"Available revisions for {args.weights} ({len(revisions)} total):")
        for rev in revisions:
            print(f"  {rev}")
        return

    # -- Build probing dataset --
    from deepsteer.datasets.pipeline import build_probing_dataset

    print(f"Building probing dataset (target={args.dataset_target} per foundation)...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(
        f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs "
        f"({len(dataset.train) + len(dataset.test)} total)"
    )

    # -- Load model --
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    tier = AccessTier.CHECKPOINTS if "A2" in args.experiments else AccessTier.WEIGHTS
    print(f"\nLoading model: {args.weights}")
    model = WhiteBoxModel(args.weights, device=args.device, access_tier=tier)
    print(
        f"Model loaded: {model.info.n_layers} layers, "
        f"{model.info.n_params / 1e6:.0f}M params, "
        f"device={model._device}, dtype={model._dtype}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Run experiments --
    experiments = args.experiments
    summaries: list[dict] = []
    total_t0 = time.time()

    # A1, A3, A4, A5 all run on the loaded model (single checkpoint)
    if "A1" in experiments:
        summaries.append(run_a1(model, dataset, output_dir))
        _clear_memory()

    if "A3" in experiments:
        summaries.append(run_a3(model, dataset, output_dir))
        _clear_memory()

    if "A4" in experiments:
        summaries.append(run_a4(model, dataset, output_dir, max_prompts=args.causal_prompts))
        _clear_memory()

    if "A5" in experiments:
        summaries.append(run_a5(model, dataset, output_dir))
        _clear_memory()

    # A2 needs to load multiple checkpoints — run last
    if "A2" in experiments:
        summaries.append(
            run_a2(
                model,
                dataset,
                output_dir,
                n_trajectory_points=args.trajectory_points,
                device=args.device,
            )
        )
        _clear_memory()

    total_elapsed = time.time() - total_t0

    # -- Summary --
    print("\n" + "=" * 60)
    print("PHASE A SUMMARY")
    print("=" * 60)
    print(f"Model: {args.weights}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")
    print(f"Output: {output_dir}")
    print()

    for s in summaries:
        exp = s["experiment"]
        if s.get("skipped"):
            print(f"  {exp}: SKIPPED ({s.get('reason', 'unknown')})")
        else:
            elapsed = s.get("elapsed_s", "?")
            probe = s.get("probe", "?")
            print(f"  {exp}: {probe} — {elapsed}s")

    # Save summary JSON
    summary_path = output_dir / "phase_a_summary.json"
    summary_data = {
        "model": args.weights,
        "dataset_target": args.dataset_target,
        "total_elapsed_s": round(total_elapsed, 1),
        "experiments": summaries,
    }
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary JSON: {summary_path}")


if __name__ == "__main__":
    main()
