#!/usr/bin/env python3
"""Phase B: Primary results — Moral Emergence Curve on OLMo-3 7B.

Produces paper-quality results testing hypotheses H1-H6 from RESEARCH_PLAN.md.
Efficient checkpoint iteration: loads each checkpoint once, runs all applicable
probes, frees memory before loading the next.

Experiments:
    B1: LayerWiseMoralProbe — full layer probing (final checkpoint) [H1]
    B2: CheckpointTrajectoryProbe — 20 evenly spaced checkpoints [H2, H3]
    B3: FoundationSpecificProbe — 8 checkpoints (early/mid/late) [H4]
    B4: MoralCausalTracer — 3 checkpoints (early/mid/late) [H5]
    B5: MoralFragilityTest — 5 checkpoints [H6]

Target model: allenai/OLMo-3-1025-7B (7B params, 32 layers)
Stage1 range: step 0 to step ~1,413,000 (5.93T tokens on Dolma 3)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory

Usage:
    # Run all experiments (will take several hours)
    python examples/phase_b.py

    # Quick test with reduced dataset
    python examples/phase_b.py --dataset-target 10 --trajectory-points 5

    # Run only B1 (single checkpoint, fast)
    python examples/phase_b.py --experiments B1

    # Resume from a specific checkpoint (skip already-completed ones)
    python examples/phase_b.py --resume-from stage1-step200000
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

REPO_ID = "allenai/OLMo-3-1025-7B"
ALL_EXPERIMENTS = ["B1", "B2", "B3", "B4", "B5"]


def _clear_memory() -> None:
    """Free GPU/MPS memory."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _cleanup_cache_revision(revision: str) -> None:
    """Delete a specific revision from HuggingFace cache to free disk space.

    Essential for 7B+ models where each revision is ~14.6 GB and disk space
    is limited. Called after each checkpoint is fully processed.
    """
    from huggingface_hub import scan_cache_dir

    try:
        cache = scan_cache_dir()
        for repo in cache.repos:
            if "OLMo-3-1025-7B" not in repo.repo_id:
                continue
            for rev in repo.revisions:
                if revision in rev.refs:
                    strategy = cache.delete_revisions(rev.commit_hash)
                    freed = strategy.expected_freed_size / 1e9
                    strategy.execute()
                    logger.info("Cleaned cache for %s (freed %.1f GB)", revision, freed)
                    return
    except Exception as e:
        logger.warning("Cache cleanup failed for %s: %s", revision, e)


def _reload_step_results(step: int, output_dir: Path) -> dict[str, dict]:
    """Reload previously saved probe results for a checkpoint step.

    Reconstructs the result dataclass objects from JSON files so they can
    be used in aggregate plot generation.
    """
    from deepsteer.core.types import (
        AccessTier,
        CausalLayerEffect,
        CausalPromptResult,
        CausalTracingResult,
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

    # Find JSON files by probe type (the filename includes model name)
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

        elif bname == "moral_causal_tracer":
            result = CausalTracingResult(
                benchmark_name=bname,
                model_info=model_info,
                mean_indirect_effect_by_layer={
                    int(k): v
                    for k, v in data.get("mean_indirect_effect_by_layer", {}).items()
                },
                peak_causal_layer=data.get("peak_causal_layer"),
                peak_mean_indirect_effect=data.get("peak_mean_indirect_effect"),
                causal_depth=data.get("causal_depth"),
            )
            results["causal_tracing"] = {"result": result, "elapsed_s": 0}

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


def _parse_step(revision: str) -> int | None:
    """Extract step number from a revision string like 'stage1-step12000'."""
    match = re.search(r"step(\d+)", revision)
    return int(match.group(1)) if match else None


def _get_stage1_revisions(all_revisions: list[str]) -> list[tuple[int, str]]:
    """Filter to stage1 revisions and return as (step, revision) sorted by step."""
    stage1: list[tuple[int, str]] = []
    for rev in all_revisions:
        if not rev.startswith("stage1-"):
            continue
        step = _parse_step(rev)
        if step is not None:
            stage1.append((step, rev))
    stage1.sort(key=lambda x: x[0])
    return stage1


def _select_evenly_spaced(
    revisions: list[tuple[int, str]], n: int
) -> list[tuple[int, str]]:
    """Select n evenly spaced entries including first and last."""
    if len(revisions) <= n:
        return list(revisions)
    if n == 1:
        return [revisions[-1]]
    indices = [round(i * (len(revisions) - 1) / (n - 1)) for i in range(n)]
    return [revisions[i] for i in indices]


def _select_early_mid_late(
    revisions: list[tuple[int, str]], n: int
) -> list[tuple[int, str]]:
    """Select n checkpoints biased toward early/mid/late regions.

    Allocates roughly equal splits to early (first 20%), mid (20-60%),
    and late (60-100%) ranges of training.
    """
    if len(revisions) <= n:
        return list(revisions)

    total = len(revisions)
    early_end = int(total * 0.2)
    mid_end = int(total * 0.6)

    # Allocate: ceil(n/3) early, ceil(n/3) mid, rest late
    n_early = max(1, (n + 2) // 3)
    n_mid = max(1, (n + 1) // 3)
    n_late = n - n_early - n_mid

    early = _select_evenly_spaced(revisions[:early_end], n_early) if early_end > 0 else []
    mid = _select_evenly_spaced(revisions[early_end:mid_end], n_mid) if mid_end > early_end else []
    late = _select_evenly_spaced(revisions[mid_end:], n_late) if total > mid_end else []

    selected = early + mid + late
    # Deduplicate while preserving order
    seen = set()
    result = []
    for item in selected:
        if item[0] not in seen:
            seen.add(item[0])
            result.append(item)
    return result


def plan_checkpoints(
    stage1_revisions: list[tuple[int, str]],
    experiments: list[str],
    n_trajectory: int = 20,
) -> dict[str, set[int]]:
    """Plan which probes to run at each checkpoint.

    Returns mapping from experiment ID to set of step numbers.
    """
    plan: dict[str, set[int]] = {}

    # B1: final checkpoint only
    if "B1" in experiments:
        final_step = stage1_revisions[-1][0]
        plan["B1"] = {final_step}

    # B2: n_trajectory evenly spaced
    if "B2" in experiments:
        b2_selected = _select_evenly_spaced(stage1_revisions, n_trajectory)
        plan["B2"] = {s for s, _ in b2_selected}

    # B3: 8 checkpoints evenly spaced
    if "B3" in experiments:
        b3_selected = _select_evenly_spaced(stage1_revisions, 8)
        plan["B3"] = {s for s, _ in b3_selected}

    # B4: 3 checkpoints (early/mid/late)
    if "B4" in experiments:
        b4_selected = _select_evenly_spaced(stage1_revisions, 3)
        plan["B4"] = {s for s, _ in b4_selected}

    # B5: 5 evenly spaced
    if "B5" in experiments:
        b5_selected = _select_evenly_spaced(stage1_revisions, 5)
        plan["B5"] = {s for s, _ in b5_selected}

    return plan


def run_probes_on_checkpoint(
    model,
    dataset,
    step: int,
    probes_to_run: set[str],
    output_dir: Path,
    *,
    causal_prompts: int = 40,
) -> dict[str, dict]:
    """Run all applicable probes on a single loaded model checkpoint."""
    results: dict[str, dict] = {}
    step_dir = output_dir / f"step_{step:07d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    if "B2" in probes_to_run or "B1" in probes_to_run:
        from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
        from deepsteer.viz import plot_layer_probing

        probe = LayerWiseMoralProbe(dataset=dataset)
        t0 = time.time()
        result = probe.run(model)
        elapsed = time.time() - t0

        png_path = plot_layer_probing(result, output_dir=step_dir)
        logger.info(
            "  LayerWiseMoralProbe: %.1fs, peak=%.1f%% @ layer %d",
            elapsed, result.peak_accuracy * 100, result.peak_layer,
        )
        results["layer_probing"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    if "B3" in probes_to_run:
        from deepsteer.benchmarks.representational.foundation_probes import (
            FoundationSpecificProbe,
        )
        from deepsteer.viz import plot_foundation_probes

        probe = FoundationSpecificProbe(dataset=dataset)
        t0 = time.time()
        result = probe.run(model)
        elapsed = time.time() - t0

        png_path = plot_foundation_probes(result, output_dir=step_dir)
        logger.info("  FoundationSpecificProbe: %.1fs", elapsed)
        results["foundation_probes"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    if "B4" in probes_to_run:
        from deepsteer.benchmarks.representational.causal_tracing import (
            MoralCausalTracer,
        )
        from deepsteer.viz import plot_causal_tracing

        tracer = MoralCausalTracer(dataset=dataset, max_prompts=causal_prompts)
        t0 = time.time()
        result = tracer.run(model)
        elapsed = time.time() - t0

        png_path = plot_causal_tracing(result, output_dir=step_dir)
        logger.info(
            "  MoralCausalTracer: %.1fs, peak_causal_layer=%d",
            elapsed, result.peak_causal_layer,
        )
        results["causal_tracing"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    if "B5" in probes_to_run:
        from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
        from deepsteer.viz import plot_fragility

        test = MoralFragilityTest(dataset=dataset)
        t0 = time.time()
        result = test.run(model)
        elapsed = time.time() - t0

        png_path = plot_fragility(result, output_dir=step_dir)
        logger.info(
            "  MoralFragilityTest: %.1fs, mean_critical=%.2f",
            elapsed, result.mean_critical_noise or 0.0,
        )
        results["fragility"] = {
            "result": result,
            "elapsed_s": round(elapsed, 1),
        }

    return results


def generate_aggregate_plots(
    all_results: dict[int, dict],
    plan: dict[str, set[int]],
    output_dir: Path,
) -> list[str]:
    """Generate cross-checkpoint aggregate plots for B2-B5."""
    from deepsteer.core.types import (
        CheckpointTrajectoryResult,
        LayerProbingResult,
    )
    from deepsteer.viz import plot_checkpoint_trajectory

    plot_paths: list[str] = []

    # B2: Checkpoint trajectory heatmap
    if "B2" in plan:
        b2_steps = sorted(plan["B2"])
        trajectory: list[LayerProbingResult] = []
        steps_used: list[int] = []
        for step in b2_steps:
            if step in all_results and "layer_probing" in all_results[step]:
                result = all_results[step]["layer_probing"]["result"]
                trajectory.append(result)
                steps_used.append(step)

        if trajectory:
            # Get model_info from the first result
            traj_result = CheckpointTrajectoryResult(
                benchmark_name="checkpoint_trajectory_probe",
                model_info=trajectory[0].model_info,
                trajectory=trajectory,
                checkpoint_steps=steps_used,
                metadata={"n_checkpoints": len(steps_used)},
            )
            png = plot_checkpoint_trajectory(traj_result, output_dir=output_dir)
            plot_paths.append(str(png))
            logger.info("B2 trajectory heatmap: %s", png)

    # B3: Foundation emergence across checkpoints
    if "B3" in plan:
        _plot_foundation_emergence(all_results, plan["B3"], output_dir)

    # B5: Fragility evolution across checkpoints
    if "B5" in plan:
        _plot_fragility_evolution(all_results, plan["B5"], output_dir)

    return plot_paths


def _plot_foundation_emergence(
    all_results: dict[int, dict],
    b3_steps: set[int],
    output_dir: Path,
) -> None:
    """Plot Figure 3: Foundation onset layer vs training step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from deepsteer.core.types import MoralFoundation

    sorted_steps = sorted(b3_steps)
    foundation_onsets: dict[str, list[tuple[int, int | None]]] = {
        f.value: [] for f in MoralFoundation
    }

    for step in sorted_steps:
        if step not in all_results or "foundation_probes" not in all_results[step]:
            continue
        result = all_results[step]["foundation_probes"]["result"]
        for fname, summary in result.per_foundation_summary.items():
            foundation_onsets[fname].append((step, summary.get("onset_layer")))

    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#795548"]
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (fname, data) in enumerate(foundation_onsets.items()):
        if not data:
            continue
        steps = [s for s, o in data if o is not None]
        onsets = [o for _, o in data if o is not None]
        if steps:
            label = fname.replace("_", "/")
            ax.plot(steps, onsets, "o-", color=colors[i % len(colors)],
                    linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Onset Layer")
    ax.set_title("Foundation Emergence Staggering — OLMo-3 7B")
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower onset = deeper encoding
    fig.tight_layout()

    png_path = output_dir / "b3_foundation_emergence.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("B3 foundation emergence plot: %s", png_path)


def _plot_fragility_evolution(
    all_results: dict[int, dict],
    b5_steps: set[int],
    output_dir: Path,
) -> None:
    """Plot Figure 5: Critical noise vs training step by layer group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sorted_steps = sorted(b5_steps)
    # Group layers into early/mid/late thirds
    groups = {"early (0-10)": range(0, 11), "mid (11-21)": range(11, 22), "late (22-31)": range(22, 32)}
    group_critical: dict[str, list[tuple[int, float]]] = {g: [] for g in groups}

    for step in sorted_steps:
        if step not in all_results or "fragility" not in all_results[step]:
            continue
        result = all_results[step]["fragility"]["result"]
        for group_name, layer_range in groups.items():
            criticals = []
            for ls in result.layer_scores:
                if ls.layer in layer_range and ls.critical_noise is not None:
                    criticals.append(ls.critical_noise)
            if criticals:
                group_critical[group_name].append((step, float(np.mean(criticals))))

    colors = {"early (0-10)": "#F44336", "mid (11-21)": "#FF9800", "late (22-31)": "#4CAF50"}
    fig, ax = plt.subplots(figsize=(10, 5))

    for group_name, data in group_critical.items():
        if not data:
            continue
        steps = [s for s, _ in data]
        criticals = [c for _, c in data]
        ax.plot(steps, criticals, "o-", color=colors[group_name],
                linewidth=2, markersize=5, label=group_name)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Critical Noise (higher = more robust)")
    ax.set_title("Moral Encoding Robustness Over Training — OLMo-3 7B")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    png_path = output_dir / "b5_fragility_evolution.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("B5 fragility evolution plot: %s", png_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase B: Paper-quality results on OLMo-3 7B.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default="outputs/phase_b",
        help="Directory for output plots and JSON (default: outputs/phase_b).",
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
        "--trajectory-points", type=int, default=20,
        help="Number of checkpoints for B2 trajectory (default: 20).",
    )
    parser.add_argument(
        "--causal-prompts", type=int, default=40,
        help="Max prompts for B4 causal tracing (default: 40).",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device (cuda, mps, cpu). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="Skip checkpoints before this revision (for resuming interrupted runs).",
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

    # -- List and plan checkpoints --
    from deepsteer.benchmarks.representational.trajectory import list_available_revisions

    print(f"\nListing revisions for {REPO_ID}...")
    all_revisions = list_available_revisions(REPO_ID)
    stage1 = _get_stage1_revisions(all_revisions)
    print(f"Found {len(stage1)} stage1 checkpoints (step {stage1[0][0]} to {stage1[-1][0]})")

    plan = plan_checkpoints(stage1, args.experiments, n_trajectory=args.trajectory_points)

    # Compute superset: all steps that need probing + which probes at each
    step_probes: dict[int, set[str]] = {}
    for exp, steps in plan.items():
        for step in steps:
            if step not in step_probes:
                step_probes[step] = set()
            step_probes[step].add(exp)

    # Map step → revision
    step_to_rev = {s: r for s, r in stage1}
    sorted_steps = sorted(step_probes.keys())

    # Handle --resume-from
    resume_step = None
    if args.resume_from:
        resume_step = _parse_step(args.resume_from)
        if resume_step is not None:
            sorted_steps = [s for s in sorted_steps if s >= resume_step]
            print(f"Resuming from step {resume_step}: {len(sorted_steps)} checkpoints remaining")

    print(f"\nPhase B plan:")
    for exp in sorted(plan.keys()):
        steps = sorted(plan[exp])
        print(f"  {exp}: {len(steps)} checkpoints — steps {steps}")
    print(f"\nTotal unique checkpoints to process: {len(sorted_steps)}")

    # Estimate time
    probes_per_step = {s: len(step_probes[s]) for s in sorted_steps}
    # Rough estimate: ~2 min per LayerWiseMoralProbe, ~3 min for Foundation, ~5 min for Causal, ~3 min for Fragility
    est_minutes = 0
    for s in sorted_steps:
        probes = step_probes[s]
        if "B2" in probes or "B1" in probes:
            est_minutes += 2
        if "B3" in probes:
            est_minutes += 3
        if "B4" in probes:
            est_minutes += 5
        if "B5" in probes:
            est_minutes += 3
    est_minutes += len(sorted_steps) * 3  # ~3 min model download/load per checkpoint
    print(f"Estimated time: ~{est_minutes} minutes ({est_minutes / 60:.1f} hours)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save plan
    plan_data = {
        "repo_id": REPO_ID,
        "experiments": {k: sorted(v) for k, v in plan.items()},
        "total_checkpoints": len(sorted_steps),
        "dataset_target": args.dataset_target,
    }
    with open(output_dir / "phase_b_plan.json", "w") as f:
        json.dump(plan_data, f, indent=2)

    # -- Iterate over checkpoints --
    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier

    all_results: dict[int, dict] = {}

    # Reload results from previously completed steps (for aggregate plots)
    if resume_step is not None:
        all_plan_steps = sorted(step_probes.keys())
        skipped_steps = [s for s in all_plan_steps if s < resume_step]
        for s in skipped_steps:
            step_json = output_dir / f"step_{s:07d}" / "step_summary.json"
            if step_json.exists():
                all_results[s] = _reload_step_results(s, output_dir)
                logger.info("Reloaded results from step %d", s)
        print(f"Reloaded {len(all_results)} previously completed checkpoints")

    total_t0 = time.time()

    for i, step in enumerate(sorted_steps):
        revision = step_to_rev[step]
        probes = step_probes[step]

        print(f"\n{'='*60}")
        print(f"Checkpoint {i+1}/{len(sorted_steps)}: {revision} (step {step})")
        print(f"  Probes: {', '.join(sorted(probes))}")
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
            model, dataset, step, probes, output_dir,
            causal_prompts=args.causal_prompts,
        )
        all_results[step] = step_results

        # Save incremental results for this step
        step_summary = {
            "step": step,
            "revision": revision,
            "probes_run": list(probes),
        }
        for probe_name, probe_data in step_results.items():
            step_summary[probe_name] = {
                "elapsed_s": probe_data["elapsed_s"],
            }
            # Add key metrics
            r = probe_data["result"]
            if hasattr(r, "peak_accuracy"):
                step_summary[probe_name]["peak_accuracy"] = round(r.peak_accuracy, 4)
                step_summary[probe_name]["onset_layer"] = r.onset_layer
                step_summary[probe_name]["moral_encoding_depth"] = round(r.moral_encoding_depth, 4)
                step_summary[probe_name]["moral_encoding_breadth"] = round(r.moral_encoding_breadth, 4)
            if hasattr(r, "peak_causal_layer"):
                step_summary[probe_name]["peak_causal_layer"] = r.peak_causal_layer
                step_summary[probe_name]["causal_depth"] = round(r.causal_depth, 4)
            if hasattr(r, "mean_critical_noise"):
                step_summary[probe_name]["mean_critical_noise"] = r.mean_critical_noise

        step_json = output_dir / f"step_{step:07d}" / "step_summary.json"
        with open(step_json, "w") as f:
            json.dump(step_summary, f, indent=2)

        # Free memory and clean cache to avoid disk space exhaustion
        del model
        _clear_memory()
        _cleanup_cache_revision(revision)

        elapsed_so_far = time.time() - total_t0
        remaining = len(sorted_steps) - (i + 1)
        avg_per_step = elapsed_so_far / (i + 1)
        eta = avg_per_step * remaining
        print(f"  Elapsed: {elapsed_so_far/60:.1f}min, ETA: {eta/60:.1f}min ({remaining} checkpoints left)")

    total_elapsed = time.time() - total_t0

    # -- Generate aggregate plots --
    print(f"\n{'='*60}")
    print("Generating aggregate plots...")
    print(f"{'='*60}")
    generate_aggregate_plots(all_results, plan, output_dir)

    # -- Summary --
    print(f"\n{'='*60}")
    print("PHASE B SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {REPO_ID}")
    print(f"Total time: {total_elapsed/60:.1f} min ({total_elapsed/3600:.1f} hours)")
    print(f"Checkpoints processed: {len(sorted_steps)}")
    print(f"Output: {output_dir}")

    # B1 summary (final checkpoint)
    if "B1" in plan:
        final = max(plan["B1"])
        if final in all_results and "layer_probing" in all_results[final]:
            r = all_results[final]["layer_probing"]["result"]
            print(f"\nB1 (H1 — Emergent Moral Decodability):")
            print(f"  peak_accuracy = {r.peak_accuracy:.1%}")
            print(f"  onset_layer = {r.onset_layer}, peak_layer = {r.peak_layer}")
            print(f"  depth = {r.moral_encoding_depth:.3f}, breadth = {r.moral_encoding_breadth:.3f}")

    # B2 summary (trajectory)
    if "B2" in plan:
        b2_steps = sorted(plan["B2"])
        depths = []
        breadths = []
        for s in b2_steps:
            if s in all_results and "layer_probing" in all_results[s]:
                r = all_results[s]["layer_probing"]["result"]
                depths.append(r.moral_encoding_depth)
                breadths.append(r.moral_encoding_breadth)
        if depths:
            print(f"\nB2 (H2/H3 — Deepening & Broadening):")
            print(f"  depth: {depths[0]:.3f} → {depths[-1]:.3f} (expected: decreasing)")
            print(f"  breadth: {breadths[0]:.3f} → {breadths[-1]:.3f} (expected: increasing)")

    # B4 summary (causal)
    if "B4" in plan:
        b4_steps = sorted(plan["B4"])
        print(f"\nB4 (H5 — Causal-Probing Alignment):")
        for s in b4_steps:
            if s in all_results and "causal_tracing" in all_results[s]:
                r = all_results[s]["causal_tracing"]["result"]
                print(f"  step {s}: peak_causal_layer={r.peak_causal_layer}, depth={r.causal_depth:.3f}")

    # Save full summary
    summary_data = {
        "repo_id": REPO_ID,
        "total_elapsed_s": round(total_elapsed, 1),
        "checkpoints_processed": len(sorted_steps),
        "plan": {k: sorted(v) for k, v in plan.items()},
    }
    summary_path = output_dir / "phase_b_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
