#!/usr/bin/env python3
"""Experiment 5: Dense vs. MoE controlled comparison.

Runs the standard moral probe + fragility battery on OLMoE-1B-7B (final
checkpoint) and OLMo-2 1B (final checkpoint), then generates side-by-side
comparison figures.

Target models:
    - OLMoE-1B-7B: allenai/OLMoE-1B-7B-0924 (6.9B total, 1.3B active, 16 layers)
    - OLMo-2 1B:   allenai/OLMo-2-0425-1B    (1.5B params, 16 layers)

Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS, bf16
Estimated runtime: ~30 min total (~15 min per model)

Outputs (in papers/persona_monitoring/outputs/exp5_dense_vs_moe/):
    - Per-model probe + fragility JSON and PNG
    - exp5_comparison.png: side-by-side layer profiles
    - exp5_summary.json: headline comparison numbers

Usage:
    python papers/persona_monitoring/scripts/exp5_dense_vs_moe.py
    python papers/persona_monitoring/scripts/exp5_dense_vs_moe.py --olmoe-only
    python papers/persona_monitoring/scripts/exp5_dense_vs_moe.py --olmo-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch
import numpy as np

logger = logging.getLogger(__name__)

# MPS doesn't implement torch.histc for integer inputs, which the
# transformers MoE router uses (grouped_mm_experts_forward). Patch it
# to fall back to CPU for that one op.
_orig_histc = torch.histc

def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)

torch.histc = _histc_mps_fallback

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"
OLMO_REPO = "allenai/OLMo-2-0425-1B"


def _clear_memory() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_probes(model, dataset, output_dir: Path, label: str) -> dict:
    """Run LayerWiseMoralProbe + MoralFragilityTest, save results."""
    from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.viz import plot_fragility, plot_layer_probing

    results = {}

    print(f"\n  [{label}] Running LayerWiseMoralProbe...")
    probe = LayerWiseMoralProbe(dataset=dataset)
    t0 = time.time()
    probe_result = probe.run(model)
    elapsed = time.time() - t0
    plot_layer_probing(probe_result, output_dir=output_dir)

    probe_json = {
        "benchmark_name": probe_result.benchmark_name,
        "model_info": asdict(probe_result.model_info),
        "layer_scores": [asdict(s) for s in probe_result.layer_scores],
        "onset_layer": probe_result.onset_layer,
        "peak_layer": probe_result.peak_layer,
        "peak_accuracy": probe_result.peak_accuracy,
        "moral_encoding_depth": probe_result.moral_encoding_depth,
        "moral_encoding_breadth": probe_result.moral_encoding_breadth,
    }
    with open(output_dir / f"probe_{label}.json", "w") as f:
        json.dump(probe_json, f, indent=2)

    print(f"  [{label}] LayerWiseMoralProbe: {elapsed:.1f}s, "
          f"peak={probe_result.peak_accuracy:.1%} @ layer {probe_result.peak_layer}")
    results["probe"] = probe_result

    print(f"  [{label}] Running MoralFragilityTest...")
    frag = MoralFragilityTest(dataset=dataset)
    t0 = time.time()
    frag_result = frag.run(model)
    elapsed = time.time() - t0
    plot_fragility(frag_result, output_dir=output_dir)

    frag_json = {
        "benchmark_name": frag_result.benchmark_name,
        "model_info": asdict(frag_result.model_info),
        "layer_scores": [
            {
                "layer": s.layer,
                "baseline_accuracy": s.baseline_accuracy,
                "accuracy_by_noise": {str(k): v for k, v in s.accuracy_by_noise.items()},
                "critical_noise": s.critical_noise,
            }
            for s in frag_result.layer_scores
        ],
        "noise_levels": frag_result.noise_levels,
        "mean_critical_noise": frag_result.mean_critical_noise,
        "most_fragile_layer": frag_result.most_fragile_layer,
        "most_robust_layer": frag_result.most_robust_layer,
    }
    with open(output_dir / f"fragility_{label}.json", "w") as f:
        json.dump(frag_json, f, indent=2)

    print(f"  [{label}] MoralFragilityTest: {elapsed:.1f}s, "
          f"mean_critical={frag_result.mean_critical_noise:.2f}")
    results["fragility"] = frag_result

    return results


def generate_comparison_plot(
    olmoe_results: dict,
    olmo_results: dict,
    output_dir: Path,
) -> None:
    """Generate side-by-side comparison figure (Figure 5 in Paper 2)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # -- Panel 1: Probe accuracy by layer --
    ax = axes[0]
    for label, results, color, marker in [
        ("OLMo-2 1B (dense)", olmo_results, "#2196F3", "o"),
        ("OLMoE-1B-7B (MoE)", olmoe_results, "#F44336", "s"),
    ]:
        probe = results["probe"]
        layers = [s.layer for s in probe.layer_scores]
        accs = [s.accuracy for s in probe.layer_scores]
        ax.plot(layers, accs, f"{marker}-", color=color, linewidth=2,
                markersize=6, label=label)

    ax.axhline(y=0.6, color="#9E9E9E", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Probing Accuracy", fontsize=12)
    ax.set_title("Moral Probing Accuracy by Layer", fontsize=13)
    ax.set_ylim(0.45, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # -- Panel 2: Fragility (critical noise) by layer --
    ax = axes[1]
    for label, results, color, marker in [
        ("OLMo-2 1B (dense)", olmo_results, "#2196F3", "o"),
        ("OLMoE-1B-7B (MoE)", olmoe_results, "#F44336", "s"),
    ]:
        frag = results["fragility"]
        layers = [s.layer for s in frag.layer_scores]
        criticals = [s.critical_noise if s.critical_noise is not None else 10.0
                     for s in frag.layer_scores]
        ax.plot(layers, criticals, f"{marker}-", color=color, linewidth=2,
                markersize=6, label=label)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Critical Noise (higher = more robust)", fontsize=12)
    ax.set_title("Moral Encoding Fragility by Layer", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Dense vs. MoE: Moral Encoding Comparison\n"
        "(Same probing dataset, same probes, different architecture)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    png_path = output_dir / "exp5_comparison.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison figure: {png_path}")


def generate_summary(
    olmoe_results: dict,
    olmo_results: dict,
    output_dir: Path,
) -> None:
    """Generate summary JSON with headline comparison numbers."""
    def _extract(results: dict) -> dict:
        probe = results["probe"]
        frag = results["fragility"]
        return {
            "probe": {
                "model": probe.model_info.name,
                "n_layers": probe.model_info.n_layers,
                "n_params": probe.model_info.n_params,
                "onset_layer": probe.onset_layer,
                "peak_layer": probe.peak_layer,
                "peak_accuracy": round(probe.peak_accuracy, 4),
                "moral_encoding_depth": round(probe.moral_encoding_depth, 4),
                "moral_encoding_breadth": round(probe.moral_encoding_breadth, 4),
                "per_layer_accuracy": [
                    round(s.accuracy, 4) for s in probe.layer_scores
                ],
            },
            "fragility": {
                "mean_critical_noise": round(frag.mean_critical_noise, 4)
                if frag.mean_critical_noise else None,
                "most_fragile_layer": frag.most_fragile_layer,
                "most_robust_layer": frag.most_robust_layer,
                "per_layer_critical_noise": [
                    s.critical_noise for s in frag.layer_scores
                ],
            },
        }

    summary = {
        "experiment": "exp5_dense_vs_moe",
        "description": "Dense vs. MoE controlled comparison on moral probing + fragility",
        "olmoe": _extract(olmoe_results),
        "olmo": _extract(olmo_results),
    }

    path = output_dir / "exp5_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5: Dense vs. MoE comparison.",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/persona_monitoring/outputs/exp5_dense_vs_moe",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument(
        "--olmoe-only", action="store_true",
        help="Run OLMoE only (skip OLMo-2 1B).",
    )
    parser.add_argument(
        "--olmo-only", action="store_true",
        help="Run OLMo-2 1B only (skip OLMoE).",
    )
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

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    olmoe_results = None
    olmo_results = None

    if not args.olmo_only:
        print(f"\n{'='*60}")
        print(f"Loading OLMoE-1B-7B: {OLMOE_REPO}")
        print(f"{'='*60}")
        t0 = time.time()
        olmoe_model = WhiteBoxModel(
            OLMOE_REPO,
            device=args.device,
            access_tier=AccessTier.WEIGHTS,
        )
        print(f"Loaded in {time.time() - t0:.1f}s "
              f"({olmoe_model.info.n_params / 1e9:.1f}B params, "
              f"{olmoe_model.info.n_layers} layers)")

        olmoe_dir = output_dir / "olmoe"
        olmoe_dir.mkdir(exist_ok=True)
        olmoe_results = run_probes(olmoe_model, dataset, olmoe_dir, "OLMoE")

        del olmoe_model
        _clear_memory()

    if not args.olmoe_only:
        print(f"\n{'='*60}")
        print(f"Loading OLMo-2 1B: {OLMO_REPO}")
        print(f"{'='*60}")
        t0 = time.time()
        olmo_model = WhiteBoxModel(
            OLMO_REPO,
            device=args.device,
            access_tier=AccessTier.WEIGHTS,
        )
        print(f"Loaded in {time.time() - t0:.1f}s "
              f"({olmo_model.info.n_params / 1e9:.1f}B params, "
              f"{olmo_model.info.n_layers} layers)")

        olmo_dir = output_dir / "olmo"
        olmo_dir.mkdir(exist_ok=True)
        olmo_results = run_probes(olmo_model, dataset, olmo_dir, "OLMo")

        del olmo_model
        _clear_memory()

    if olmoe_results and olmo_results:
        print(f"\n{'='*60}")
        print("Generating comparison plots...")
        print(f"{'='*60}")
        generate_comparison_plot(olmoe_results, olmo_results, output_dir)
        generate_summary(olmoe_results, olmo_results, output_dir)

        # Print headline comparison
        op = olmoe_results["probe"]
        mp = olmo_results["probe"]
        of = olmoe_results["fragility"]
        mf = olmo_results["fragility"]
        print(f"\n{'='*60}")
        print("HEADLINE COMPARISON")
        print(f"{'='*60}")
        print(f"{'':20s} {'OLMoE (MoE)':>15s} {'OLMo-2 (dense)':>15s}")
        print(f"{'Params':20s} {op.model_info.n_params/1e9:>14.1f}B {mp.model_info.n_params/1e9:>14.1f}B")
        print(f"{'Layers':20s} {op.model_info.n_layers:>15d} {mp.model_info.n_layers:>15d}")
        print(f"{'Peak accuracy':20s} {op.peak_accuracy:>14.1%} {mp.peak_accuracy:>14.1%}")
        print(f"{'Peak layer':20s} {op.peak_layer:>15d} {mp.peak_layer:>15d}")
        print(f"{'Onset layer':20s} {op.onset_layer:>15} {mp.onset_layer:>15}")
        print(f"{'Encoding depth':20s} {op.moral_encoding_depth:>15.3f} {mp.moral_encoding_depth:>15.3f}")
        print(f"{'Encoding breadth':20s} {op.moral_encoding_breadth:>15.3f} {mp.moral_encoding_breadth:>15.3f}")
        print(f"{'Mean crit. noise':20s} {of.mean_critical_noise:>15.2f} {mf.mean_critical_noise:>15.2f}")
        print(f"{'Most fragile layer':20s} {of.most_fragile_layer:>15} {mf.most_fragile_layer:>15}")
        print(f"{'Most robust layer':20s} {of.most_robust_layer:>15} {mf.most_robust_layer:>15}")
    elif olmoe_results or olmo_results:
        label = "OLMoE" if olmoe_results else "OLMo"
        r = olmoe_results or olmo_results
        print(f"\n{label} results saved. Run without --*-only flags for comparison.")

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
