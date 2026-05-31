#!/usr/bin/env python3
"""B.2: Register transfer quantification.

Quantify how well declarative-register probe directions transfer to
narrative-register dilemma text. Compare probe-weight vs mean-difference
directions: does the training-free mean-diff method generalize better
across text registers?

For each foundation f, at each layer:
  - Same-register:  pair accuracy on declarative test pairs
  - Cross-register: pair accuracy on dilemma pairs containing f

Pair accuracy: fraction of pairs where dot(direction, moral_act) >
dot(direction, neutral_act). Threshold-free.

Usage:
    python papers/3_moral_geometry/scripts/probe_engineering/register_transfer.py
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from shared import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    DILEMMA_TO_PROBE,
    compute_mean_diff_directions,
    load_probe_directions,
    pair_accuracy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="B.2: Register transfer quantification.")
    parser.add_argument("--probe-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--dilemma-dataset",
                        default="deepsteer/datasets/dilemma_pairs_final.json")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/probe_engineering")
    parser.add_argument("--figures-dir",
                        default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier, MoralFoundation
    from deepsteer.datasets.pipeline import build_probing_dataset
    from deepsteer.datasets.types import ProbingPair, NeutralDomain, GenerationMethod
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

    print(f"{'='*60}")
    print("B.2: Register Transfer Quantification")
    print(f"{'='*60}")

    # ── Load declarative probing dataset ──
    dataset = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    print(f"Declarative dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    train_foundation_idx: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.train):
        train_foundation_idx[pair.foundation.value].append(i)

    test_foundation_idx: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(dataset.test):
        test_foundation_idx[pair.foundation.value].append(i)

    # ── Load dilemma dataset ──
    with open(args.dilemma_dataset) as f:
        dilemma_data = json.load(f)
    dilemma_pairs_raw = dilemma_data["pairs"]
    print(f"Dilemma dataset: {len(dilemma_pairs_raw)} pairs")

    # Convert to ProbingPair objects for activation collection.
    # Use the first foundation in the pair as the nominal label.
    dilemma_probing: list[ProbingPair] = []
    dilemma_foundation_idx: dict[str, list[int]] = defaultdict(list)

    for dp in dilemma_pairs_raw:
        idx = len(dilemma_probing)
        probe_pair = ProbingPair(
            moral=dp["moral"],
            neutral=dp["neutral"],
            foundation=MoralFoundation(DILEMMA_TO_PROBE[dp["foundation_pair"][0]]),
            neutral_domain=NeutralDomain.MATCHED,
            generation_method=GenerationMethod.HANDWRITTEN,
            moral_word_count=len(dp["moral"].split()),
            neutral_word_count=len(dp["neutral"].split()),
        )
        dilemma_probing.append(probe_pair)
        for f_short in dp["foundation_pair"]:
            dilemma_foundation_idx[DILEMMA_TO_PROBE[f_short]].append(idx)

    for fv in FOUNDATION_ORDER:
        print(f"  Dilemma pairs for {FOUNDATION_SHORT[fv]}: {len(dilemma_foundation_idx.get(fv, []))}")

    # ── Load model ──
    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    # ── Collect activations ──
    print("\nCollecting declarative train activations...")
    t0 = time.time()
    decl_train_acts = LayerWiseMoralProbe._collect_all_activations(model, dataset.train)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Collecting declarative test activations...")
    t0 = time.time()
    decl_test_acts = LayerWiseMoralProbe._collect_all_activations(model, dataset.test)
    print(f"  Done in {time.time() - t0:.1f}s")

    print("Collecting dilemma activations...")
    t0 = time.time()
    dilemma_acts = LayerWiseMoralProbe._collect_all_activations(model, dilemma_probing)
    print(f"  Done in {time.time() - t0:.1f}s")

    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # ── Compute directions ──
    print("\nComputing mean-difference directions from declarative train...")
    md_directions = compute_mean_diff_directions(decl_train_acts, n_layers, train_foundation_idx)
    pw_directions = load_probe_directions(args.probe_directions)

    # ── Evaluate transfer ──
    print("\n--- Register Transfer Results ---\n")
    print(f"{'Foundation':<14s}  {'Method':<10s}  {'Same-reg':>8s}  {'Cross-reg':>9s}  {'Gap':>6s}")
    print("-" * 55)

    results: dict[str, dict] = {}

    for fv in FOUNDATION_ORDER:
        fname = FOUNDATION_SHORT[fv]
        test_idx = test_foundation_idx.get(fv, [])
        dilemma_idx = dilemma_foundation_idx.get(fv, [])
        results[fv] = {"probe_weight": {}, "mean_diff": {}}

        for method_name, directions in [("probe_weight", pw_directions), ("mean_diff", md_directions)]:
            same_reg_layers = []
            cross_reg_layers = []
            for layer in range(n_layers):
                d = directions[fv].get(layer)
                if d is None:
                    continue
                X_test, _ = decl_test_acts[layer]
                X_dilemma, _ = dilemma_acts[layer]
                sr = pair_accuracy(d, X_test, test_idx)
                cr = pair_accuracy(d, X_dilemma, dilemma_idx)
                same_reg_layers.append(sr)
                cross_reg_layers.append(cr)
                results[fv][method_name][str(layer)] = {
                    "same_register": round(sr, 4),
                    "cross_register": round(cr, 4),
                    "gap": round(sr - cr, 4),
                }

            mean_sr = np.mean(same_reg_layers) if same_reg_layers else 0
            mean_cr = np.mean(cross_reg_layers) if cross_reg_layers else 0
            print(f"  {fname:<12s}  {method_name:<10s}  {mean_sr:>8.3f}  {mean_cr:>9.3f}  {mean_sr - mean_cr:>+6.3f}")

            results[fv][method_name]["mean_same_register"] = round(float(mean_sr), 4)
            results[fv][method_name]["mean_cross_register"] = round(float(mean_cr), 4)
            results[fv][method_name]["mean_gap"] = round(float(mean_sr - mean_cr), 4)

    # ── Summary statistics ──
    print("\n--- Summary ---")
    for method_name in ["probe_weight", "mean_diff"]:
        ind_gaps = [results[fv][method_name]["mean_gap"]
                    for fv in ["care_harm", "fairness_cheating", "liberty_oppression"]]
        bind_gaps = [results[fv][method_name]["mean_gap"]
                     for fv in ["loyalty_betrayal", "authority_subversion", "sanctity_degradation"]]
        ind_cr = [results[fv][method_name]["mean_cross_register"]
                  for fv in ["care_harm", "fairness_cheating", "liberty_oppression"]]
        bind_cr = [results[fv][method_name]["mean_cross_register"]
                   for fv in ["loyalty_betrayal", "authority_subversion", "sanctity_degradation"]]
        print(f"\n  {method_name}:")
        print(f"    Individualizing: mean cross-reg = {np.mean(ind_cr):.3f}, mean gap = {np.mean(ind_gaps):+.3f}")
        print(f"    Binding:         mean cross-reg = {np.mean(bind_cr):.3f}, mean gap = {np.mean(bind_gaps):+.3f}")

    # ── Save results ──
    out_path = output_dir / "register_transfer.json"
    with open(out_path, "w") as f:
        json.dump({"analysis": "register_transfer", "n_layers": n_layers, "results": results}, f, indent=2)
    print(f"\nResults saved: {out_path}")

    # ── Generate figure ──
    generate_figure(results, n_layers, figures_dir)


def generate_figure(results: dict, n_layers: int, figures_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = list(range(n_layers))

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)

    colors = {"same_register": "#1E88E5", "cross_register": "#E53935"}
    method_styles = {"probe_weight": ("-", "o", 2.0), "mean_diff": ("--", "s", 1.5)}

    for idx, fv in enumerate(FOUNDATION_ORDER):
        row, col = divmod(idx, 3)
        ax = axes[row, col]
        fname = FOUNDATION_SHORT[fv]

        for method_name, (ls, marker, lw) in method_styles.items():
            method_label = "Probe wt" if method_name == "probe_weight" else "Mean diff"
            sr_vals = [results[fv][method_name].get(str(l), {}).get("same_register", 0.5) for l in layers]
            cr_vals = [results[fv][method_name].get(str(l), {}).get("cross_register", 0.5) for l in layers]
            ax.plot(layers, sr_vals, linestyle=ls, marker=marker, color=colors["same_register"],
                    linewidth=lw, markersize=4, label=f"{method_label} (same)")
            ax.plot(layers, cr_vals, linestyle=ls, marker=marker, color=colors["cross_register"],
                    linewidth=lw, markersize=4, label=f"{method_label} (cross)")

        ax.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(fname, fontsize=12, fontweight="bold")
        ax.set_ylim(0.3, 1.05)
        ax.set_xticks(layers[::2])
        ax.grid(True, alpha=0.2)

        if row == 1:
            ax.set_xlabel("Layer", fontsize=10)
        if col == 0:
            ax.set_ylabel("Pair Accuracy", fontsize=10)

    axes[0, 2].legend(fontsize=7, loc="lower left")

    fig.suptitle("B.2: Register Transfer — Declarative → Narrative Dilemma\n"
                 "(blue = same register, red = cross register; solid = probe weight, dashed = mean diff)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    for ext in ("png", "pdf"):
        fig.savefig(figures_dir / f"fig_b2_register_transfer.{ext}",
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {figures_dir / 'fig_b2_register_transfer.png'}")


if __name__ == "__main__":
    main()
