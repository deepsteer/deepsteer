#!/usr/bin/env python3
"""Experiment 7: Framework-specific fragility.

Tests whether different moral foundations have different robustness profiles
by applying the standard fragility protocol (Gaussian noise injection) to
each foundation-specific probe separately.

Hypothesis: more universal foundations (care/harm) are more robustly encoded
than more culturally variable foundations (sanctity/degradation, loyalty/betrayal).

Cross-architecture: runs on both OLMo-2 1B and OLMoE-1B-7B.

Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS
Estimated runtime: ~30 min (6 foundations × standard fragility battery × 2 models)

Usage:
    python papers/3_moral_geometry/scripts/exp7_framework_fragility.py
    python papers/3_moral_geometry/scripts/exp7_framework_fragility.py --olmo-only
    python papers/3_moral_geometry/scripts/exp7_framework_fragility.py --olmoe-only
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OLMO_REPO = "allenai/OLMo-2-0425-1B"
OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"


def _model_label(model_id: str) -> str:
    """Short human-readable label for an OLMo model id (e.g. 'OLMo-2 7B')."""
    name = model_id.rstrip("/").split("/")[-1]
    if "OLMo-2" in name:
        scale = name.split("-")[-1]  # e.g. '1B' or '7B'
        return f"OLMo-2 {scale}"
    return name


NOISE_LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0]
# Extended grid lifts the σ=10 censoring that is heavy at 7B (~28% of cells).
EXTENDED_NOISE_LEVELS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
FRAGILITY_THRESHOLD = 0.6
N_NOISE_SEEDS = 10
PROBE_SEED = 42
N_BOOTSTRAP = 1000

# MPS histc fix
_orig_histc = torch.histc


def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)


torch.histc = _histc_mps_fallback

from exp1_2_3_framework_geometry import (
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING,
)


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_foundation_fragility(
    model,
    dataset,
    output_dir: Path,
    label: str,
    *,
    noise_levels: list[float] | None = None,
    n_epochs: int = 50,
    lr: float = 1e-2,
    fragility_threshold: float = FRAGILITY_THRESHOLD,
    n_noise_seeds: int = N_NOISE_SEEDS,
    seed: int = PROBE_SEED,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict:
    """Run per-foundation fragility analysis.

    For each foundation:
    1. Collect activations for foundation-specific train/test pairs
    2. Train a linear probe on clean activations (fixed probe seed)
    3. Evaluate under increasing Gaussian noise, averaging accuracy over
       ``n_noise_seeds`` independent noise draws per (layer, σ)
    4. Find the critical noise level per layer from the seed-mean curve,
       censoring never-fragile layers at max(noise_levels) (cap-at-max)
    5. Report a per-foundation bootstrap CI over the noise seeds

    Matches the shared ``MoralFragilityTest`` conventions (Phase 0).
    """
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
    from deepsteer.core.types import MoralFoundation
    from deepsteer.datasets.types import ProbingPair

    if noise_levels is None:
        noise_levels = list(EXTENDED_NOISE_LEVELS)
    noise_cap = max(noise_levels)

    def _sigma_star(acc_by_noise: dict[float, float]) -> float:
        """First σ where accuracy < threshold, else cap (never-fragile)."""
        for nl in sorted(noise_levels):
            if acc_by_noise[nl] < fragility_threshold:
                return nl
        return noise_cap

    n_layers = model.info.n_layers
    assert n_layers is not None

    train_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    test_by_foundation: dict[str, list[ProbingPair]] = defaultdict(list)
    for pair in dataset.train:
        train_by_foundation[pair.foundation.value].append(pair)
    for pair in dataset.test:
        test_by_foundation[pair.foundation.value].append(pair)

    results: dict[str, dict] = {}

    for foundation_val in FOUNDATION_ORDER:
        train_pairs = train_by_foundation.get(foundation_val, [])
        test_pairs = test_by_foundation.get(foundation_val, [])

        if len(train_pairs) < 5 or len(test_pairs) < 1:
            continue

        print(f"  [{label}] Fragility for {FOUNDATION_SHORT[foundation_val]}...")

        all_train = LayerWiseMoralProbe._collect_all_activations(model, train_pairs)
        all_test = LayerWiseMoralProbe._collect_all_activations(model, test_pairs)

        per_layer: dict[int, dict] = {}
        # Per-seed accuracy curves per layer, for a bootstrap CI that is
        # consistent with the point estimate (resample seeds, recompute the
        # seed-mean curve, derive σ* from it — not by averaging per-seed σ*).
        layer_seed_acc: list[list[dict[float, float]]] = []  # [layer][seed] -> {σ: acc}

        for layer_idx in range(n_layers):
            train_X, train_y = all_train[layer_idx]
            test_X, test_y = all_test[layer_idx]

            # Train probe (fixed seed for reproducibility)
            hidden_dim = train_X.shape[1]
            torch.manual_seed(seed)
            probe = nn.Linear(hidden_dim, 1)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
            loss_fn = nn.BCEWithLogitsLoss()

            probe.train()
            for _ in range(n_epochs):
                logits = probe(train_X).squeeze(-1)
                loss = loss_fn(logits, train_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probe.eval()

            # Baseline accuracy
            with torch.no_grad():
                baseline_logits = probe(test_X).squeeze(-1)
                baseline_preds = (baseline_logits > 0).float()
                baseline_acc = (baseline_preds == test_y).float().mean().item()

            # Noised accuracies averaged over n_noise_seeds; also retain
            # per-seed accuracy curves so σ* can be bootstrapped over seeds.
            acc_mean: dict[float, float] = {}
            acc_std: dict[float, float] = {}
            per_seed_acc: list[dict[float, float]] = [dict() for _ in range(n_noise_seeds)]
            for noise_level in noise_levels:
                seed_accs = []
                for s in range(n_noise_seeds):
                    torch.manual_seed(seed + s)
                    with torch.no_grad():
                        noised_X = test_X + torch.randn_like(test_X) * noise_level
                        noised_logits = probe(noised_X).squeeze(-1)
                        noised_preds = (noised_logits > 0).float()
                        a = (noised_preds == test_y).float().mean().item()
                    seed_accs.append(a)
                    per_seed_acc[s][noise_level] = a
                acc_mean[noise_level] = float(np.mean(seed_accs))
                acc_std[noise_level] = float(np.std(seed_accs))

            # σ* from the seed-mean curve (cap-at-max).
            critical_capped = _sigma_star(acc_mean)
            never_fragile = all(acc_mean[nl] >= fragility_threshold for nl in noise_levels)
            critical_noise = None if never_fragile else critical_capped
            layer_seed_acc.append([dict(per_seed_acc[s]) for s in range(n_noise_seeds)])

            per_layer[layer_idx] = {
                "baseline_accuracy": round(baseline_acc, 4),
                "accuracy_by_noise": {str(k): round(v, 4) for k, v in acc_mean.items()},
                "accuracy_std_by_noise": {str(k): round(v, 4) for k, v in acc_std.items()},
                "critical_noise": critical_noise,
                "critical_noise_capped": round(float(critical_capped), 4),
            }

        # Summary for this foundation. The mean over capped σ* is cap-sensitive
        # when layers are censored (a single never-fragile layer counts as the
        # grid maximum); we also report the median, which is robust to that.
        capped = [d["critical_noise_capped"] for d in per_layer.values()]
        n_never_fragile = sum(1 for d in per_layer.values() if d["critical_noise"] is None)
        mean_critical = float(np.mean(capped)) if capped else None
        median_critical = float(np.median(capped)) if capped else None
        most_fragile_layer = min(per_layer.items(), key=lambda kv: kv[1]["critical_noise_capped"])[0]
        most_robust_layer = max(per_layer.items(), key=lambda kv: kv[1]["critical_noise_capped"])[0]

        # Bootstrap CI on the per-foundation mean σ*, consistent with the point
        # estimate: resample noise seeds, recompute each layer's seed-mean
        # accuracy curve, derive σ* (cap-at-max) from it, average over layers.
        rng = np.random.RandomState(seed)
        boot_means, boot_medians = [], []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, n_noise_seeds, size=n_noise_seeds)
            layer_sigmas = []
            for seed_curves in layer_seed_acc:
                mean_curve = {nl: float(np.mean([seed_curves[s][nl] for s in idx]))
                              for nl in noise_levels}
                layer_sigmas.append(_sigma_star(mean_curve))
            boot_means.append(float(np.mean(layer_sigmas)))
            boot_medians.append(float(np.median(layer_sigmas)))
        ci_lo, ci_hi = (float(np.percentile(boot_means, 2.5)),
                        float(np.percentile(boot_means, 97.5)))
        med_ci_lo, med_ci_hi = (float(np.percentile(boot_medians, 2.5)),
                                float(np.percentile(boot_medians, 97.5)))

        results[foundation_val] = {
            "per_layer": {str(k): v for k, v in per_layer.items()},
            "mean_critical_noise": round(float(mean_critical), 4) if mean_critical else None,
            "mean_critical_noise_ci": [round(ci_lo, 4), round(ci_hi, 4)],
            "median_critical_noise": round(float(median_critical), 4) if median_critical else None,
            "median_critical_noise_ci": [round(med_ci_lo, 4), round(med_ci_hi, 4)],
            "n_never_fragile": n_never_fragile,
            "most_fragile_layer": most_fragile_layer,
            "most_robust_layer": most_robust_layer,
            "n_train_pairs": len(train_pairs),
            "n_test_pairs": len(test_pairs),
        }

        print(f"    Mean σ*={mean_critical:.2f} [{ci_lo:.2f}, {ci_hi:.2f}], "
              f"median σ*={median_critical:.2f}, n_never_fragile={n_never_fragile}"
              if mean_critical is not None else
              "    No critical noise reached (all robust)")

    return results


def generate_fragility_figures(
    all_results: dict[str, dict],
    figures_dir: Path,
) -> None:
    """Generate framework-specific fragility figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "care_harm": "#E53935",
        "fairness_cheating": "#1E88E5",
        "liberty_oppression": "#43A047",
        "loyalty_betrayal": "#FB8C00",
        "authority_subversion": "#8E24AA",
        "sanctity_degradation": "#00ACC1",
    }

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

    for col, (model_label, foundation_results) in enumerate(all_results.items()):
        ax = axes[0, col]

        for fv in FOUNDATION_ORDER:
            if fv not in foundation_results:
                continue
            fdata = foundation_results[fv]
            layers = sorted(int(k) for k in fdata["per_layer"].keys())
            critical_values = []
            for l in layers:
                # Use the cap-at-max value so never-fragile layers plot at the
                # grid maximum (most robust), not as an inflated sentinel.
                cn = fdata["per_layer"][str(l)].get("critical_noise_capped")
                if cn is None:
                    raw = fdata["per_layer"][str(l)]["critical_noise"]
                    cn = raw if raw is not None else max(NOISE_LEVELS)
                critical_values.append(cn)

            marker = "o" if fv in INDIVIDUALIZING else "s"
            ax.plot(layers, critical_values, f"{marker}-",
                    color=colors.get(fv, "#666"), linewidth=2, markersize=5,
                    label=FOUNDATION_SHORT[fv])

        ax.set_xlabel("Layer", fontsize=11)
        ax.set_ylabel("Critical Noise (higher = more robust)", fontsize=11)
        ax.set_title(model_label, fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(0.05, 120)
        ax.legend(fontsize=8, loc="lower right", ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Framework-Specific Fragility: Critical Noise by Foundation\n"
                 "(circles = individualizing, squares = binding)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig5_framework_fragility.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "fig5_framework_fragility.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Framework fragility: {figures_dir / 'fig5_framework_fragility.png'}")

    # -- Summary bar chart: mean critical noise per foundation --
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(FOUNDATION_ORDER))
    width = 0.8 / n_models

    for i, (model_label, foundation_results) in enumerate(all_results.items()):
        means, lo_err, hi_err = [], [], []
        for fv in FOUNDATION_ORDER:
            fr = foundation_results.get(fv)
            m = fr["mean_critical_noise"] if fr and fr.get("mean_critical_noise") is not None else 0
            means.append(m)
            ci = (fr or {}).get("mean_critical_noise_ci")
            if ci:
                lo_err.append(max(0.0, m - ci[0]))
                hi_err.append(max(0.0, ci[1] - m))
            else:
                lo_err.append(0.0)
                hi_err.append(0.0)
        ax.bar(x + i * width, means, width, label=model_label,
               color=["#2196F3", "#F44336"][i] if n_models > 1 else "#2196F3",
               alpha=0.8,
               yerr=[lo_err, hi_err], capsize=3,
               error_kw={"elinewidth": 1, "alpha": 0.6})

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels([FOUNDATION_SHORT[f] for f in FOUNDATION_ORDER], fontsize=10)
    ax.set_ylabel("Mean Critical Noise", fontsize=11)
    ax.set_title("Mean Critical Noise by Foundation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(figures_dir / "exp7_mean_critical_bars.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / "exp7_mean_critical_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Mean critical bars: {figures_dir / 'exp7_mean_critical_bars.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 7: Framework-specific fragility.",
    )
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/exp7_fragility")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument("--model", default=OLMO_REPO,
                        help="HuggingFace model ID for the dense OLMo model.")
    parser.add_argument("--olmo-only", action="store_true")
    parser.add_argument("--olmoe-only", action="store_true")
    parser.add_argument("--grid", choices=["default", "extended"], default="extended",
                        help="default = {0.1..10}; extended = {0.1..100} (lifts σ=10 cap).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    grid = list(NOISE_LEVELS) if args.grid == "default" else list(EXTENDED_NOISE_LEVELS)

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
    figures_dir = Path("papers/3_moral_geometry/outputs/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target, dataset_version="v2")
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    all_results: dict[str, dict] = {}

    if not args.olmoe_only:
        print(f"\n{'='*60}")
        print(f"Loading dense OLMo: {args.model}")
        print(f"{'='*60}")
        t0 = time.time()
        olmo_model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
        print(f"Loaded in {time.time() - t0:.1f}s")

        olmo_label = _model_label(args.model)
        print(f"\n{'='*60}")
        print(f"EXPERIMENT 7: Foundation Fragility on {olmo_label}")
        print(f"{'='*60}")
        t0 = time.time()
        olmo_results = run_foundation_fragility(olmo_model, dataset, output_dir, "OLMo", noise_levels=grid)
        print(f"OLMo fragility complete: {time.time() - t0:.1f}s")

        with open(output_dir / "exp7_olmo_fragility.json", "w") as f:
            json.dump({"model": args.model, "per_foundation": olmo_results}, f, indent=2)

        all_results[f"{olmo_label} (dense)"] = olmo_results

        del olmo_model
        _clear_memory()

    if not args.olmo_only:
        print(f"\n{'='*60}")
        print(f"Loading OLMoE-1B-7B: {OLMOE_REPO}")
        print(f"{'='*60}")
        t0 = time.time()
        olmoe_model = WhiteBoxModel(OLMOE_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
        print(f"Loaded in {time.time() - t0:.1f}s")

        print(f"\n{'='*60}")
        print("EXPERIMENT 7: Foundation Fragility on OLMoE")
        print(f"{'='*60}")
        t0 = time.time()
        olmoe_results = run_foundation_fragility(olmoe_model, dataset, output_dir, "OLMoE", noise_levels=grid)
        print(f"OLMoE fragility complete: {time.time() - t0:.1f}s")

        with open(output_dir / "exp7_olmoe_fragility.json", "w") as f:
            json.dump({"model": OLMOE_REPO, "per_foundation": olmoe_results}, f, indent=2)

        all_results["OLMoE-1B-7B (MoE)"] = olmoe_results

        del olmoe_model
        _clear_memory()

    # Generate figures
    if all_results:
        print(f"\n{'='*60}")
        print("Generating figures...")
        print(f"{'='*60}")
        generate_fragility_figures(all_results, figures_dir)

        # Print summary
        print(f"\n{'='*60}")
        print("FRAMEWORK FRAGILITY SUMMARY")
        print(f"{'='*60}")
        print(f"\n{'Foundation':15s}", end="")
        for model_label in all_results:
            print(f"  {model_label:>20s}", end="")
        print()
        for fv in FOUNDATION_ORDER:
            print(f"{FOUNDATION_SHORT[fv]:15s}", end="")
            for model_label, results in all_results.items():
                if fv in results and results[fv]["mean_critical_noise"] is not None:
                    print(f"  {results[fv]['mean_critical_noise']:>20.2f}", end="")
                else:
                    print(f"  {'N/A':>20s}", end="")
            print()

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
