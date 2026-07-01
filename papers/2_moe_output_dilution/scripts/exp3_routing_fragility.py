#!/usr/bin/env python3
"""Experiment 3: Routing vs Expert Fragility.

Isolates the source of MoE moral fragility by perturbing three components:
  A) Router logits — changes expert selection and weights
  B) Expert outputs — degrades individual expert representations
  C) Aggregated output — direct noise on MoE output (control)

For each condition, probes trained on clean data are evaluated on perturbed
outputs to find critical noise sigma* (accuracy < 0.6).

Hypothesis: Router perturbation has lower sigma* than expert perturbation,
confirming that MoE fragility is a routing problem, not a representation problem.

Hardware: MacBook Pro M4 Pro, 24 GB, MPS
Estimated runtime: ~5 min (2.5 min collection, rest is fast matrix ops)

Usage:
    python papers/2_moe_output_dilution/scripts/exp3_routing_fragility.py
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from deepsteer.core.device import enable_mps_histc_fallback  # noqa: E402
enable_mps_histc_fallback()

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"
NOISE_LEVELS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
FRAGILITY_THRESHOLD = 0.6
N_SEEDS = 10
TOP_K = 8


@torch.no_grad()
def collect_moe_data(model, tokenizer, texts, layers, device):
    """Collect per-expert outputs, router logits, and clean aggregated MoE outputs.

    Returns (all on CPU):
        expert_acts: dict[layer, Tensor(n_texts, 64, hidden_dim)] float16
        router_logits: dict[layer, Tensor(n_texts, 64)] float32
        clean_agg: dict[layer, Tensor(n_texts, hidden_dim)] float16
    """
    expert_acts = {l: [] for l in layers}
    router_logits = {l: [] for l in layers}
    clean_agg = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Collecting MoE data: %d/%d", i + 1, len(texts))

        pre_moe = {}
        hooks = []

        for li in layers:
            layer_mod = model.model.layers[li]
            def _hook(mod, inp, out, idx=li):
                pre_moe[idx] = out.detach().cpu()
            hooks.append(
                layer_mod.post_attention_layernorm.register_forward_hook(_hook)
            )

        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        for li in layers:
            hidden = pre_moe[li].to(device).squeeze(0)  # (seq, hidden_dim)

            experts_mod = model.model.layers[li].mlp.experts
            gate_weight = model.model.layers[li].mlp.gate.weight.detach()  # (64, hidden_dim)
            gate_up_proj = experts_mod.gate_up_proj.detach()
            down_proj = experts_mod.down_proj.detach()
            act_fn = experts_mod.act_fn

            # Mean-pool across sequence
            h_mean = hidden.mean(dim=0)  # (hidden_dim,)

            # Router logits on mean-pooled hidden
            logits = h_mean @ gate_weight.T  # (64,)
            router_logits[li].append(logits.float().cpu())

            # All 64 expert outputs on mean-pooled hidden
            gate_up = torch.einsum("h,eoh->eo", h_mean, gate_up_proj)
            gate_part, up_part = gate_up.chunk(2, dim=-1)
            intermediate = act_fn(gate_part) * up_part
            expert_out = torch.einsum("eo,eho->eh", intermediate, down_proj)  # (64, hidden_dim)
            expert_acts[li].append(expert_out.half().cpu())

            # Clean aggregated: top-k weighted sum
            probs = F.softmax(logits, dim=-1)
            topk_p, topk_i = torch.topk(probs, k=TOP_K)
            topk_w = topk_p / topk_p.sum()
            selected = expert_out[topk_i]
            agg = (topk_w.unsqueeze(-1) * selected).sum(dim=0)
            clean_agg[li].append(agg.half().cpu())

    for li in layers:
        expert_acts[li] = torch.stack(expert_acts[li])
        router_logits[li] = torch.stack(router_logits[li])
        clean_agg[li] = torch.stack(clean_agg[li])

    return expert_acts, router_logits, clean_agg


def train_clean_probes(clean_agg, labels, train_mask, test_mask, layers,
                       n_epochs=50, lr=1e-2):
    """Train one probe per layer on clean aggregated MoE outputs."""
    probes = {}
    clean_accs = {}

    for li in layers:
        X_train = clean_agg[li][train_mask].float()
        X_test = clean_agg[li][test_mask].float()
        y_train = labels[train_mask].float()
        y_test = labels[test_mask].float()

        probe = nn.Linear(X_train.shape[1], 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

        for _ in range(n_epochs):
            out = probe(X_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(out, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            preds = (probe(X_test).squeeze(-1) > 0).long()
            acc = (preds == y_test.long()).float().mean().item()

        probes[li] = probe
        clean_accs[li] = acc
        logger.info("  Layer %d clean probe: %.3f", li, acc)

    return probes, clean_accs


def perturb_and_evaluate(expert_acts, router_logits, clean_agg, probes, labels,
                         test_mask, noise_levels, layers, n_seeds=N_SEEDS):
    """Run all three perturbation conditions and return accuracy grids."""
    results = {"router": {}, "expert": {}, "output": {}}
    y_test = labels[test_mask]

    for li in layers:
        results["router"][li] = {}
        results["expert"][li] = {}
        results["output"][li] = {}
        probe = probes[li]

        li_logits = router_logits[li][test_mask]             # (n_test, 64)
        li_experts = expert_acts[li][test_mask].float()      # (n_test, 64, hidden_dim)
        li_agg = clean_agg[li][test_mask].float()            # (n_test, hidden_dim)
        n_test = li_experts.shape[0]

        # Clean routing (reused across expert-perturbation seeds)
        probs_clean = F.softmax(li_logits, dim=-1)
        topk_p_c, topk_i_c = torch.topk(probs_clean, k=TOP_K, dim=-1)
        topk_w_c = topk_p_c / topk_p_c.sum(dim=-1, keepdim=True)

        for sigma in noise_levels:
            accs_r, accs_e, accs_o = [], [], []

            for seed in range(n_seeds):
                # --- A: Router perturbation ---
                gen_r = torch.Generator().manual_seed(42 + seed)
                noise_r = torch.randn(li_logits.shape, generator=gen_r) * sigma
                perturbed_logits = li_logits + noise_r
                probs_r = F.softmax(perturbed_logits, dim=-1)
                topk_p_r, topk_i_r = torch.topk(probs_r, k=TOP_K, dim=-1)
                topk_w_r = topk_p_r / topk_p_r.sum(dim=-1, keepdim=True)

                agg_r = torch.zeros(n_test, li_experts.shape[2])
                for j in range(n_test):
                    sel = li_experts[j, topk_i_r[j]]
                    agg_r[j] = (topk_w_r[j].unsqueeze(-1) * sel).sum(dim=0)

                with torch.no_grad():
                    preds = (probe(agg_r).squeeze(-1) > 0).long()
                    accs_r.append((preds == y_test.long()).float().mean().item())

                # --- B: Expert perturbation ---
                gen_e = torch.Generator().manual_seed(42 + seed + 1000)
                noise_e = torch.randn(li_experts.shape, generator=gen_e) * sigma
                perturbed_experts = li_experts + noise_e

                agg_e = torch.zeros(n_test, li_experts.shape[2])
                for j in range(n_test):
                    sel = perturbed_experts[j, topk_i_c[j]]
                    agg_e[j] = (topk_w_c[j].unsqueeze(-1) * sel).sum(dim=0)

                with torch.no_grad():
                    preds = (probe(agg_e).squeeze(-1) > 0).long()
                    accs_e.append((preds == y_test.long()).float().mean().item())

                # --- C: Output perturbation ---
                gen_o = torch.Generator().manual_seed(42 + seed + 2000)
                noise_o = torch.randn(li_agg.shape, generator=gen_o) * sigma
                perturbed_agg = li_agg + noise_o

                with torch.no_grad():
                    preds = (probe(perturbed_agg).squeeze(-1) > 0).long()
                    accs_o.append((preds == y_test.long()).float().mean().item())

            results["router"][li][sigma] = float(np.mean(accs_r))
            results["expert"][li][sigma] = float(np.mean(accs_e))
            results["output"][li][sigma] = float(np.mean(accs_o))

        logger.info("  Layer %d perturbation done", li)

    return results


def find_critical(layer_results, threshold=FRAGILITY_THRESHOLD):
    for sigma in sorted(layer_results.keys()):
        if layer_results[sigma] < threshold:
            return sigma
    return None


def generate_plots(results, clean_accs, noise_levels, layers, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_layers = len(layers)
    conditions = [
        ("Router noise", "router", "#F44336", "s"),
        ("Expert noise", "expert", "#4CAF50", "o"),
        ("Output noise", "output", "#2196F3", "^"),
    ]

    # --- Fig 1: Critical noise bar chart ---
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_layers)
    width = 0.25

    for offset, (label, key, color, _) in zip([-width, 0, width], conditions):
        vals = [
            find_critical(results[key][li]) or max(noise_levels)
            for li in layers
        ]
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Critical Noise (log scale)", fontsize=12)
    ax.set_title(
        "Where Does MoE Fragility Live?\n"
        "Critical noise by perturbation target (higher = more robust)",
        fontsize=13, fontweight="bold",
    )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_critical_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: exp3_critical_noise.png")

    # --- Fig 2: Degradation curves at peak layer ---
    peak = max(clean_accs, key=clean_accs.get)
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, key, color, marker in conditions:
        accs = [results[key][peak][s] for s in noise_levels]
        ax.plot(noise_levels, accs, f"{marker}-", color=color, linewidth=2.5,
                markersize=8, label=label)

    ax.axhline(y=FRAGILITY_THRESHOLD, color="#9E9E9E", linestyle="--",
               linewidth=1, alpha=0.7, label=f"Threshold ({FRAGILITY_THRESHOLD})")
    ax.set_xlabel("Noise level", fontsize=12)
    ax.set_ylabel("Probe Accuracy", fontsize=12)
    ax.set_title(
        f"Probe Degradation Under Perturbation (Layer {peak})\n"
        f"Clean accuracy: {clean_accs[peak]:.1%}",
        fontsize=13,
    )
    ax.set_xscale("log")
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_degradation_peak.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: exp3_degradation_peak.png")

    # --- Fig 3: Perturbation heatmaps side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    for ax, (label, key, _, _) in zip(axes, conditions):
        matrix = np.array([
            [results[key][li][s] for s in noise_levels]
            for li in layers
        ])
        im = ax.imshow(
            matrix, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0,
            origin="lower",
        )
        ax.set_xlabel("Noise Level", fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(noise_levels)))
        ax.set_xticklabels([str(s) for s in noise_levels], rotation=45)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([str(l) for l in layers])
        plt.colorbar(im, ax=ax, shrink=0.8)

    axes[0].set_ylabel("Layer", fontsize=11)
    fig.suptitle(
        "Probe Accuracy Under Perturbation — OLMoE-1B-7B\n"
        "(Green = robust, Red = fragile)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: exp3_heatmaps.png")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: Routing vs Expert Fragility",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/2_moe_output_dilution/outputs/exp3_routing_fragility",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
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
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target, dataset_version="v2")

    all_texts = []
    labels_list = []
    train_idx, test_idx = [], []
    idx = 0
    for pair in dataset.train:
        all_texts.extend([pair.moral, pair.neutral])
        labels_list.extend([1, 0])
        train_idx.extend([idx, idx + 1])
        idx += 2
    for pair in dataset.test:
        all_texts.extend([pair.moral, pair.neutral])
        labels_list.extend([1, 0])
        test_idx.extend([idx, idx + 1])
        idx += 2

    labels = torch.tensor(labels_list)
    train_mask = torch.zeros(len(all_texts), dtype=torch.bool)
    test_mask = torch.zeros(len(all_texts), dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    print(f"Dataset: {len(all_texts)} texts ({train_mask.sum()} train, {test_mask.sum()} test)")

    print(f"\nLoading {OLMOE_REPO}...")
    t0 = time.time()
    model = WhiteBoxModel(OLMOE_REPO, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    layers = list(range(n_layers))
    print(f"Loaded in {time.time() - t0:.1f}s ({n_layers} layers)")

    print(f"\nCollecting MoE data ({len(all_texts)} texts x {n_layers} layers x 64 experts)...")
    t0 = time.time()
    expert_acts, router_logits, clean_agg = collect_moe_data(
        model._model, model._tokenizer, all_texts, layers, model._device,
    )
    elapsed_collect = time.time() - t0
    print(f"Collection done in {elapsed_collect:.1f}s")

    # Report signal scales for interpreting noise levels
    print("\nSignal scales (std across test set):")
    for li in [0, n_layers // 2, n_layers - 1]:
        rl_std = router_logits[li][test_mask].std().item()
        ea_std = expert_acts[li][test_mask].float().std().item()
        ca_std = clean_agg[li][test_mask].float().std().item()
        print(f"  Layer {li:>2d}: router_logits={rl_std:.3f}, "
              f"expert_out={ea_std:.3f}, aggregated={ca_std:.3f}")

    del model
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print(f"\nTraining {n_layers} clean probes on aggregated MoE outputs...")
    t0 = time.time()
    probes, clean_accs = train_clean_probes(
        clean_agg, labels, train_mask, test_mask, layers,
    )
    elapsed_train = time.time() - t0
    print(f"Probe training done in {elapsed_train:.1f}s")

    n_evals = len(NOISE_LEVELS) * 3 * n_layers * N_SEEDS
    print(f"\nRunning perturbation experiments ({n_evals} evaluations)...")
    t0 = time.time()
    results = perturb_and_evaluate(
        expert_acts, router_logits, clean_agg, probes, labels,
        test_mask, NOISE_LEVELS, layers,
    )
    elapsed_perturb = time.time() - t0
    print(f"Perturbation experiments done in {elapsed_perturb:.1f}s")

    # Compute critical noise
    critical = {}
    for cond in ["router", "expert", "output"]:
        critical[cond] = {}
        for li in layers:
            critical[cond][li] = find_critical(results[cond][li])

    print("\nGenerating plots...")
    generate_plots(results, clean_accs, NOISE_LEVELS, layers, output_dir)

    # Summary JSON
    summary = {
        "experiment": "exp3_routing_fragility",
        "model": OLMOE_REPO,
        "noise_levels": NOISE_LEVELS,
        "n_seeds": N_SEEDS,
        "fragility_threshold": FRAGILITY_THRESHOLD,
        "timings": {
            "collection_s": round(elapsed_collect, 1),
            "probe_training_s": round(elapsed_train, 1),
            "perturbation_s": round(elapsed_perturb, 1),
        },
        "clean_probe_accuracy": {str(k): round(v, 4) for k, v in clean_accs.items()},
        "perturbation_accuracy": {
            cond: {
                str(li): {str(s): round(results[cond][li][s], 4) for s in NOISE_LEVELS}
                for li in layers
            }
            for cond in ["router", "expert", "output"]
        },
        "critical_noise": {
            cond: {str(li): critical[cond][li] for li in layers}
            for cond in ["router", "expert", "output"]
        },
    }

    for cond in ["router", "expert", "output"]:
        vals = [v for v in critical[cond].values() if v is not None]
        summary[f"mean_critical_{cond}"] = round(float(np.mean(vals)), 4) if vals else None
        summary[f"n_fragile_layers_{cond}"] = len(vals)

    with open(output_dir / "exp3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print headline
    print(f"\n{'='*70}")
    print("EXPERIMENT 3: ROUTING VS EXPERT FRAGILITY")
    print(f"{'='*70}")

    def _fmt(v):
        return f"{v:>10.2f}" if v is not None else f"{'> max':>10s}"

    print(f"\n{'Layer':>6s}  {'Clean':>6s}  {'Router':>10s}  {'Expert':>10s}  {'Output':>10s}")
    for li in layers:
        print(f"{li:>6d}  {clean_accs[li]:>6.3f}  "
              f"{_fmt(critical['router'][li])}  "
              f"{_fmt(critical['expert'][li])}  "
              f"{_fmt(critical['output'][li])}")

    print(f"\nMean critical noise (over fragile layers):")
    for cond in ["router", "expert", "output"]:
        vals = [v for v in critical[cond].values() if v is not None]
        if vals:
            print(f"  {cond.capitalize():>8s}: {np.mean(vals):.3f} "
                  f"({len(vals)}/{n_layers} layers fragile)")
        else:
            print(f"  {cond.capitalize():>8s}: > {max(NOISE_LEVELS)} "
                  f"(no layers fragile)")

    r_vals = [v for v in critical["router"].values() if v is not None]
    e_vals = [v for v in critical["expert"].values() if v is not None]
    if r_vals and e_vals:
        ratio = np.mean(e_vals) / np.mean(r_vals)
        print(f"\n  Expert/Router fragility ratio: {ratio:.1f}x "
              f"(higher = router more fragile)")

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
