#!/usr/bin/env python3
"""Experiments 1+2: Per-expert moral probing and router analysis on OLMoE.

Exp 1: For each of 64 experts at each of 16 layers, compute the expert's
    FFN output on all input tokens (bypassing routing), mean-pool across
    the sequence, and train a binary moral/neutral probe. Produces a
    64x16 accuracy heatmap showing which experts encode moral signal.

Exp 2: Capture router logits during normal forward passes and compare
    expert selection distributions for moral vs neutral inputs.

Target: allenai/OLMoE-1B-7B-0924 (6.9B total, 1.3B active, 16 layers,
    64 experts/layer, top-8 routing)

Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS, fp16
Estimated runtime: ~30 min

Usage:
    python papers/moe_output_dilution/scripts/exp1_2_expert_probing.py
    python papers/moe_output_dilution/scripts/exp1_2_expert_probing.py --layers 0 5 10 15
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

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"

# MPS histc fix (same as exp5)
_orig_histc = torch.histc


def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)


torch.histc = _histc_mps_fallback


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Per-expert activation collection
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_expert_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    device: str,
) -> dict[int, torch.Tensor]:
    """Compute per-expert mean-pooled outputs for all texts.

    For each layer, manually applies all 64 expert FFNs to the pre-MoE
    hidden state (bypassing the router). Returns activations mean-pooled
    across the sequence dimension.

    Returns:
        dict[layer_idx, Tensor of shape (n_texts, 64, hidden_dim)]
    """
    n_experts = 64
    all_expert_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    all_router_logits: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Collecting expert activations: %d/%d", i + 1, len(texts))

        pre_moe_states: dict[int, torch.Tensor] = {}
        router_logits_per_layer: dict[int, torch.Tensor] = {}
        hooks = []

        # Hook pre-MoE hidden state (input to the MoE block = output of
        # post_attention_layernorm, which is the input to mlp.forward())
        for layer_idx in layers:
            layer_module = model.model.layers[layer_idx]

            def _pre_moe_hook(mod, inp, out, idx=layer_idx):
                pre_moe_states[idx] = out.detach().cpu()

            hooks.append(
                layer_module.post_attention_layernorm.register_forward_hook(_pre_moe_hook)
            )

            def _router_hook(mod, inp, out, idx=layer_idx):
                router_logits_per_layer[idx] = out[0].detach().cpu()

            hooks.append(layer_module.mlp.gate.register_forward_hook(_router_hook))

        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        # Compute per-expert outputs from pre-MoE hidden states
        for layer_idx in layers:
            hidden = pre_moe_states[layer_idx].to(device)  # (1, seq, hidden_dim)
            hidden = hidden.squeeze(0)  # (seq, hidden_dim)

            experts_module = model.model.layers[layer_idx].mlp.experts
            gate_up_proj = experts_module.gate_up_proj  # (64, 2*inter, hidden)
            down_proj = experts_module.down_proj  # (64, hidden, inter)
            act_fn = experts_module.act_fn

            # Batch all 64 experts: hidden @ gate_up_proj^T for each expert
            # hidden: (seq, hidden_dim), gate_up_proj: (64, 2*inter, hidden)
            # result: (64, seq, 2*inter)
            gate_up = torch.einsum("sh,eoh->eso", hidden, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)  # each (64, seq, inter)
            intermediate = act_fn(gate) * up  # (64, seq, inter)
            expert_out = torch.einsum("eso,eho->esh", intermediate, down_proj)
            # expert_out: (64, seq, hidden_dim)

            # Mean-pool across sequence
            expert_mean = expert_out.mean(dim=1)  # (64, hidden_dim)
            all_expert_acts[layer_idx].append(expert_mean.cpu())

            # Store router logits (already seq-mean-pooled later)
            all_router_logits[layer_idx].append(
                router_logits_per_layer[layer_idx]  # (seq, 64)
            )

    # Stack into tensors
    result_acts = {}
    result_router = {}
    for l in layers:
        result_acts[l] = torch.stack(all_expert_acts[l])  # (n_texts, 64, hidden_dim)
        result_router[l] = all_router_logits[l]  # list of (seq_i, 64) tensors

    return result_acts, result_router


# ---------------------------------------------------------------------------
# Per-expert probing (Experiment 1)
# ---------------------------------------------------------------------------


def train_expert_probes(
    expert_acts: dict[int, torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> dict[int, np.ndarray]:
    """Train a binary probe per expert per layer.

    Returns:
        dict[layer_idx, accuracy array of shape (64,)]
    """
    results = {}

    for layer_idx, acts in expert_acts.items():
        n_experts = acts.shape[1]
        accs = np.zeros(n_experts)

        train_acts = acts[train_mask]  # (n_train, 64, hidden_dim)
        test_acts = acts[test_mask]
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        for expert_idx in range(n_experts):
            X_train = train_acts[:, expert_idx, :].float()  # (n_train, hidden_dim)
            X_test = test_acts[:, expert_idx, :].float()
            y_train = train_labels.float()
            y_test = test_labels.float()

            hidden_dim = X_train.shape[1]
            probe = nn.Linear(hidden_dim, 1)
            optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

            for _ in range(n_epochs):
                logits = probe(X_train).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                preds = (probe(X_test).squeeze(-1) > 0).long()
                acc = (preds == y_test.long()).float().mean().item()
            accs[expert_idx] = acc

        results[layer_idx] = accs
        logger.info(
            "  Layer %d: expert acc range [%.3f, %.3f], mean=%.3f, max expert=%d",
            layer_idx, accs.min(), accs.max(), accs.mean(), accs.argmax(),
        )

    return results


# ---------------------------------------------------------------------------
# Router analysis (Experiment 2)
# ---------------------------------------------------------------------------


def analyze_routing(
    router_logits: dict[int, list[torch.Tensor]],
    labels: torch.Tensor,
    n_experts: int = 64,
) -> dict[int, dict]:
    """Analyze moral vs neutral routing patterns per layer."""
    moral_mask = labels == 1
    neutral_mask = labels == 0
    results = {}

    for layer_idx, logits_list in router_logits.items():
        # Mean router probability per sentence (average across tokens)
        mean_probs = torch.stack([lg.mean(dim=0) for lg in logits_list])  # (n_texts, 64)

        moral_probs = mean_probs[moral_mask].mean(dim=0)  # (64,)
        neutral_probs = mean_probs[neutral_mask].mean(dim=0)  # (64,)

        # Per-expert preference: moral_prob - neutral_prob
        preference = (moral_probs - neutral_probs).numpy()

        # Top-k selection frequency: count how often each expert is in the top-8
        top8_moral_counts = np.zeros(n_experts)
        top8_neutral_counts = np.zeros(n_experts)
        moral_total = 0
        neutral_total = 0

        for idx, logits in enumerate(logits_list):
            top8 = logits.topk(8, dim=-1).indices  # (seq, 8)
            for expert in top8.flatten().numpy():
                if labels[idx] == 1:
                    top8_moral_counts[expert] += 1
                else:
                    top8_neutral_counts[expert] += 1
            if labels[idx] == 1:
                moral_total += logits.shape[0] * 8
            else:
                neutral_total += logits.shape[0] * 8

        moral_freq = top8_moral_counts / max(moral_total, 1)
        neutral_freq = top8_neutral_counts / max(neutral_total, 1)
        freq_diff = moral_freq - neutral_freq

        # Experts with significant routing preference
        n_sig = int((np.abs(freq_diff) > 0.005).sum())

        results[layer_idx] = {
            "moral_mean_prob": moral_probs.numpy().tolist(),
            "neutral_mean_prob": neutral_probs.numpy().tolist(),
            "preference": preference.tolist(),
            "moral_top8_freq": moral_freq.tolist(),
            "neutral_top8_freq": neutral_freq.tolist(),
            "freq_diff": freq_diff.tolist(),
            "n_significant_preference": n_sig,
            "max_preference_expert": int(np.argmax(np.abs(preference))),
            "max_preference_magnitude": float(np.max(np.abs(preference))),
        }

        logger.info(
            "  Layer %d: %d experts with routing preference > 0.005, "
            "max preference=%.4f (expert %d)",
            layer_idx, n_sig,
            results[layer_idx]["max_preference_magnitude"],
            results[layer_idx]["max_preference_expert"],
        )

    return results


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def generate_plots(
    probe_results: dict[int, np.ndarray],
    routing_results: dict[int, dict],
    output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    layers = sorted(probe_results.keys())
    n_experts = 64

    # Figure 1: Per-expert probe accuracy heatmap
    matrix = np.zeros((n_experts, len(layers)))
    for col, layer_idx in enumerate(layers):
        matrix[:, col] = probe_results[layer_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(layers) * 0.7), 16))
    sns.heatmap(
        matrix,
        ax=ax,
        xticklabels=[str(l) for l in layers],
        yticklabels=[str(e) for e in range(n_experts)],
        cmap="RdYlGn",
        vmin=0.4,
        vmax=1.0,
        cbar_kws={"label": "Probe Accuracy"},
    )
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Expert Index", fontsize=12)
    ax.set_title(
        "Per-Expert Moral Probe Accuracy\n"
        f"OLMoE-1B-7B ({n_experts} experts × {len(layers)} layers)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_expert_accuracy_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: exp1_expert_accuracy_heatmap.png")

    # Gini coefficient of accuracy across experts at each layer
    gini_per_layer = {}
    for layer_idx in layers:
        accs = probe_results[layer_idx]
        sorted_accs = np.sort(accs)
        n = len(sorted_accs)
        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * sorted_accs)) / (n * np.sum(sorted_accs)) - (n + 1) / n
        gini_per_layer[layer_idx] = gini

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([str(l) for l in layers], [gini_per_layer[l] for l in layers],
           color="#F44336", alpha=0.7)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title(
        "Expert Moral Specialization (Gini Coefficient of Accuracy)\n"
        "Higher = more concentrated specialization",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "exp1_gini_per_layer.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: exp1_gini_per_layer.png")

    # Figure 2: Routing preference per layer
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i, layer_idx in enumerate(layers):
        ax = axes[i // 4][i % 4]
        pref = np.array(routing_results[layer_idx]["preference"])
        colors = ["#F44336" if p > 0 else "#2196F3" for p in pref]
        ax.bar(range(n_experts), pref, color=colors, width=1.0, alpha=0.7)
        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_xlim(-1, n_experts)
        ax.axhline(0, color="black", linewidth=0.5)
        if i >= 12:
            ax.set_xlabel("Expert", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("Moral - Neutral\nrouter prob", fontsize=8)

    fig.suptitle(
        "Router Moral Preference per Expert\n"
        "Red = routes moral tokens more; Blue = routes neutral tokens more",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_routing_preference.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: exp2_routing_preference.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiments 1+2: Per-expert probing and router analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/moe_output_dilution/outputs/exp1_2_expert_probing",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layers to probe (default: all 16).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import _resolve_device
    from deepsteer.datasets.pipeline import build_probing_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target)
    print(f"Dataset: {len(dataset.train)} train, {len(dataset.test)} test pairs")

    # Build texts and labels: moral=1, neutral=0
    # Interleave: train morals, train neutrals, test morals, test neutrals
    all_texts = []
    all_labels = []
    train_indices = []
    test_indices = []

    for pair in dataset.train:
        train_indices.append(len(all_texts))
        all_texts.append(pair.moral)
        all_labels.append(1)
        train_indices.append(len(all_texts))
        all_texts.append(pair.neutral)
        all_labels.append(0)

    for pair in dataset.test:
        test_indices.append(len(all_texts))
        all_texts.append(pair.moral)
        all_labels.append(1)
        test_indices.append(len(all_texts))
        all_texts.append(pair.neutral)
        all_labels.append(0)

    labels = torch.tensor(all_labels)
    train_mask = torch.zeros(len(all_texts), dtype=torch.bool)
    train_mask[train_indices] = True
    test_mask = torch.zeros(len(all_texts), dtype=torch.bool)
    test_mask[test_indices] = True

    print(f"Total texts: {len(all_texts)} ({train_mask.sum()} train, {test_mask.sum()} test)")

    # Load model
    print(f"\nLoading {OLMOE_REPO}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(OLMOE_REPO)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        OLMOE_REPO,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    load_time = time.time() - t0
    n_layers = len(model.model.layers)
    print(f"Loaded in {load_time:.1f}s ({n_layers} layers)")

    layers = args.layers or list(range(n_layers))
    print(f"Probing layers: {layers}")

    # Collect expert activations + router logits
    print(f"\nCollecting per-expert activations ({len(all_texts)} texts × "
          f"{len(layers)} layers × 64 experts)...")
    t0 = time.time()
    expert_acts, router_logits = collect_expert_activations(
        model, tokenizer, all_texts, layers, device,
    )
    collect_time = time.time() - t0
    print(f"Collection done in {collect_time:.1f}s")

    # Free model memory
    del model
    _clear_memory()

    # Experiment 1: Per-expert probing
    print(f"\nExperiment 1: Training {len(layers) * 64} per-expert probes...")
    t0 = time.time()
    probe_results = train_expert_probes(
        expert_acts, labels, train_mask, test_mask,
    )
    probe_time = time.time() - t0
    print(f"Probing done in {probe_time:.1f}s")

    # Experiment 2: Router analysis
    print(f"\nExperiment 2: Analyzing router moral preference...")
    t0 = time.time()
    routing_results = analyze_routing(router_logits, labels)
    route_time = time.time() - t0
    print(f"Routing analysis done in {route_time:.1f}s")

    # Generate plots
    print(f"\nGenerating plots...")
    generate_plots(probe_results, routing_results, output_dir)

    # Save results JSON
    summary = {
        "experiment": "exp1_2_expert_probing",
        "model": OLMOE_REPO,
        "n_layers": n_layers,
        "layers_probed": layers,
        "n_experts": 64,
        "n_texts": len(all_texts),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "timings": {
            "model_load_s": round(load_time, 1),
            "activation_collection_s": round(collect_time, 1),
            "probe_training_s": round(probe_time, 1),
            "routing_analysis_s": round(route_time, 1),
        },
        "exp1_probe_accuracy": {
            str(l): {
                "per_expert": probe_results[l].tolist(),
                "mean": round(float(probe_results[l].mean()), 4),
                "std": round(float(probe_results[l].std()), 4),
                "min": round(float(probe_results[l].min()), 4),
                "max": round(float(probe_results[l].max()), 4),
                "gini": round(float(
                    (2 * np.sum(np.arange(1, 65) * np.sort(probe_results[l])))
                    / (64 * np.sum(probe_results[l])) - 65 / 64
                ), 4),
                "n_above_90": int((probe_results[l] > 0.9).sum()),
                "n_above_80": int((probe_results[l] > 0.8).sum()),
                "n_below_60": int((probe_results[l] < 0.6).sum()),
                "best_expert": int(probe_results[l].argmax()),
                "worst_expert": int(probe_results[l].argmin()),
            }
            for l in layers
        },
        "exp2_routing": {
            str(l): {
                "n_significant_preference": routing_results[l]["n_significant_preference"],
                "max_preference_expert": routing_results[l]["max_preference_expert"],
                "max_preference_magnitude": routing_results[l]["max_preference_magnitude"],
            }
            for l in layers
        },
    }

    summary_path = output_dir / "exp1_2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    # Print headline results
    print(f"\n{'='*60}")
    print("EXPERIMENT 1: PER-EXPERT MORAL PROBING")
    print(f"{'='*60}")
    print(f"{'Layer':>6s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} "
          f"{'Gini':>8s} {'>90%':>6s} {'<60%':>6s}")
    for l in layers:
        s = summary["exp1_probe_accuracy"][str(l)]
        print(f"{l:>6d} {s['mean']:>8.3f} {s['std']:>8.3f} {s['min']:>8.3f} "
              f"{s['max']:>8.3f} {s['gini']:>8.4f} {s['n_above_90']:>6d} "
              f"{s['n_below_60']:>6d}")

    print(f"\n{'='*60}")
    print("EXPERIMENT 2: ROUTER MORAL PREFERENCE")
    print(f"{'='*60}")
    print(f"{'Layer':>6s} {'N sig':>8s} {'Max pref':>10s} {'Expert':>8s}")
    for l in layers:
        s = summary["exp2_routing"][str(l)]
        print(f"{l:>6d} {s['n_significant_preference']:>8d} "
              f"{s['max_preference_magnitude']:>10.4f} "
              f"{s['max_preference_expert']:>8d}")

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
