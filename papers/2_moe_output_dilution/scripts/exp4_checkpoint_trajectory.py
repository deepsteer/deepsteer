#!/usr/bin/env python3
"""Experiment 4: Checkpoint trajectory analysis for OLMoE.

Tracks per-expert moral specialization (Gini coefficient) and mean probe
accuracy across OLMoE training checkpoints. Tests whether expert moral
specialization emerges during training or remains absent throughout.

Selects ~18 checkpoints spanning training: dense early sampling +
logarithmic spacing. At each checkpoint, runs per-expert probing
(Experiment 1) and router analysis (Experiment 2).

Supports resume: each checkpoint's results are saved to individual JSON
files; completed checkpoints are skipped on re-run.

Target: allenai/OLMoE-1B-7B-0924 (244 training checkpoints available)
Hardware: MacBook Pro M4 Pro, 24 GB unified memory, MPS, fp16
Estimated runtime: ~1 hour compute + download time

Usage:
    python papers/2_moe_output_dilution/scripts/exp4_checkpoint_trajectory.py
    python papers/2_moe_output_dilution/scripts/exp4_checkpoint_trajectory.py --max-checkpoints 5
    python papers/2_moe_output_dilution/scripts/exp4_checkpoint_trajectory.py --steps 5000 50000 500000
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
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"

DEFAULT_TARGET_STEPS = [
    5000, 10000, 20000, 50000, 100000,
    200000, 400000, 600000,
    800000, 1000000, 1200000,
]

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


def resolve_checkpoint_branches(target_steps: list[int]) -> list[tuple[int, str]]:
    """Map target step numbers to exact HuggingFace branch names."""
    from huggingface_hub import list_repo_refs

    refs = list_repo_refs(OLMOE_REPO)
    branches = {}
    for b in refs.branches:
        m = re.match(r"step(\d+)-tokens\d+B", b.name)
        if m:
            branches[int(m.group(1))] = b.name

    resolved = []
    for step in sorted(target_steps):
        if step in branches:
            resolved.append((step, branches[step]))
        else:
            closest = min(branches.keys(), key=lambda s: abs(s - step))
            logger.warning(
                "Step %d not found, using closest: %d (%s)",
                step, closest, branches[closest],
            )
            resolved.append((closest, branches[closest]))

    return resolved


def gini_coefficient(values: np.ndarray) -> float:
    sorted_v = np.sort(values)
    n = len(sorted_v)
    index = np.arange(1, n + 1)
    return float(
        (2 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1) / n
    )


@torch.no_grad()
def collect_expert_activations(
    model,
    tokenizer,
    texts: list[str],
    layers: list[int],
    device: str,
) -> tuple[dict[int, torch.Tensor], dict[int, list[torch.Tensor]]]:
    """Compute per-expert mean-pooled outputs for all texts at given layers."""
    all_expert_acts: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    all_router_logits: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if (i + 1) % 100 == 0:
            logger.info("  Activations: %d/%d", i + 1, len(texts))

        pre_moe_states: dict[int, torch.Tensor] = {}
        router_logits_per_layer: dict[int, torch.Tensor] = {}
        hooks = []

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

        for layer_idx in layers:
            hidden = pre_moe_states[layer_idx].to(device).squeeze(0)

            experts_module = model.model.layers[layer_idx].mlp.experts
            gate_up_proj = experts_module.gate_up_proj
            down_proj = experts_module.down_proj
            act_fn = experts_module.act_fn

            gate_up = torch.einsum("sh,eoh->eso", hidden, gate_up_proj)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = act_fn(gate) * up
            expert_out = torch.einsum("eso,eho->esh", intermediate, down_proj)

            expert_mean = expert_out.mean(dim=1)
            all_expert_acts[layer_idx].append(expert_mean.cpu())
            all_router_logits[layer_idx].append(router_logits_per_layer[layer_idx])

    result_acts = {l: torch.stack(all_expert_acts[l]) for l in layers}
    result_router = {l: all_router_logits[l] for l in layers}
    return result_acts, result_router


def train_expert_probes(
    expert_acts: dict[int, torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-2,
) -> dict[int, np.ndarray]:
    """Train a binary probe per expert per layer. Returns accuracy arrays."""
    results = {}
    for layer_idx, acts in expert_acts.items():
        n_experts = acts.shape[1]
        accs = np.zeros(n_experts)
        train_acts = acts[train_mask]
        test_acts = acts[test_mask]
        train_labels = labels[train_mask]
        test_labels = labels[test_mask]

        for expert_idx in range(n_experts):
            X_train = train_acts[:, expert_idx, :].float()
            X_test = test_acts[:, expert_idx, :].float()
            y_train = train_labels.float()
            y_test = test_labels.float()

            probe = nn.Linear(X_train.shape[1], 1)
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
    return results


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
        mean_probs = torch.stack([lg.mean(dim=0) for lg in logits_list])
        moral_probs = mean_probs[moral_mask].mean(dim=0)
        neutral_probs = mean_probs[neutral_mask].mean(dim=0)
        preference = (moral_probs - neutral_probs).numpy()

        results[layer_idx] = {
            "max_preference_magnitude": float(np.max(np.abs(preference))),
            "max_preference_expert": int(np.argmax(np.abs(preference))),
        }

    return results


def run_single_checkpoint(
    revision: str,
    step: int,
    texts: list[str],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: str,
    output_dir: Path,
) -> dict:
    """Run full per-expert probing + routing analysis on one checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt_file = output_dir / f"ckpt_step{step}.json"
    if ckpt_file.exists():
        logger.info("Checkpoint step %d already done, loading cached result", step)
        with open(ckpt_file) as f:
            return json.load(f)

    logger.info("Loading checkpoint: %s (step %d)", revision, step)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(OLMOE_REPO, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        OLMOE_REPO,
        revision=revision,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()
    load_time = time.time() - t0

    n_layers = len(model.model.layers)
    layers = list(range(n_layers))

    logger.info("  Loaded in %.1fs (%d layers), collecting activations...", load_time, n_layers)
    t1 = time.time()
    expert_acts, router_logits = collect_expert_activations(
        model, tokenizer, texts, layers, device,
    )
    collect_time = time.time() - t1

    del model, tokenizer
    _clear_memory()

    logger.info("  Activations collected in %.1fs, training probes...", collect_time)
    t2 = time.time()
    probe_results = train_expert_probes(expert_acts, labels, train_mask, test_mask)
    probe_time = time.time() - t2

    t3 = time.time()
    routing_results = analyze_routing(router_logits, labels)
    route_time = time.time() - t3

    del expert_acts, router_logits
    _clear_memory()

    layer_summaries = {}
    for l in layers:
        accs = probe_results[l]
        layer_summaries[str(l)] = {
            "mean_accuracy": round(float(accs.mean()), 4),
            "std_accuracy": round(float(accs.std()), 4),
            "min_accuracy": round(float(accs.min()), 4),
            "max_accuracy": round(float(accs.max()), 4),
            "gini": round(gini_coefficient(accs), 4),
            "n_above_90": int((accs > 0.9).sum()),
            "n_above_75": int((accs > 0.75).sum()),
            "top5_experts": [int(x) for x in np.argsort(accs)[-5:][::-1]],
            "top5_accuracies": [round(float(accs[x]), 4) for x in np.argsort(accs)[-5:][::-1]],
            "router_max_preference": routing_results[l]["max_preference_magnitude"],
        }

    peak_layer = max(layers, key=lambda l: probe_results[l].mean())

    result = {
        "step": step,
        "revision": revision,
        "n_layers": n_layers,
        "timings": {
            "load_s": round(load_time, 1),
            "collect_s": round(collect_time, 1),
            "probe_s": round(probe_time, 1),
            "route_s": round(route_time, 1),
            "total_s": round(load_time + collect_time + probe_time + route_time, 1),
        },
        "peak_layer": peak_layer,
        "peak_mean_accuracy": round(float(probe_results[peak_layer].mean()), 4),
        "peak_gini": round(gini_coefficient(probe_results[peak_layer]), 4),
        "overall_mean_accuracy": round(float(
            np.mean([probe_results[l].mean() for l in layers])
        ), 4),
        "overall_mean_gini": round(float(
            np.mean([gini_coefficient(probe_results[l]) for l in layers])
        ), 4),
        "layers": layer_summaries,
    }

    with open(ckpt_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "  Step %d done: peak_acc=%.3f, peak_gini=%.4f, peak_layer=%d (%.1fs total)",
        step, result["peak_mean_accuracy"], result["peak_gini"],
        peak_layer, result["timings"]["total_s"],
    )
    return result


def generate_trajectory_plots(results: list[dict], output_dir: Path) -> None:
    """Generate trajectory plots from checkpoint results."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = sorted(results, key=lambda r: r["step"])
    steps = [r["step"] for r in results]
    steps_k = [s / 1000 for s in steps]

    peak_accs = [r["peak_mean_accuracy"] for r in results]
    peak_ginis = [r["peak_gini"] for r in results]
    overall_accs = [r["overall_mean_accuracy"] for r in results]
    overall_ginis = [r["overall_mean_gini"] for r in results]

    # Figure 1: Headline — Gini vs mean accuracy across training
    fig, ax1 = plt.subplots(figsize=(12, 5))
    color_acc = "#2196F3"
    color_gini = "#F44336"

    ax1.set_xlabel("Training Step (×1000)", fontsize=12)
    ax1.set_ylabel("Mean Expert Accuracy (peak layer)", fontsize=12, color=color_acc)
    ax1.plot(steps_k, peak_accs, "o-", color=color_acc, linewidth=2, markersize=5,
             label="Peak-layer mean accuracy")
    ax1.plot(steps_k, overall_accs, "s--", color=color_acc, linewidth=1, markersize=4,
             alpha=0.5, label="All-layer mean accuracy")
    ax1.tick_params(axis="y", labelcolor=color_acc)
    ax1.set_ylim(0.45, 1.02)
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Chance")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Gini Coefficient", fontsize=12, color=color_gini)
    ax2.plot(steps_k, peak_ginis, "^-", color=color_gini, linewidth=2, markersize=5,
             label="Peak-layer Gini")
    ax2.plot(steps_k, overall_ginis, "v--", color=color_gini, linewidth=1, markersize=4,
             alpha=0.5, label="All-layer mean Gini")
    ax2.tick_params(axis="y", labelcolor=color_gini)
    ax2.set_ylim(-0.01, 0.15)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title(
        "Expert Moral Specialization Across Training\n"
        "OLMoE-1B-7B: Gini coefficient stays near zero throughout",
        fontsize=13,
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "exp4_trajectory_headline.png", dpi=150)
    plt.close(fig)

    # Figure 2: Per-layer accuracy heatmap across training
    n_layers = results[0]["n_layers"]
    acc_matrix = np.zeros((n_layers, len(results)))
    gini_matrix = np.zeros((n_layers, len(results)))
    for col, r in enumerate(results):
        for l in range(n_layers):
            acc_matrix[l, col] = r["layers"][str(l)]["mean_accuracy"]
            gini_matrix[l, col] = r["layers"][str(l)]["gini"]

    fig, (ax_acc, ax_gini) = plt.subplots(1, 2, figsize=(16, 6))

    import matplotlib.colors as mcolors
    im1 = ax_acc.imshow(acc_matrix, aspect="auto", cmap="RdYlGn",
                         vmin=0.45, vmax=1.0, origin="lower")
    ax_acc.set_xlabel("Checkpoint", fontsize=11)
    ax_acc.set_ylabel("Layer", fontsize=11)
    ax_acc.set_title("Mean Expert Accuracy", fontsize=12)
    ax_acc.set_xticks(range(len(results)))
    ax_acc.set_xticklabels([f"{s/1000:.0f}K" for s in steps], rotation=45, fontsize=7)
    ax_acc.set_yticks(range(n_layers))
    plt.colorbar(im1, ax=ax_acc, label="Accuracy")

    im2 = ax_gini.imshow(gini_matrix, aspect="auto", cmap="Reds",
                          vmin=0, vmax=0.1, origin="lower")
    ax_gini.set_xlabel("Checkpoint", fontsize=11)
    ax_gini.set_ylabel("Layer", fontsize=11)
    ax_gini.set_title("Gini Coefficient (Specialization)", fontsize=12)
    ax_gini.set_xticks(range(len(results)))
    ax_gini.set_xticklabels([f"{s/1000:.0f}K" for s in steps], rotation=45, fontsize=7)
    ax_gini.set_yticks(range(n_layers))
    plt.colorbar(im2, ax=ax_gini, label="Gini")

    fig.suptitle("OLMoE-1B-7B: Expert Moral Encoding Across Training", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "exp4_trajectory_heatmaps.png", dpi=150)
    plt.close(fig)

    # Figure 3: Expert identity stability
    peak_layers = [r["peak_layer"] for r in results]
    top5_sets = []
    for r in results:
        pl = r["peak_layer"]
        top5_sets.append(set(r["layers"][str(pl)]["top5_experts"]))

    jaccard_over_time = []
    for i in range(1, len(top5_sets)):
        intersection = len(top5_sets[i] & top5_sets[i - 1])
        union = len(top5_sets[i] | top5_sets[i - 1])
        jaccard_over_time.append(intersection / union if union > 0 else 0)

    if jaccard_over_time:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(steps_k[1:], jaccard_over_time, "o-", color="#9C27B0", linewidth=2, markersize=5)
        ax.set_xlabel("Training Step (×1000)", fontsize=12)
        ax.set_ylabel("Jaccard Similarity\n(top-5 experts vs. previous)", fontsize=11)
        ax.set_title("Expert Identity Stability Across Training", fontsize=13)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(5 / 64, color="gray", linestyle=":", alpha=0.5, label="Random baseline (5/64)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "exp4_expert_identity_stability.png", dpi=150)
        plt.close(fig)

    logger.info("Trajectory plots saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 4: OLMoE checkpoint trajectory analysis.",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/2_moe_output_dilution/outputs/exp4_checkpoint_trajectory",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=None,
        help="Specific training steps to analyze (default: 17 log-spaced).",
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=None,
        help="Limit number of checkpoints (for testing).",
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

    target_steps = args.steps or DEFAULT_TARGET_STEPS
    if args.max_checkpoints:
        target_steps = target_steps[:args.max_checkpoints]

    print(f"\nResolving {len(target_steps)} checkpoint branches...")
    checkpoints = resolve_checkpoint_branches(target_steps)
    print(f"Checkpoints to process: {[s for s, _ in checkpoints]}")

    n_cached = sum(1 for s, _ in checkpoints if (output_dir / f"ckpt_step{s}.json").exists())
    if n_cached > 0:
        print(f"  ({n_cached} already cached, will be skipped)")

    all_results = []
    total_t0 = time.time()

    for i, (step, revision) in enumerate(checkpoints):
        print(f"\n{'='*60}")
        print(f"Checkpoint {i+1}/{len(checkpoints)}: step {step} ({revision})")
        print(f"{'='*60}")

        result = run_single_checkpoint(
            revision=revision,
            step=step,
            texts=all_texts,
            labels=labels,
            train_mask=train_mask,
            test_mask=test_mask,
            device=device,
            output_dir=output_dir,
        )
        all_results.append(result)

        elapsed = time.time() - total_t0
        per_ckpt = elapsed / (i + 1)
        remaining = per_ckpt * (len(checkpoints) - i - 1)
        print(f"  Elapsed: {elapsed/60:.1f}min, est remaining: {remaining/60:.1f}min")

    total_time = time.time() - total_t0

    # Aggregate summary
    all_results_sorted = sorted(all_results, key=lambda r: r["step"])
    summary = {
        "experiment": "exp4_checkpoint_trajectory",
        "model": OLMOE_REPO,
        "n_checkpoints": len(all_results_sorted),
        "total_time_s": round(total_time, 1),
        "checkpoints": [
            {
                "step": r["step"],
                "revision": r["revision"],
                "peak_layer": r["peak_layer"],
                "peak_mean_accuracy": r["peak_mean_accuracy"],
                "peak_gini": r["peak_gini"],
                "overall_mean_accuracy": r["overall_mean_accuracy"],
                "overall_mean_gini": r["overall_mean_gini"],
            }
            for r in all_results_sorted
        ],
    }

    summary_path = output_dir / "exp4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EXPERIMENT 4: CHECKPOINT TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    print(f"{'Step':>10s} {'Peak Acc':>10s} {'Peak Gini':>10s} {'Overall Acc':>12s} "
          f"{'Overall Gini':>12s} {'Peak Layer':>10s}")
    for r in all_results_sorted:
        print(f"{r['step']:>10d} {r['peak_mean_accuracy']:>10.4f} "
              f"{r['peak_gini']:>10.4f} {r['overall_mean_accuracy']:>12.4f} "
              f"{r['overall_mean_gini']:>12.4f} {r['peak_layer']:>10d}")

    print(f"\nTotal time: {total_time/60:.1f} min")

    print("\nGenerating trajectory plots...")
    generate_trajectory_plots(all_results_sorted, output_dir)

    print(f"\nAll outputs: {output_dir}")


if __name__ == "__main__":
    main()
