#!/usr/bin/env python3
"""Output scale comparison: OLMoE MoE block vs OLMo-2 dense MLP.

Measures the scale (std, L2 norm) of each model's feedforward block output
at every layer, using the same input texts. Tests the output-dilution
hypothesis: MoE aggregation produces smaller-scale outputs than dense MLPs,
explaining OLMoE's greater fragility.

Runtime: ~3 min (30s OLMoE load, 1 min OLMoE inference, 5s OLMo load, 20s OLMo inference)
"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_orig_histc = torch.histc

def _histc_mps_fallback(input, bins=100, min=0, max=0):
    if input.device.type == "mps" or not input.is_floating_point():
        return _orig_histc(input.cpu().float(), bins, min, max).to(input.device)
    return _orig_histc(input, bins, min, max)

torch.histc = _histc_mps_fallback

OLMOE_REPO = "allenai/OLMoE-1B-7B-0924"
OLMO_REPO = "allenai/OLMo-2-0425-1B"


@torch.no_grad()
def collect_ffn_scales(hf_model, tokenizer, texts, layers, device):
    """Collect feedforward block output statistics per layer.

    Hooks each layer's MLP (dense or MoE) to capture its output before
    residual addition, mean-pools across tokens, and computes scale stats.
    """
    ffn_outputs = {l: [] for l in layers}

    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info("  Processing: %d/%d", i + 1, len(texts))

        ffn_store = {}
        hooks = []

        for li in layers:
            def _hook(mod, inp, out, idx=li):
                o = out[0] if isinstance(out, tuple) else out
                ffn_store[idx] = o.detach()
            hooks.append(hf_model.model.layers[li].mlp.register_forward_hook(_hook))

        try:
            inputs = tokenizer(text, return_tensors="pt").to(device)
            hf_model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        for li in layers:
            out = ffn_store[li].float().mean(dim=1).squeeze(0)  # (hidden_dim,)
            ffn_outputs[li].append(out.cpu())

    results = {}
    for li in layers:
        stacked = torch.stack(ffn_outputs[li])  # (n_texts, hidden_dim)
        results[li] = {
            "output_std": round(float(stacked.std()), 6),
            "mean_l2_norm": round(float(stacked.norm(dim=1).mean()), 4),
            "mean_abs": round(float(stacked.abs().mean()), 6),
            "per_dim_std_mean": round(float(stacked.std(dim=0).mean()), 6),
        }

    return results


def generate_plot(olmoe_scales, olmo_scales, layers, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    x = np.array(layers)

    # Panel 1: Output std (overall)
    ax = axes[0]
    olmoe_std = [olmoe_scales[l]["output_std"] for l in layers]
    olmo_std = [olmo_scales[l]["output_std"] for l in layers]
    ax.plot(x, olmo_std, "o-", color="#2196F3", linewidth=2.5, markersize=7,
            label="OLMo-2 1B (dense)")
    ax.plot(x, olmoe_std, "s-", color="#F44336", linewidth=2.5, markersize=7,
            label="OLMoE-1B-7B (MoE)")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("FFN Output Std", fontsize=12)
    ax.set_title("Feedforward Output Scale by Layer", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel 2: Ratio (dense / MoE)
    ax = axes[1]
    ratios = [olmo_std[i] / olmoe_std[i] if olmoe_std[i] > 0 else 0
              for i in range(len(layers))]
    colors = ["#4CAF50" if r > 1 else "#F44336" for r in ratios]
    ax.bar(x, ratios, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=1.0, color="#9E9E9E", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Dense / MoE Output Scale Ratio", fontsize=12)
    ax.set_title("How Much Larger Is Dense MLP Output?", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "Output Dilution: MoE Aggregation Reduces Signal Scale\n"
        "(Same input texts, same 16-layer architecture)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    path = output_dir / "output_scale_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset

    output_dir = Path("papers/2_moe_output_dilution/outputs/exp3_routing_fragility")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building dataset...")
    dataset = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    texts = []
    for pair in dataset.train[:50]:
        texts.extend([pair.moral, pair.neutral])
    print(f"Using {len(texts)} texts")

    # --- OLMoE ---
    print(f"\nLoading {OLMOE_REPO}...")
    t0 = time.time()
    olmoe = WhiteBoxModel(OLMOE_REPO, access_tier=AccessTier.WEIGHTS)
    layers = list(range(olmoe.info.n_layers))
    print(f"Loaded in {time.time() - t0:.1f}s")

    print("Collecting OLMoE MoE output scales...")
    t0 = time.time()
    olmoe_scales = collect_ffn_scales(
        olmoe._model, olmoe._tokenizer, texts, layers, olmoe._device,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    del olmoe
    gc.collect()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # --- OLMo-2 ---
    print(f"\nLoading {OLMO_REPO}...")
    t0 = time.time()
    olmo = WhiteBoxModel(OLMO_REPO, access_tier=AccessTier.WEIGHTS)
    print(f"Loaded in {time.time() - t0:.1f}s")

    print("Collecting OLMo-2 MLP output scales...")
    t0 = time.time()
    olmo_scales = collect_ffn_scales(
        olmo._model, olmo._tokenizer, texts, layers, olmo._device,
    )
    print(f"Done in {time.time() - t0:.1f}s")

    del olmo
    gc.collect()

    # --- Compare ---
    print(f"\n{'='*70}")
    print("FFN OUTPUT SCALE COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Layer':>6s}  {'OLMoE std':>12s}  {'OLMo std':>12s}  {'Ratio':>8s}  "
          f"{'OLMoE norm':>12s}  {'OLMo norm':>12s}  {'Ratio':>8s}")

    ratios_std = []
    ratios_norm = []
    for li in layers:
        os = olmoe_scales[li]["output_std"]
        ms = olmo_scales[li]["output_std"]
        on = olmoe_scales[li]["mean_l2_norm"]
        mn = olmo_scales[li]["mean_l2_norm"]
        rs = ms / os if os > 1e-10 else float("inf")
        rn = mn / on if on > 1e-10 else float("inf")
        ratios_std.append(rs)
        ratios_norm.append(rn)
        print(f"{li:>6d}  {os:>12.6f}  {ms:>12.6f}  {rs:>7.1f}x  "
              f"{on:>12.4f}  {mn:>12.4f}  {rn:>7.1f}x")

    print(f"\nMean ratio (OLMo/OLMoE):")
    print(f"  Output std:    {np.mean(ratios_std):.1f}x")
    print(f"  L2 norm:       {np.mean(ratios_norm):.1f}x")

    generate_plot(olmoe_scales, olmo_scales, layers, output_dir)

    summary = {
        "experiment": "output_scale_comparison",
        "n_texts": len(texts),
        "olmoe": {str(k): v for k, v in olmoe_scales.items()},
        "olmo": {str(k): v for k, v in olmo_scales.items()},
        "std_ratios": {str(li): round(ratios_std[i], 2) for i, li in enumerate(layers)},
        "norm_ratios": {str(li): round(ratios_norm[i], 2) for i, li in enumerate(layers)},
        "mean_std_ratio": round(float(np.mean(ratios_std)), 2),
        "mean_norm_ratio": round(float(np.mean(ratios_norm)), 2),
    }
    path = output_dir / "output_scale_comparison.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
