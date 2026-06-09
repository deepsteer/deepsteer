#!/usr/bin/env python3
"""Generate Figure 3 — output dilution explains MoE fragility (§4.4).

Two-panel side-by-side figure.

Left panel: per-layer feedforward output scale (standard deviation of the
mean-pooled FFN output) for the OLMoE MoE block vs. the dense OLMo-2 MLP,
on a log axis.  The dense MLP output is ~74x larger on average; the MoE
block operates orders of magnitude smaller.

Right panel: per-layer critical noise for the three MoE perturbation
targets (router logits, individual expert outputs, aggregated output).
The ranking reverses the natural hypothesis: the router is most robust
($\\sigma^*$ 9.1), the aggregated output most fragile ($\\sigma^*$ 0.56),
because the output operates on the diluted scale from the left panel.

Sources:
    outputs/exp3_routing_fragility/output_scale_comparison.json
    outputs/exp3_routing_fragility/exp3_summary.json

Outputs:
    figures/figure_3_output_dilution.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/2_moe_output_dilution")
SCALE_JSON = PAPER_DIR / "outputs/exp3_routing_fragility/output_scale_comparison.json"
EXP3_JSON = PAPER_DIR / "outputs/exp3_routing_fragility/exp3_summary.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_3_output_dilution.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_3_output_dilution.png"

# Critical-noise grid floor for None ("never fragile at tested noise").
NOISE_CEIL = 10.0

COMPONENTS = [
    ("router", "Router logits",      "#3F51B5", "o"),
    ("expert", "Expert outputs",     "#FF9800", "^"),
    ("output", "Aggregated output",  "#F44336", "s"),
]


def main() -> None:
    with open(SCALE_JSON) as f:
        scale = json.load(f)
    with open(EXP3_JSON) as f:
        exp3 = json.load(f)

    n_layers = 16
    layers = np.arange(n_layers)

    fig, (ax_scale, ax_frag) = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True,
    )

    # --- Left: feedforward output scale (MoE vs dense MLP) ----------
    moe_std = [scale["olmoe"][str(l)]["output_std"] for l in layers]
    mlp_std = [scale["olmo"][str(l)]["output_std"] for l in layers]
    ax_scale.plot(layers, mlp_std, "o-", color="#3F51B5", linewidth=2,
                  markersize=5, label="OLMo-2 dense MLP", alpha=0.9)
    ax_scale.plot(layers, moe_std, "s-", color="#F44336", linewidth=2,
                  markersize=5, label="OLMoE MoE block", alpha=0.9)
    ax_scale.set_xlabel("Transformer layer", fontsize=10)
    ax_scale.set_ylabel("Feedforward output std (log)", fontsize=10)
    ax_scale.set_xticks(layers)
    ax_scale.set_yscale("log")
    ax_scale.grid(True, which="both", alpha=0.3)
    ax_scale.legend(loc="upper left", fontsize=9)
    ax_scale.set_title(
        f"(a) MoE output is {scale['mean_std_ratio']:.0f}$\\times$ smaller "
        f"than the dense MLP",
        fontsize=10, loc="left",
    )

    # --- Right: component critical noise (the reversed ranking) -----
    crit = exp3["critical_noise"]
    for slug, label, color, marker in COMPONENTS:
        vals = [crit[slug][str(l)] for l in layers]
        vals = [(NOISE_CEIL if v is None else v) for v in vals]
        ax_frag.plot(layers, vals, marker + "-", color=color, linewidth=2,
                     markersize=5, label=label, alpha=0.9)
    ax_frag.set_xlabel("Transformer layer", fontsize=10)
    ax_frag.set_ylabel("Critical noise $\\sigma$ (log; higher = more robust)",
                       fontsize=10)
    ax_frag.set_xticks(layers)
    ax_frag.set_yscale("log")
    ax_frag.set_yticks([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    ax_frag.set_yticklabels(["0.01", "0.03", "0.1", "0.3", "1.0", "3.0", "10.0"])
    ax_frag.grid(True, which="both", alpha=0.3)
    ax_frag.legend(loc="lower right", fontsize=9)
    ax_frag.set_title(
        f"(b) ranking reverses: router robust ($\\sigma^*${exp3['mean_critical_router']:.1f}), "
        f"output fragile ($\\sigma^*${exp3['mean_critical_output']:.2f})",
        fontsize=9.5, loc="left",
    )

    fig.suptitle(
        "Output dilution: the MoE block injects a much smaller residual-stream "
        "signal, so its moral content is easier to overwhelm",
        fontsize=11,
    )

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"  mean MoE/MLP std ratio: {scale['mean_std_ratio']:.1f}x")
    for slug, label, _, _ in COMPONENTS:
        print(f"  {label:<18}: mean sigma* = {exp3['mean_critical_' + slug]:.3f}")


if __name__ == "__main__":
    main()
