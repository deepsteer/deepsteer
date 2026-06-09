#!/usr/bin/env python3
"""Generate Figure 1 — dense vs. MoE: same accuracy, different robustness (§4.1).

Two-panel side-by-side figure.

Left panel: per-layer moral probing accuracy for OLMoE-1B-7B and dense
OLMo-2 1B.  The two architectures are near-indistinguishable (both peak at
99.0%); OLMoE only trails at early layers 0-3.

Right panel: per-layer critical noise (the smallest $\\sigma$ at which probe
accuracy drops below 0.6).  OLMoE is 5.1x more fragile (mean $\\sigma^*$
0.84 vs. 4.25) and concentrates robustness in the final layers.

Sources:
    outputs/exp5_dense_vs_moe/exp5_summary.json

Outputs:
    figures/figure_1_dense_vs_moe.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/2_moe_output_dilution")
EXP5_JSON = PAPER_DIR / "outputs/exp5_dense_vs_moe/exp5_summary.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_1_dense_vs_moe.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_1_dense_vs_moe.png"

MODELS = [
    ("olmoe", "OLMoE-1B-7B (MoE)", "#F44336", "s"),
    ("olmo",  "OLMo-2 1B (dense)", "#3F51B5", "o"),
]


def main() -> None:
    with open(EXP5_JSON) as f:
        d = json.load(f)

    n_layers = d["olmoe"]["probe"]["n_layers"]
    layers = np.arange(n_layers)

    fig, (ax_acc, ax_frag) = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True,
    )

    # --- Left: per-layer probing accuracy ---------------------------
    for slug, label, color, marker in MODELS:
        accs = d[slug]["probe"]["per_layer_accuracy"]
        ax_acc.plot(layers, accs, marker + "-", color=color, linewidth=2,
                    markersize=5, label=label, alpha=0.9)
    ax_acc.set_xlabel("Transformer layer", fontsize=10)
    ax_acc.set_ylabel("Moral probing accuracy", fontsize=10)
    ax_acc.set_xticks(layers)
    ax_acc.set_ylim(0.75, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.set_title("(a) probing accuracy: indistinguishable (peak 99.0%)",
                     fontsize=10, loc="left")

    # --- Right: per-layer critical noise (log grid) -----------------
    for slug, label, color, marker in MODELS:
        crits = d[slug]["fragility"]["per_layer_critical_noise"]
        crits = [(0.1 if c is None else c) for c in crits]
        ax_frag.plot(layers, crits, marker + "-", color=color, linewidth=2,
                     markersize=5, label=label, alpha=0.9)
    ax_frag.set_xlabel("Transformer layer", fontsize=10)
    ax_frag.set_ylabel("Critical noise $\\sigma$ (log grid; higher = more robust)",
                       fontsize=10)
    ax_frag.set_xticks(layers)
    ax_frag.set_yscale("log")
    ax_frag.set_yticks([0.1, 0.3, 1.0, 3.0, 10.0])
    ax_frag.set_yticklabels(["0.1", "0.3", "1.0", "3.0", "10.0"])
    ax_frag.set_ylim(0.08, 13)
    ax_frag.grid(True, which="both", alpha=0.3)
    ax_frag.legend(loc="upper left", fontsize=9)
    moe_mean = d["olmoe"]["fragility"]["mean_critical_noise"]
    dense_mean = d["olmo"]["fragility"]["mean_critical_noise"]
    ax_frag.set_title(
        f"(b) fragility: MoE {dense_mean / moe_mean:.1f}$\\times$ more fragile "
        f"($\\sigma^*$ {moe_mean:.2f} vs. {dense_mean:.2f})",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "Dense vs. MoE: same moral-probing accuracy, very different fragility "
        "(OLMoE-1B-7B vs. OLMo-2 1B)",
        fontsize=11,
    )

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    for slug, label, _, _ in MODELS:
        pr = d[slug]["probe"]
        fr = d[slug]["fragility"]
        print(f"  {label:<20}: peak_acc={pr['peak_accuracy']:.3f} @L{pr['peak_layer']}  "
              f"mean_crit_noise={fr['mean_critical_noise']:.3f}")


if __name__ == "__main__":
    main()
