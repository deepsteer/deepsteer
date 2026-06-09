#!/usr/bin/env python3
"""Generate Figure 2 — no expert moral specialization (§4.2).

Two-panel side-by-side figure built from 1,024 per-expert binary probes
(64 experts x 16 layers) on OLMoE-1B-7B.

Left panel: distribution of the 64 per-expert probe accuracies at each
layer (box plot), with the per-layer mean overlaid.  The distributions are
tight and uniformly high; no sparse subset of "moral experts" emerges.

Right panel: per-layer Gini coefficient of expert accuracy (concentration
of moral signal across experts).  Gini stays in [0.016, 0.023] -- near
perfect uniformity -- and is lowest at the late layers where encoding peaks.

Sources:
    outputs/exp1_2_expert_probing/exp1_2_summary.json

Outputs:
    figures/figure_2_expert_uniformity.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/2_moe_output_dilution")
EXP1_2_JSON = PAPER_DIR / "outputs/exp1_2_expert_probing/exp1_2_summary.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_2_expert_uniformity.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_2_expert_uniformity.png"

MOE_RED = "#F44336"


def main() -> None:
    with open(EXP1_2_JSON) as f:
        d = json.load(f)

    acc = d["exp1_probe_accuracy"]
    n_layers = d["n_layers"]
    layers = list(range(n_layers))
    per_expert = [acc[str(l)]["per_expert"] for l in layers]
    means = [acc[str(l)]["mean"] for l in layers]
    ginis = [acc[str(l)]["gini"] for l in layers]
    peak_layer = int(np.argmax(means))

    fig, (ax_box, ax_gini) = plt.subplots(
        1, 2, figsize=(11, 4.8), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    # --- Left: per-expert accuracy distribution by layer ------------
    ax_box.boxplot(
        per_expert, positions=layers, widths=0.6,
        showfliers=True, patch_artist=True,
        boxprops=dict(facecolor=MOE_RED, alpha=0.35, edgecolor=MOE_RED),
        medianprops=dict(color=MOE_RED, linewidth=1.5),
        whiskerprops=dict(color=MOE_RED, alpha=0.6),
        capprops=dict(color=MOE_RED, alpha=0.6),
        flierprops=dict(marker=".", markersize=3,
                        markerfacecolor="#9E9E9E", markeredgecolor="none"),
    )
    ax_box.plot(layers, means, "k--", linewidth=1.2, alpha=0.7, label="per-layer mean")
    ax_box.set_xlabel("Transformer layer", fontsize=10)
    ax_box.set_ylabel("Per-expert probe accuracy (64 experts)", fontsize=10)
    ax_box.set_xticks(layers)
    ax_box.set_xticklabels(layers)
    ax_box.set_ylim(0.6, 1.02)
    ax_box.grid(True, axis="y", alpha=0.3)
    ax_box.legend(loc="lower right", fontsize=9)
    ax_box.set_title("(a) every expert encodes moral content (no specialists)",
                     fontsize=10, loc="left")

    # --- Right: per-layer Gini coefficient --------------------------
    ax_gini.bar(layers, ginis, color=MOE_RED, alpha=0.75, width=0.7)
    ax_gini.axhline(0.03, color="#9E9E9E", linestyle=":", linewidth=1.2,
                    label="uniformity ceiling (0.03)")
    ax_gini.set_xlabel("Transformer layer", fontsize=10)
    ax_gini.set_ylabel("Gini coefficient of expert accuracy", fontsize=10)
    ax_gini.set_xticks(layers)
    ax_gini.set_ylim(0, 0.05)
    ax_gini.grid(True, axis="y", alpha=0.3)
    ax_gini.legend(loc="upper right", fontsize=9)
    ax_gini.set_title("(b) concentration stays near zero (uniform encoding)",
                      fontsize=10, loc="left")

    fig.suptitle(
        "No expert moral specialization: every expert encodes morality, "
        "concentration stays near zero (1,024 probes)",
        fontsize=11,
    )

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"  peak layer {peak_layer}: mean={means[peak_layer]:.3f} "
          f"gini={ginis[peak_layer]:.4f} "
          f"min={acc[str(peak_layer)]['min']:.3f}")
    print(f"  gini range: [{min(ginis):.4f}, {max(ginis):.4f}]")


if __name__ == "__main__":
    main()
