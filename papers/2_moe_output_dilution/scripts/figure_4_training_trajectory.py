#!/usr/bin/env python3
"""Generate Figure 4 — specialization never emerges during training (§4.5).

Single twin-axis figure across 11 OLMoE checkpoints (step 5K -> 1.2M).

Left axis: peak-layer mean per-expert accuracy and overall mean accuracy.
Moral encoding is present from the earliest checkpoint and stays in a
narrow band rather than progressively sharpening.

Right axis: peak-layer Gini and overall mean Gini of per-expert accuracy.
Gini stays flat near zero (and mildly decreases), so the model never
concentrates moral content into specific experts.

Sources:
    outputs/exp4_checkpoint_trajectory/exp4_summary.json

Outputs:
    figures/figure_4_training_trajectory.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER_DIR = Path("papers/2_moe_output_dilution")
EXP4_JSON = PAPER_DIR / "outputs/exp4_checkpoint_trajectory/exp4_summary.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_4_training_trajectory.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_4_training_trajectory.png"

ACC_COLOR = "#2E7D32"
GINI_COLOR = "#F44336"


def main() -> None:
    with open(EXP4_JSON) as f:
        d = json.load(f)
    ckpts = sorted(d["checkpoints"], key=lambda c: c["step"])
    steps = [c["step"] for c in ckpts]
    peak_acc = [c["peak_mean_accuracy"] for c in ckpts]
    overall_acc = [c["overall_mean_accuracy"] for c in ckpts]
    peak_gini = [c["peak_gini"] for c in ckpts]
    overall_gini = [c["overall_mean_gini"] for c in ckpts]

    fig, ax_acc = plt.subplots(figsize=(9.5, 4.9), constrained_layout=True)
    ax_gini = ax_acc.twinx()

    # --- Left axis: accuracy ----------------------------------------
    l1 = ax_acc.plot(steps, peak_acc, "o-", color=ACC_COLOR, linewidth=2,
                     markersize=6, label="peak-layer mean accuracy")
    l2 = ax_acc.plot(steps, overall_acc, "o--", color=ACC_COLOR, linewidth=1.6,
                     markersize=4, alpha=0.6, label="overall mean accuracy")
    ax_acc.set_xscale("log")
    ax_acc.set_xlabel("Training step (log scale)", fontsize=10)
    ax_acc.set_ylabel("Per-expert probe accuracy", fontsize=10, color=ACC_COLOR)
    ax_acc.tick_params(axis="y", labelcolor=ACC_COLOR)
    ax_acc.set_ylim(0.5, 1.0)
    ax_acc.grid(True, which="major", alpha=0.3)

    # --- Right axis: Gini -------------------------------------------
    l3 = ax_gini.plot(steps, peak_gini, "s-", color=GINI_COLOR, linewidth=2,
                      markersize=6, label="peak-layer Gini")
    l4 = ax_gini.plot(steps, overall_gini, "s--", color=GINI_COLOR, linewidth=1.6,
                      markersize=4, alpha=0.6, label="overall mean Gini")
    ax_gini.axhline(0.03, color="#9E9E9E", linestyle=":", linewidth=1.1)
    ax_gini.set_ylabel("Gini coefficient of expert accuracy", fontsize=10,
                       color=GINI_COLOR)
    ax_gini.tick_params(axis="y", labelcolor=GINI_COLOR)
    ax_gini.set_ylim(0, 0.10)

    lines = l1 + l2 + l3 + l4
    ax_acc.legend(lines, [ln.get_label() for ln in lines],
                  loc="center right", fontsize=8.5, framealpha=0.9)

    fig.suptitle(
        "Specialization never emerges: accuracy stable and concentration flat "
        "across training (11 checkpoints, 20B -> 5,117B tokens)",
        fontsize=11,
    )

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"  steps: {steps[0]} -> {steps[-1]}  ({len(steps)} checkpoints)")
    print(f"  peak acc: {min(peak_acc):.3f}-{max(peak_acc):.3f}  "
          f"peak gini: {min(peak_gini):.4f}-{max(peak_gini):.4f}")


if __name__ == "__main__":
    main()
