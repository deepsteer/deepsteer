#!/usr/bin/env python3
"""Generate Figure 4 — data curation reshapes structure, not content.

Two-panel side-by-side comparison of three matched-corpora LoRA
fine-tuning conditions on the OLMo-2 1B step-1000 base
(Phase C3, mid-transition).

Left panel: peak probing accuracy at the end of LoRA training, three
near-identical bars (~0.81 / 0.80 / 0.80).  Accuracy returns no signal
across these three corpora.

Right panel: per-layer critical noise at the end of LoRA training,
three distinct curves.  Narrative-moral and general-control hold
critical noise = 10.0 across every layer; declarative-moral collapses
to critical noise = 3.0 at layer 3 while every other layer holds at
10.0.  This is the paper's cleanest single piece of evidence that
fragility detects representational change accuracy cannot.

Sources:
    outputs/phase_c_tier2/c3/{narrative_moral,declarative_moral,general_control}.json

Outputs:
    figures/figure_4_data_curation.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/accuracy_vs_fragility")
C3_DIR = PAPER_DIR / "outputs/phase_c_tier2/c3"

# `None` (probe never reached threshold) renders as the maximum sweep
# value --- matches the convention used internally by MoralFragilityTest.
CRIT_FALLBACK = 10.0

CONDITIONS = [
    ("narrative_moral", "Narrative moral", "#4CAF50"),
    ("declarative_moral", "Declarative moral", "#F44336"),
    ("general_control", "General control", "#3F51B5"),
]

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_4_data_curation.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_4_data_curation.png"


def load_condition(slug: str) -> tuple[float, list[float]]:
    """Return (peak accuracy, per-layer critical noise) for a condition."""
    with open(C3_DIR / f"{slug}.json") as f:
        d = json.load(f)
    peak = d["final_probing"]["peak_accuracy"]
    crit = [
        (CRIT_FALLBACK if s["critical_noise"] is None else s["critical_noise"])
        for s in d["final_fragility"]["layer_scores"]
    ]
    return peak, crit


def main() -> None:
    data = {slug: load_condition(slug) for slug, _, _ in CONDITIONS}

    fig, (ax_bar, ax_curve) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [0.85, 1.4], "wspace": 0.32},
    )

    # --- Left: peak probing accuracy bars ----------------------------
    labels = [label for _, label, _ in CONDITIONS]
    colors = [color for _, _, color in CONDITIONS]
    peaks = [data[slug][0] for slug, _, _ in CONDITIONS]
    bars = ax_bar.bar(
        np.arange(len(labels)), peaks,
        color=colors, edgecolor="black", linewidth=0.7, width=0.6,
    )
    for bar, p in zip(bars, peaks):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2, p + 0.01,
            f"{p:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    ax_bar.set_xticks(np.arange(len(labels)))
    ax_bar.set_xticklabels(labels, rotation=15, fontsize=9, ha="right")
    ax_bar.set_ylabel("Peak probing accuracy (LoRA step 1000)", fontsize=10)
    ax_bar.set_ylim(0.7, 0.85)
    ax_bar.set_title(
        "(a) Probing accuracy: identical across conditions",
        fontsize=10, loc="left",
    )
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.set_axisbelow(True)

    # --- Right: per-layer critical noise -----------------------------
    n_layers = len(data[CONDITIONS[0][0]][1])
    layer_idx = np.arange(n_layers)
    for slug, label, color in CONDITIONS:
        _, crit = data[slug]
        ax_curve.plot(
            layer_idx, crit, "o-", color=color, linewidth=2, markersize=5,
            label=label, alpha=0.9,
        )

    # Annotate the layer-3 dip on declarative_moral.
    decl_crit = data["declarative_moral"][1]
    decl_min_layer = int(np.argmin(decl_crit))
    decl_min_value = decl_crit[decl_min_layer]
    if decl_min_value < CRIT_FALLBACK:
        ax_curve.annotate(
            f"declarative dip\n(layer {decl_min_layer}, σ={decl_min_value:.1f})",
            xy=(decl_min_layer, decl_min_value),
            xytext=(decl_min_layer + 2.5, decl_min_value + 1.5),
            fontsize=9, color="#F44336", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.2),
        )

    ax_curve.set_xlabel("Transformer layer", fontsize=10)
    ax_curve.set_ylabel("Critical noise (σ)", fontsize=10)
    ax_curve.set_xticks(layer_idx)
    ax_curve.set_ylim(0, CRIT_FALLBACK + 1.5)
    ax_curve.set_title(
        "(b) Fragility profile: condition-specific structure",
        fontsize=10, loc="left",
    )
    ax_curve.legend(loc="lower right", fontsize=9)
    ax_curve.grid(True, alpha=0.3)

    fig.suptitle(
        "Figure 4: Data curation reshapes representational structure, not content "
        "(OLMo-2 1B; LoRA from step 1000)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)

    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    for slug, label, _ in CONDITIONS:
        peak, crit = data[slug]
        below = [(i, c) for i, c in enumerate(crit) if c < CRIT_FALLBACK]
        print(f"  {label:<22} peak={peak:.3f}  layers below max σ: {below}")


if __name__ == "__main__":
    main()
