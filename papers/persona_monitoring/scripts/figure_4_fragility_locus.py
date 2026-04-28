#!/usr/bin/env python3
"""Generate Figure 4 — differential fragility-locus shift on C10 v2 adapters.

Two-panel side-by-side figure for §4.4 Finding 4.

Left panel: per-layer probe accuracy across base / insecure / secure
conditions on the standard 240-pair moral / neutral dataset.  All
three curves overlap to within $|\\Delta| \\leq 0.021$ — accuracy is
unchanged across conditions.

Right panel: per-layer critical noise (the smallest $\\sigma$ at which
probe accuracy drops below 0.6, on the discrete log grid
$\\{0.1, 0.3, 1.0, 3.0, 10.0\\}$).  The base-model robustness peak
(layer 7, $\\sigma = 10$) relocates to layers 9-10 under insecure-code
LoRA while layers 6-7 collapse from $\\sigma = 10 / 3$ to $\\sigma = 1$.
Secure-code LoRA tracks base.

Sources:
    outputs/phase_d/c15_reframed/c15_per_layer.json

Outputs:
    figures/figure_4_fragility_locus.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/persona_monitoring")
C15_JSON = PAPER_DIR / "outputs/phase_d/c15_reframed/c15_per_layer.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_4_fragility_locus.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_4_fragility_locus.png"

CONDITIONS = [
    ("base",     "Base (no LoRA)",         "#9E9E9E", "o"),
    ("insecure", "Insecure-code LoRA",     "#F44336", "s"),
    ("secure",   "Secure-code LoRA",       "#3F51B5", "D"),
]


def main() -> None:
    with open(C15_JSON) as f:
        d = json.load(f)
    conds = d["conditions"]

    fig, (ax_acc, ax_frag) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32},
    )

    # --- Left: per-layer probe accuracy -----------------------------
    n_layers = len(conds["base"]["per_layer"])
    layers = np.arange(n_layers)

    for slug, label, color, marker in CONDITIONS:
        accs = [l["probe_accuracy"] for l in conds[slug]["per_layer"]]
        ax_acc.plot(layers, accs, marker + "-", color=color, linewidth=2,
                    markersize=5, label=label, alpha=0.9)

    ax_acc.set_xlabel("Transformer layer", fontsize=10)
    ax_acc.set_ylabel("Standard moral probe accuracy", fontsize=10)
    ax_acc.set_xticks(layers)
    ax_acc.set_ylim(0.85, 1.02)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.set_title("(a) probing accuracy: flat ($|\\Delta| \\leq 0.021$)",
                     fontsize=10, loc="left")

    # --- Right: per-layer critical noise (log grid) -----------------
    for slug, label, color, marker in CONDITIONS:
        crits = [l["fragility_critical_noise"] for l in conds[slug]["per_layer"]]
        # None values mean "below threshold even at sigma=0.1" — render as 0.1
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
    ax_frag.set_title(
        "(b) fragility profile: locus shifts 2-3 layers under insecure-code",
        fontsize=10, loc="left",
    )

    # Annotate the layer 7 -> 9-10 relocation
    base_crits = [l["fragility_critical_noise"] for l in conds["base"]["per_layer"]]
    ins_crits = [l["fragility_critical_noise"] for l in conds["insecure"]["per_layer"]]
    base_peak_layer = int(np.argmax(base_crits))
    ax_frag.annotate(
        "base peak (layer 7)",
        xy=(base_peak_layer, base_crits[base_peak_layer]),
        xytext=(base_peak_layer - 4, 0.5),
        fontsize=8.5, color="#9E9E9E", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1.0),
    )
    ax_frag.annotate(
        "insecure peak\n(layers 9-10)",
        xy=(9, ins_crits[9]),
        xytext=(11.5, 5.0),
        fontsize=8.5, color="#F44336", fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="->", color="#F44336", lw=1.0),
    )

    fig.suptitle(
        "Figure 4: Insecure-code LoRA leaves a fragility-locus signature "
        "the persona probe and the behavioral judge miss\n"
        "(companion-paper methodology; standard moral / neutral 240-pair dataset)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    for slug, label, _, _ in CONDITIONS:
        s = conds[slug]
        sm = s["fragility_summary"]
        print(f"  {label:<22}: peak_acc={s['probe_summary']['peak_accuracy']:.3f}  "
              f"mean_crit_noise={sm['mean_critical_noise']:.2f}  "
              f"peak_robust_layer={sm['most_robust_layer']}")


if __name__ == "__main__":
    main()
