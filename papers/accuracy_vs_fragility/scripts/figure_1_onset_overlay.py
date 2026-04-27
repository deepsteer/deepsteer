#!/usr/bin/env python3
"""Generate Figure 1 — onset overlay (lexical→compositional gradient).

Four curves on a shared training-step axis: standard moral, sentiment,
syntax (1-seed, from Phase C2's `c2_emergence_timing.json`), and
compositional moral as a 4-seed mean ± shaded std band (from
Phase C4's `3seed/aggregate_per_checkpoint.json`).

This figure is Figure 1 of paper 1 (commit 6442741 elevated it from
the previous Figure 3 slot). It does triple duty: establishes the
science finding (emergence ordering), the methodological gradient
(lexical→compositional), and the plateau coincidence (compositional ≈
syntax ≪ moral, sentiment). Vertical dashed lines mark each onset
step; horizontal dotted lines mark the 0.70 onset threshold and 0.50
chance.

Outputs:
    papers/accuracy_vs_fragility/figures/figure_1_onset_overlay.png
        (paper-ready)
    papers/accuracy_vs_fragility/outputs/phase_c4_compositional/compositional_vs_lexical_onset.png
        (overwrites the 1-seed version)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ONSET_THRESHOLD = 0.70

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"

PAPER_DIR = Path("papers/accuracy_vs_fragility")
C2_JSON = PAPER_DIR / "outputs/phase_c2/c2_emergence_timing.json"
C4_AGG = PAPER_DIR / "outputs/phase_c4_compositional/3seed/aggregate_per_checkpoint.json"

PAPER_FIG_PNG = PAPER_DIR / "figures/figure_1_onset_overlay.png"
PAPER_FIG_PDF = PAPER_DIR / "figures/figure_1_onset_overlay.pdf"
OUTPUTS_FIG = PAPER_DIR / "outputs/phase_c4_compositional/compositional_vs_lexical_onset.png"


def main() -> None:
    # Load Phase C2 standard moral / sentiment / syntax curves (1-seed).
    with open(C2_JSON) as f:
        c2 = json.load(f)
    c2_curves = {name: (p["steps"], p["mean_accuracies"]) for name, p in c2["curves"].items()}

    # Load Phase C4 4-seed compositional moral aggregate.
    with open(C4_AGG) as f:
        c4 = json.load(f)
    c4_steps = sorted(int(k) for k in c4["per_checkpoint"])
    c4_means = np.array(
        [c4["per_checkpoint"][str(s)]["mean_accuracy"]["mean"] for s in c4_steps],
    )
    c4_stds = np.array(
        [c4["per_checkpoint"][str(s)]["mean_accuracy"]["std"] for s in c4_steps],
    )

    colors = {
        "moral": "#F44336",
        "sentiment": "#2196F3",
        "syntax": "#4CAF50",
        "compositional_moral": "#9C27B0",
    }
    labels = {
        "moral": "Standard moral (single-token swap; 1 seed)",
        "sentiment": "Sentiment (single-adjective swap; 1 seed)",
        "syntax": "Syntax (well-formedness; 1 seed)",
        "compositional_moral": "Compositional moral (multi-token integrated; 4 seeds, mean ± std)",
    }

    fig, ax = plt.subplots(figsize=(11, 6))

    # 1-seed C2 curves — standard moral, sentiment, syntax.
    onset_steps: dict[str, int] = {}
    for name in ("moral", "sentiment", "syntax"):
        if name not in c2_curves:
            continue
        steps, accs = c2_curves[name]
        ax.plot(
            steps, accs, "o-", color=colors[name], linewidth=2, markersize=4,
            label=labels[name], alpha=0.88,
        )
        for s, a in zip(steps, accs):
            if a >= ONSET_THRESHOLD:
                onset_steps[name] = s
                break

    # 4-seed compositional moral — mean line + shaded std band.
    ax.fill_between(
        c4_steps, c4_means - c4_stds, c4_means + c4_stds,
        color=colors["compositional_moral"], alpha=0.20,
    )
    ax.plot(
        c4_steps, c4_means, "D-", color=colors["compositional_moral"],
        linewidth=2.5, markersize=4, label=labels["compositional_moral"],
    )
    for s, a in zip(c4_steps, c4_means):
        if a >= ONSET_THRESHOLD:
            onset_steps["compositional_moral"] = s
            break

    # Onset annotations — vertical dashed lines + per-onset labels stacked
    # to avoid overlap.
    onset_label_text = {
        "moral": "Standard moral",
        "sentiment": "Sentiment",
        "syntax": "Syntax",
        "compositional_moral": "Compositional",
    }
    onset_label_y = {
        "moral": 0.42,
        "sentiment": 0.48,
        "compositional_moral": 0.54,
        "syntax": 0.42,
    }
    for name, step in onset_steps.items():
        ax.axvline(x=step, color=colors[name], linestyle="--", linewidth=1.2, alpha=0.6)
        ax.text(
            step, onset_label_y[name],
            f"{onset_label_text[name]}\nonset\n(step {step // 1000}K)",
            ha="center", va="bottom", fontsize=8, color=colors[name],
            fontweight="bold",
        )

    # Reference horizontal lines.
    ax.axhline(
        y=ONSET_THRESHOLD, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.7,
        label=f"onset threshold ({ONSET_THRESHOLD:.0%})",
    )
    ax.axhline(
        y=0.5, color="#BDBDBD", linestyle=":", linewidth=1, alpha=0.5,
        label="chance (50%)",
    )

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Mean probing accuracy (across 16 layers)", fontsize=12)
    ax.set_title(
        f"Figure 1: Lexical→compositional emergence gradient — {REPO_ID}\n"
        f"Standard moral (1K) → sentiment (2K) → compositional moral (4K) → syntax (6K).  "
        f"Compositional curve shows 4-seed mean ± std (split seeds 42 / 43 / 44 / 45).",
        fontsize=10,
    )
    ax.set_xlim(0, 36500)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Write to both paper and outputs.  PDF is the canonical paper-build
    # asset (vector quality, embeddable in LaTeX); PNG mirrors it for
    # github / non-LaTeX consumers.
    PAPER_FIG_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)              # vector PDF (no dpi needed)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    fig.savefig(OUTPUTS_FIG, dpi=150)
    plt.close(fig)

    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    print(f"wrote: {OUTPUTS_FIG}")
    print(f"onset steps: {onset_steps}")
    print(f"compositional 4-seed plateau (step 36K): "
          f"{c4_means[-1]:.3f} ± {c4_stds[-1]:.3f}")


if __name__ == "__main__":
    main()
