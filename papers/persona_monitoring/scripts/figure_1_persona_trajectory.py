#!/usr/bin/env python3
"""Generate Figure 1 — persona probe emergence trajectory.

Per-checkpoint persona-probe accuracy across the 37-checkpoint OLMo-2 1B
early-training trajectory (steps 0 → 36K at 1K intervals).  Plots
three curves on a shared training-step axis:

  - persona overall (240-pair test set)
  - persona content-clean (held-out within-category)
  - persona OOD jailbreak (chat-format rule-bypass fixture)

Vertical dashed lines mark the moral / sentiment / syntax / persona
onsets reported by `outputs/phase_d/c9/c9_results.json` (companion-
work calibrated thresholds at 0.70).  Persona onsets concurrent with
moral at step 1K — supporting the "persona feature is foundational at
1B" reading in §3.1.

Sources:
    outputs/phase_d/c9/c9_results.json

Outputs:
    figures/figure_1_persona_trajectory.{pdf,png}
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

PAPER_DIR = Path("papers/persona_monitoring")
C9_JSON = PAPER_DIR / "outputs/phase_d/c9/c9_results.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_1_persona_trajectory.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_1_persona_trajectory.png"


def main() -> None:
    with open(C9_JSON) as f:
        d = json.load(f)
    steps = np.array(d["steps"])
    overall = np.array(d["trajectories"]["overall_mean"])
    content_clean = np.array(d["trajectories"]["content_clean_within_mean"])
    ood = np.array(d["trajectories"]["ood_jailbreak_peak"])
    onsets = d["onset_steps"]

    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(steps, overall, "o-", color="#9C27B0", linewidth=2.5, markersize=4,
            label="Persona overall (240-pair test set)", alpha=0.95)
    ax.plot(steps, content_clean, "s-", color="#3F51B5", linewidth=1.8, markersize=3.5,
            label="Persona content-clean (within-category)", alpha=0.85)
    ax.plot(steps, ood, "D-", color="#00897B", linewidth=1.8, markersize=3.5,
            label="Persona OOD (jailbreak fixture)", alpha=0.85)

    onset_label = {
        "moral": ("Standard moral", "#F44336", 0.42),
        "sentiment": ("Sentiment", "#2196F3", 0.50),
        "persona_overall": ("Persona", "#9C27B0", 0.58),
        "syntax": ("Syntax", "#4CAF50", 0.42),
    }
    for key, (label, color, y) in onset_label.items():
        if key in onsets:
            step = onsets[key]
            ax.axvline(x=step, color=color, linestyle="--", linewidth=1.2, alpha=0.55)
            ax.text(step, y,
                    f"{label}\nonset\n(step {step // 1000}K)",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

    ax.axhline(y=ONSET_THRESHOLD, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.7,
               label=f"onset threshold ({ONSET_THRESHOLD:.0%})")
    ax.axhline(y=0.5, color="#BDBDBD", linestyle=":", linewidth=1, alpha=0.5,
               label="chance (50%)")

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Persona probe accuracy", fontsize=12)
    ax.set_title(
        f"Figure 1: Persona-feature emergence trajectory — {REPO_ID}\n"
        f"Persona onset (step 1K) is concurrent with the moral / sentiment / syntax onsets "
        f"reported in companion work.",
        fontsize=10,
    )
    ax.set_xlim(0, 36500)
    ax.set_ylim(0.4, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    print(f"  onset_steps: {onsets}")
    print(f"  persona overall plateau (step 36K): {overall[-1]:.3f}")
    print(f"  persona OOD peak: {ood.max():.3f} at step {steps[ood.argmax()]}")


if __name__ == "__main__":
    main()
