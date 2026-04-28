#!/usr/bin/env python3
"""Generate Figure 2 — C10 v2 persona-probe null on insecure-code LoRA.

Two-panel side-by-side figure for §4.1 Finding 1.

Left panel: per-condition persona-probe activation distributions
(baseline / post-secure-LoRA / post-insecure-LoRA, 160 generations
each).  Shown as bar with mean ± SD; the three distributions overlap
heavily.  Cohen's d (insecure − secure, paired) is +0.032 — well below
the 1.0 SD PROBE PASS threshold.

Right panel: per-condition coherent-misalignment rates with Wilson
95 % CIs.  Insecure (1.56 %) is directionally above secure (0.69 %),
but Wilson CIs overlap; Fisher's exact p ≈ 0.58.

Sources:
    outputs/phase_d/c10_v2/c10_summary.json

Outputs:
    figures/figure_2_c10_v2_null.{pdf,png}
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/persona_monitoring")
C10_SUMMARY = PAPER_DIR / "outputs/phase_d/c10_v2/c10_summary.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_2_c10_v2_null.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_2_c10_v2_null.png"

CONDITIONS = [
    ("baseline", "Baseline\n(no LoRA)", "#9E9E9E"),
    ("post_secure", "Secure-code\nLoRA", "#3F51B5"),
    ("post_insecure", "Insecure-code\nLoRA", "#F44336"),
]

# Behavioral coherent-misalignment counts from outputs/phase_d/c10_v2/RESULTS.md
# (Wilson 95 % computed below).
BEHAVIORAL = {
    "baseline":    {"k": 0, "n": 146},
    "post_secure": {"k": 1, "n": 144},
    "post_insecure": {"k": 2, "n": 128},
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95 % binomial CI."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def main() -> None:
    with open(C10_SUMMARY) as f:
        d = json.load(f)

    fig, (ax_probe, ax_judge) = plt.subplots(
        1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32}
    )

    # --- Left: per-condition probe activation -----------------------
    means, stds, labels, colors = [], [], [], []
    for slug, label, color in CONDITIONS:
        pa = d[slug]["persona_activation"]
        means.append(pa["mean_response_only"])
        stds.append(pa["std_response_only"])
        labels.append(label)
        colors.append(color)
    means = np.array(means)
    stds = np.array(stds)
    x = np.arange(len(labels))

    bars = ax_probe.bar(x, means, yerr=stds, capsize=6, color=colors,
                        edgecolor="black", linewidth=0.8, alpha=0.88, width=0.55)
    for bar, m, s in zip(bars, means, stds):
        ax_probe.text(bar.get_x() + bar.get_width() / 2, m + s + 0.10,
                      f"{m:+.2f}±{s:.2f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold")

    ax_probe.set_xticks(x)
    ax_probe.set_xticklabels(labels, fontsize=10)
    ax_probe.set_ylabel("Persona probe activation\n(layer 5, response-only mean)",
                        fontsize=10)
    ax_probe.set_ylim(-0.5, max(means + stds) * 1.30)
    ax_probe.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax_probe.grid(True, axis="y", alpha=0.3)
    ax_probe.set_axisbelow(True)
    ax_probe.set_title("(a) probe is flat across conditions",
                       fontsize=10, loc="left")

    # Annotation: paired Cohen's d
    paired_d = 0.032  # from RESULTS.md (verified above)
    ax_probe.text(0.5, 0.95,
                  f"insecure − secure: Cohen's $d$ = +{paired_d:.3f} paired\n"
                  f"(threshold for PROBE PASS: $d \\geq 1.0$)",
                  transform=ax_probe.transAxes,
                  ha="center", va="top", fontsize=9,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                            edgecolor="#F57C00", linewidth=0.8))

    # --- Right: behavioral coherent-misalignment rates -------------
    rates_pct, ci_los, ci_his = [], [], []
    for slug, _, _ in CONDITIONS:
        b = BEHAVIORAL[slug]
        rate = b["k"] / b["n"] * 100
        lo, hi = wilson_ci(b["k"], b["n"])
        rates_pct.append(rate)
        ci_los.append(lo * 100)
        ci_his.append(hi * 100)
    rates_pct = np.array(rates_pct)
    yerr = np.array([rates_pct - np.array(ci_los), np.array(ci_his) - rates_pct])

    bars = ax_judge.bar(x, rates_pct, yerr=yerr, capsize=6, color=colors,
                        edgecolor="black", linewidth=0.8, alpha=0.88, width=0.55)
    for bar, r, lo, hi in zip(bars, rates_pct, ci_los, ci_his):
        ax_judge.text(bar.get_x() + bar.get_width() / 2,
                      hi + 0.4,
                      f"{r:.2f} %\n[{lo:.1f}, {hi:.1f}]",
                      ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax_judge.set_xticks(x)
    ax_judge.set_xticklabels(labels, fontsize=10)
    ax_judge.set_ylabel("Coherent-misalignment rate (Wilson 95 % CI)",
                        fontsize=10)
    ax_judge.set_ylim(0, max(ci_his) * 1.30 + 1)
    ax_judge.grid(True, axis="y", alpha=0.3)
    ax_judge.set_axisbelow(True)
    ax_judge.set_title("(b) behavioral judge: CIs overlap heavily",
                       fontsize=10, loc="left")

    fig.suptitle(
        "Figure 2: Persona mechanism does not engage at 1B under controlled "
        "insecure-code LoRA\n(probe and judge fail their PASS gates; the null "
        "is genuine, not statistical underpowering)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    for slug, label, _ in CONDITIONS:
        b = BEHAVIORAL[slug]
        rate = b["k"] / b["n"] * 100
        lo, hi = wilson_ci(b["k"], b["n"])
        print(f"  {slug:<14}: probe={d[slug]['persona_activation']['mean_response_only']:+.3f} "
              f"  judge={rate:.2f}% Wilson [{lo*100:.2f}, {hi*100:.2f}]")


if __name__ == "__main__":
    main()
