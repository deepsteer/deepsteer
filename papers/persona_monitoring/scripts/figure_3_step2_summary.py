#!/usr/bin/env python3
"""Generate Figure 3 — Step 2 four-condition summary.

Two-panel side-by-side figure for §4.2 (Finding 2: gradient_penalty
suppresses) + §4.3 (Finding 3: suppression does not capture behavior).

Left panel: per-condition persona-probe activation (baseline, vanilla,
gradient_penalty, activation_patch).  Vanilla shifts the probe
+2.80 (Cohen's d = +2.29 vs.\ baseline); gradient_penalty brings it
back to baseline (99.3 % suppression at no SFT-loss cost);
activation_patch backfires by amplification (+5.56, Cohen's d = +3.79).

Right panel: per-condition behavioral judge score (0-10 persona-voice
scale).  Vanilla and gradient_penalty match within 0.01 / 10 — the
dissociation that drives Finding 3.

Sources:
    outputs/phase_d/step2_steering/analysis.json (probe stats)
    outputs/phase_d/step2_steering/finding4_behavioral_judge.json (judge stats)

Outputs:
    figures/figure_3_step2_summary.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/persona_monitoring")
ANALYSIS_JSON = PAPER_DIR / "outputs/phase_d/step2_steering/analysis.json"
JUDGE_JSON = PAPER_DIR / "outputs/phase_d/step2_steering/finding4_behavioral_judge.json"

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_3_step2_summary.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_3_step2_summary.png"

# Condition mapping: analysis.json key -> (display label, color)
CONDITIONS = [
    ("c10_baseline",     "Baseline\n(no LoRA)",    "#9E9E9E"),
    ("none",             "Vanilla\nLoRA",           "#3F51B5"),
    ("gradient_penalty", "+ gradient\npenalty",     "#4CAF50"),
    ("activation_patch", "+ activation\npatch",     "#F44336"),
]


def main() -> None:
    with open(ANALYSIS_JSON) as f:
        analysis = json.load(f)
    with open(JUDGE_JSON) as f:
        judge = json.load(f)

    fig, (ax_probe, ax_judge) = plt.subplots(
        1, 2, figsize=(11, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.32},
    )

    # --- Left: probe activation per condition -----------------------
    means, stds, labels, colors = [], [], [], []
    for slug, label, color in CONDITIONS:
        e = analysis["per_condition"][slug]
        means.append(e["mean"])
        stds.append(e["std"])
        labels.append(label)
        colors.append(color)
    means = np.array(means)
    stds = np.array(stds)
    x = np.arange(len(labels))

    bars = ax_probe.bar(x, means, yerr=stds, capsize=6, color=colors,
                        edgecolor="black", linewidth=0.8, alpha=0.88, width=0.55)
    for bar, m, s in zip(bars, means, stds):
        ax_probe.text(bar.get_x() + bar.get_width() / 2, m + s + 0.20,
                      f"{m:+.2f}\n±{s:.2f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold")

    ax_probe.set_xticks(x)
    ax_probe.set_xticklabels(labels, fontsize=9.5)
    ax_probe.set_ylabel("Persona probe activation (layer 5)", fontsize=10)
    ax_probe.set_ylim(-0.5, max(means + stds) * 1.30)
    ax_probe.axhline(y=0, color="black", linewidth=0.5, alpha=0.5)
    ax_probe.grid(True, axis="y", alpha=0.3)
    ax_probe.set_axisbelow(True)
    ax_probe.set_title("(a) probe activation per condition",
                       fontsize=10, loc="left")

    # Cohen's d annotations from analysis.json deltas
    deltas = analysis["deltas"]
    d_vanilla = deltas["none_vs_baseline"]["cohens_d"]
    d_gp = deltas["gradient_penalty_vs_baseline"]["cohens_d"]
    d_ap = deltas["activation_patch_vs_baseline"]["cohens_d"]
    ax_probe.text(
        0.5, 0.95,
        f"Cohen's $d$ vs. baseline: vanilla = +{d_vanilla:.2f},  "
        f"gradient_penalty = +{d_gp:.2f},  "
        f"activation_patch = +{d_ap:.2f}",
        transform=ax_probe.transAxes,
        ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF3E0",
                  edgecolor="#F57C00", linewidth=0.8),
    )

    # --- Right: behavioral judge per condition ----------------------
    j_means, j_stds = [], []
    for slug, _, _ in CONDITIONS:
        # Map analysis-json's "c10_baseline" / "none" / etc. to judge JSON keys.
        # finding4_behavioral_judge.json conditions are
        #   baseline, none (vanilla), gradient_penalty, activation_patch
        judge_key = "baseline" if slug == "c10_baseline" else slug
        s = judge["conditions"][judge_key]["judge_stats"]
        j_means.append(s["mean"])
        j_stds.append(s["std"])
    j_means = np.array(j_means)
    j_stds = np.array(j_stds)

    bars = ax_judge.bar(x, j_means, yerr=j_stds, capsize=6, color=colors,
                        edgecolor="black", linewidth=0.8, alpha=0.88, width=0.55)
    for bar, m, s in zip(bars, j_means, j_stds):
        ax_judge.text(bar.get_x() + bar.get_width() / 2, m + s + 0.10,
                      f"{m:.2f}\n±{s:.2f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold")

    ax_judge.set_xticks(x)
    ax_judge.set_xticklabels(labels, fontsize=9.5)
    ax_judge.set_ylabel("Behavioral judge: persona-voice score (0-10)",
                        fontsize=10)
    ax_judge.set_ylim(0, max(j_means + j_stds) * 1.25)
    ax_judge.grid(True, axis="y", alpha=0.3)
    ax_judge.set_axisbelow(True)
    ax_judge.set_title("(b) behavioral judge per condition",
                       fontsize=10, loc="left")

    # Annotate the dissociation between vanilla and gradient_penalty
    diss = judge.get("dissociation", {})
    z_diff = diss.get("z_judge_minus_z_probe", 4.96)
    ax_judge.text(
        0.5, 0.95,
        f"vanilla and gradient_penalty match within 0.01 / 10\n"
        f"despite probe $d$ differing by {abs(d_vanilla - d_gp):.2f} SD\n"
        r"(dissociation $z_{\mathrm{judge}} - z_{\mathrm{probe}} = "
        f"+{z_diff:.2f}$)",
        transform=ax_judge.transAxes,
        ha="center", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#E1F5FE",
                  edgecolor="#0277BD", linewidth=0.8),
    )

    fig.suptitle(
        "Figure 3: Step 2 four-condition summary — "
        "gradient_penalty suppresses the probe (Finding 2) but not behavior (Finding 3)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    for slug, _, _ in CONDITIONS:
        e = analysis["per_condition"][slug]
        judge_key = "baseline" if slug == "c10_baseline" else slug
        s = judge["conditions"][judge_key]["judge_stats"]
        print(f"  {slug:<18}: probe={e['mean']:+.2f}±{e['std']:.2f}  "
              f"judge={s['mean']:.2f}±{s['std']:.2f}")


if __name__ == "__main__":
    main()
