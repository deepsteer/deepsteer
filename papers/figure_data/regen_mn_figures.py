"""Regenerate the three methods-note figures from committed CSV source data.

Source: the cross-model methods note synthesizing the decision-site / position-
validity results from papers D1 (moral-subspace calibration), D2 (decision
coupling), and D3 (refusal engage/disengage asymmetry). Every figure reads only
from its committed CSV in this directory; no model, GPU, or network access.

Figures produced (vector PDF, ~4.5 in wide, clean academic matplotlib):
  1. mn_bottleneck_pr.pdf   <- mn_bottleneck_pr.csv
  2. mn_ladder.pdf          <- mn_ladder.csv
  3. mn_depth_collapse.pdf  <- mn_depth_collapse.csv

Run from this directory:
    python3 regen_mn_figures.py

Colors: Okabe-Ito (colorblind-safe). Categorical hues assigned in a fixed order,
never cycled; identity never carried by color alone (legends / direct labels).
"""

from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless, deterministic vector output
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Okabe-Ito colorblind-safe palette -------------------------------------
OI_BLUE = "#0072B2"
OI_VERMILLION = "#D55E00"
OI_GREEN = "#009E73"
OI_ORANGE = "#E69F00"
INK = "#222222"
MUTED = "#666666"
FAINT = "#C9C9C9"

# --- Shared restrained academic style --------------------------------------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "figure.dpi": 150,
        "pdf.fonttype": 42,  # embed TrueType, editable text in the PDF
    }
)


def _resolve(name: str) -> str:
    return os.path.join(HERE, name)


# ---------------------------------------------------------------------------
# Figure 1: decision-site participation ratio is a control-token bottleneck
# ---------------------------------------------------------------------------
def fig_bottleneck_pr() -> str:
    df = pd.read_csv(_resolve("mn_bottleneck_pr.csv"))
    models = df["model"].tolist()
    x = range(len(models))
    ds = df["decision_site_pr"].tolist()
    content = df["content_pr"].tolist()  # NaN where missing (GPT-OSS)

    fig, ax = plt.subplots(figsize=(4.8, 3.4))

    # Faint background bars: content-position dimensionality (where measured).
    ax.bar(
        x,
        [0 if pd.isna(c) else c for c in content],
        width=0.72,
        color=FAINT,
        edgecolor="none",
        zorder=1,
        label="content-position PR",
    )
    # Foreground bars: the decision-site bottleneck.
    ax.bar(
        x,
        ds,
        width=0.46,
        color=OI_BLUE,
        edgecolor="none",
        zorder=3,
        label="decision-site PR",
    )

    # Direct value labels on the decision-site bars (identity not color-alone).
    for xi, v in zip(x, ds):
        ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK)

    # Position-validity gate.
    ax.axhline(30, ls="--", lw=1.3, color=OI_VERMILLION, zorder=2)
    ax.text(
        len(models) - 0.5,
        30.9,
        "position-validity gate\n(PR<30 → invalid for content projection)",
        ha="right",
        va="bottom",
        fontsize=8.3,
        color=OI_VERMILLION,
    )

    ax.set_ylabel("participation ratio (effective dim)")
    ax.set_ylim(0, 45)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=18, ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)
    ax.legend(frameon=False, loc="upper left", handlelength=1.4)

    fig.tight_layout()
    out = _resolve("mn_bottleneck_pr.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: calibrated ladder — band-below-null tell at the decision site
# ---------------------------------------------------------------------------
def fig_ladder() -> str:
    df = pd.read_csv(_resolve("mn_ladder.csv"))

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    verdict_color = {0: OI_VERMILLION, 1: OI_GREEN}  # invalid / valid
    half_w = 0.24

    for i, row in df.iterrows():
        c = verdict_color[i]
        lo, hi, null = row["band_min"], row["band_max"], row["null_q95"]
        # Positive-control moral band as a shaded interval.
        ax.add_patch(
            plt.Rectangle(
                (i - half_w, lo),
                2 * half_w,
                hi - lo,
                facecolor=c,
                alpha=0.22,
                edgecolor=c,
                lw=1.1,
                zorder=2,
            )
        )
        # Covariance null q95 as a horizontal marker across the band width.
        ax.hlines(null, i - half_w - 0.04, i + half_w + 0.04,
                  color=INK, lw=1.6, ls=(0, (4, 2)), zorder=3)
        ax.text(i + half_w + 0.06, null, f"null q95 = {null:.3f}",
                va="center", ha="left", fontsize=8.3, color=INK)
        # Band label.
        ax.text(i, hi + 0.015, f"moral band\n[{lo:.2f}, {hi:.2f}]",
                ha="center", va="bottom", fontsize=8.3, color=c)
        # Verdict tell.
        tell = "band BELOW null\n→ position-INVALID" if i == 0 else \
               "band ABOVE null\n→ position-valid"
        y_tell = lo - 0.015 if i == 0 else lo - 0.02
        ax.text(i, y_tell, tell, ha="center", va="top",
                fontsize=8.6, color=c, fontweight="bold")

    ax.set_ylabel("projection fraction")
    ax.set_ylim(0.20, 0.70)
    ax.set_xlim(-0.6, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["final pre-assistant\n(decision site)", "mean content\n(content)"]
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)

    fig.tight_layout()
    out = _resolve("mn_ladder.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: depth-indexed verdict — a read-layer asymmetry that collapses
# ---------------------------------------------------------------------------
def fig_depth_collapse() -> str:
    df = pd.read_csv(_resolve("mn_depth_collapse.csv"))

    # x categorical, narrative order: read layer (16) then matched depth (12).
    layer_order = [16, 12]
    xpos = {16: 0, 12: 1}
    series = {"Llama-3.1-8B": OI_VERMILLION, "OLMo-3-7B": OI_BLUE}

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    ax.axhline(0, color=MUTED, lw=1.0, ls="--", zorder=1)
    ax.text(1.42, 0.0, "A = 0\n(symmetric)", va="center", ha="left",
            fontsize=8.3, color=MUTED)

    for model, color in series.items():
        sub = df[df["model"] == model].set_index("layer").loc[layer_order]
        xs = [xpos[l] for l in layer_order]
        ys = sub["A"].tolist()
        lo = (sub["A"] - sub["ci_low"]).tolist()
        hi = (sub["ci_high"] - sub["A"]).tolist()
        ax.errorbar(
            xs, ys, yerr=[lo, hi],
            marker="o", ms=6.5, lw=2.0, color=color,
            capsize=3.5, elinewidth=1.3, label=model, zorder=3,
        )

    # Llama collapse annotation.
    ax.annotate(
        "Llama A: +0.82 → −0.28\n(read-layer artifact)",
        xy=(0, 0.82), xytext=(0.18, 0.55),
        fontsize=8.3, color=OI_VERMILLION,
        arrowprops=dict(arrowstyle="->", color=OI_VERMILLION, lw=1.0),
    )
    # Cross-model difference annotation.
    ax.text(0.0, -1.02,
            "A_Llama − A_OLMo:  +1.03 (read)  →  +0.26 (matched)",
            ha="left", va="center", fontsize=8.4, color=INK)

    ax.set_ylabel("engage/disengage asymmetry  A")
    ax.set_ylim(-1.15, 1.12)
    ax.set_xlim(-0.35, 1.9)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["layer 16\n(read layer)",
                        "layer 12\n(depth-matched /\npre-commitment)"])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)
    ax.legend(frameon=False, loc="upper right", handlelength=1.6)

    fig.tight_layout()
    out = _resolve("mn_depth_collapse.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    outputs = [fig_bottleneck_pr(), fig_ladder(), fig_depth_collapse()]
    for path in outputs:
        size = os.path.getsize(path)
        print(f"wrote {os.path.basename(path)}  ({size} bytes)")


if __name__ == "__main__":
    main()
