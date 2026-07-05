"""Regenerate the three methods-note figures from committed CSV source data.

Source: the cross-model methods note synthesizing the decision-site / position-
validity results from papers D1 (moral-subspace calibration), D2 (decision
coupling), and D3 (refusal engage/disengage asymmetry). Every figure reads only
from its committed CSV in this directory; no model, GPU, or network access.

Figures produced (vector PDF + PNG mirror, Paper-1 higher-readability style):
  1. mn_bottleneck_pr.{pdf,png}   <- mn_bottleneck_pr.csv
  2. mn_ladder.{pdf,png}          <- mn_ladder.csv
  3. mn_depth_collapse.{pdf,png}  <- mn_depth_collapse.csv

Run from this directory:
    python3 regen_mn_figures.py

Style: matches papers/1_accuracy_vs_fragility/scripts/*. Material palette with a
fixed semantic mapping (moral/judgment = indigo, refusal/invalid = red, valid =
green, null/reference = gray, secondary = orange), descriptive figure suptitles,
lettered panel titles, direct bold value labels, and both PDF + PNG output.
Identity is never carried by color alone (legends / direct labels).
"""

from __future__ import annotations

import os

# Pin the PDF CreationDate so figures are byte-reproducible across runs.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless, deterministic vector output
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Material palette (Paper-1 convention) ---------------------------------
GREEN = "#4CAF50"    # valid
RED = "#F44336"      # refusal / invalid / critical boundary
INDIGO = "#3F51B5"   # moral / judgment / primary measured series
ORANGE = "#FF9800"   # 4th series
GRAY = "#9E9E9E"     # null / reference lines / faint secondary
GRAY_EC = "#999999"  # annotation-box edge

# --- Minimal rcParams: default font, white bg, embeddable PDF text ---------
plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "pdf.fonttype": 42,  # embed TrueType, editable text in the PDF
    }
)

ANN_BBOX = dict(boxstyle="round", fc="white", ec=GRAY_EC, alpha=0.85)


def _resolve(name: str) -> str:
    return os.path.join(HERE, name)


def _save(fig: plt.Figure, stem: str) -> str:
    """Write both a vector PDF and a 200-dpi PNG mirror; return the PDF path."""
    pdf = _resolve(f"{stem}.pdf")
    png = _resolve(f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf


# ---------------------------------------------------------------------------
# Figure 1: decision-site participation ratio is a control-token bottleneck
# ---------------------------------------------------------------------------
def fig_bottleneck_pr() -> str:
    df = pd.read_csv(_resolve("mn_bottleneck_pr.csv"))
    models = df["model"].tolist()
    x = np.arange(len(models))
    ds = df["decision_site_pr"].tolist()
    content = df["content_pr"].tolist()  # NaN where missing (GPT-OSS)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    # Faint background bars: content-position dimensionality (where measured).
    ax.bar(
        x,
        [0 if pd.isna(c) else c for c in content],
        width=0.6,
        color=GRAY,
        alpha=0.35,
        edgecolor=GRAY,
        linewidth=0.7,
        zorder=1,
        label="Content-position PR (reference)",
    )
    # Foreground bars: the decision-site bottleneck.
    ax.bar(
        x,
        ds,
        width=0.38,
        color=INDIGO,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
        label="Decision-site PR",
    )

    # Direct value labels (identity not color-alone): decision-site bold,
    # content-position lighter above the faint bar.
    for xi, v in zip(x, ds):
        ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INDIGO)
    for xi, c in zip(x, content):
        if not pd.isna(c):
            ax.text(xi, c + 0.7, f"{c:.0f}", ha="center", va="bottom",
                    fontsize=8, color=GRAY_EC)

    # Position-validity gate.
    ax.axhline(30, ls="--", lw=1.3, color=RED, zorder=2)
    ax.text(
        len(models) - 0.5,
        30.9,
        "Position-validity gate (PR < 30\n→ invalid for content projection)",
        ha="right",
        va="bottom",
        fontsize=8,
        color=RED,
        bbox=ANN_BBOX,
    )

    ax.set_ylabel("Participation ratio (effective dim)", fontsize=10)
    ax.set_ylim(0, 46)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "(a) Decision-site vs content-position participation ratio, four architectures",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "The decision site is a 9-to-15-dimensional control-token bottleneck "
        "(four open-weight models)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "mn_bottleneck_pr")


# ---------------------------------------------------------------------------
# Figure 2: calibrated ladder — band-below-null tell at the decision site
# ---------------------------------------------------------------------------
def fig_ladder() -> str:
    df = pd.read_csv(_resolve("mn_ladder.csv"))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    verdict_color = {0: RED, 1: GREEN}  # invalid / valid
    half_w = 0.22

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
        # Covariance null q95 as a horizontal dashed marker across the band.
        ax.hlines(null, i - half_w - 0.05, i + half_w + 0.05,
                  color=GRAY, lw=1.8, ls=(0, (4, 2)), zorder=3)
        ax.text(i + half_w + 0.07, null, f"Null q95 = {null:.3f}",
                va="center", ha="left", fontsize=8, color=GRAY_EC)
        # Band label.
        ax.text(i, hi + 0.012, f"Moral band [{lo:.2f}, {hi:.2f}]",
                ha="center", va="bottom", fontsize=8, fontweight="bold", color=c)
        # Verdict tell.
        tell = "Band below null\n→ position-invalid" if i == 0 else \
               "Band above null\n→ position-valid"
        ax.text(i, lo - 0.016, tell, ha="center", va="top",
                fontsize=9, color=c, fontweight="bold")

    ax.set_ylabel("Projection fraction", fontsize=10)
    ax.set_ylim(0.20, 0.70)
    ax.set_xlim(-0.6, 1.75)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Final pre-assistant\n(decision site)", "Mean content\n(content)"],
        fontsize=9,
    )
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "(a) Moral band vs covariance-matched null at two positions",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "The moral band sits below the covariance null at the decision site "
        "(OLMo-3-7B-Instruct)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "mn_ladder")


# ---------------------------------------------------------------------------
# Figure 3: depth-indexed verdict — a read-layer asymmetry that collapses
# ---------------------------------------------------------------------------
def fig_depth_collapse() -> str:
    df = pd.read_csv(_resolve("mn_depth_collapse.csv"))

    # x categorical, narrative order: read layer (16) then matched depth (12).
    layer_order = [16, 12]
    xpos = {16: 0, 12: 1}
    series = {"OLMo-3-7B": INDIGO, "Llama-3.1-8B": RED}

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.axhline(0, color=GRAY, lw=1.0, ls="--", zorder=1)
    ax.text(1.5, 0.0, "A = 0\n(symmetric)", va="center", ha="left",
            fontsize=8, color=GRAY_EC)

    for model, color in series.items():
        sub = df[df["model"] == model].set_index("layer").loc[layer_order]
        xs = [xpos[l] for l in layer_order]
        ys = sub["A"].tolist()
        lo = (sub["A"] - sub["ci_low"]).tolist()
        hi = (sub["ci_high"] - sub["A"]).tolist()
        ax.errorbar(
            xs, ys, yerr=[lo, hi],
            marker="o", ms=5, lw=2, color=color, alpha=0.9,
            capsize=3.5, elinewidth=1.3, label=model, zorder=3,
        )

    # Llama collapse annotation.
    ax.annotate(
        "Llama A: +0.82 → −0.28\n(read-layer artifact)",
        xy=(0, 0.82), xytext=(0.34, 0.66),
        fontsize=8, color=RED, bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
    # Cross-model difference annotation.
    ax.text(0.0, -1.0,
            "A(Llama) − A(OLMo):  +1.03 (read)  →  +0.26 (matched)",
            ha="left", va="center", fontsize=8, color="black", bbox=ANN_BBOX)

    ax.set_ylabel("Engage/disengage asymmetry A", fontsize=10)
    ax.set_ylim(-1.15, 1.12)
    ax.set_xlim(-0.35, 1.95)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Layer 16\n(read layer)",
                        "Layer 12\n(depth-matched /\npre-commitment)"], fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        "(a) Asymmetry A vs patch layer (read layer 16 to depth-matched 12)",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "The Llama read-layer asymmetry collapses at matched depth "
        "(Llama-3.1-8B vs OLMo-3-7B)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "mn_depth_collapse")


def main() -> None:
    outputs = [fig_bottleneck_pr(), fig_ladder(), fig_depth_collapse()]
    for path in outputs:
        size = os.path.getsize(path)
        print(f"wrote {os.path.basename(path)}  ({size} bytes)")


if __name__ == "__main__":
    main()
