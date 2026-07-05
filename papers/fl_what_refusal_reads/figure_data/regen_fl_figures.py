"""Regenerate all six figures for the flagship "What Refusal Reads".

Every figure reads only from its committed CSV in this directory; no model, GPU,
or network access. The numbers are the verified, committed values from the
program's results (crystallize-then-rotate trajectory, the nested rank sweep, the
GPT-OSS reversible-reader flips, and the three reused cross-model instruments);
this script only draws them. The flagship is self-contained: the three reused
methods-note instruments live here under fl_-prefixed CSV copies.

Figures produced (vector PDF + PNG mirror, Paper-1 higher-readability style),
written to ../figures/:
  1. fl_bottleneck_pr.{pdf,png}        <- fl_bottleneck_pr.csv        (reused)
  2. fl_calibration_ladder.{pdf,png}   <- fl_calibration_ladder.csv   (reused)
  3. fl_crystallization.{pdf,png}      <- fl_crystallization.csv
  4. fl_one_knob.{pdf,png}             <- fl_one_knob.csv
  5. fl_depth_collapse.{pdf,png}       <- fl_depth_collapse.csv        (reused)
  6. fl_gpt_oss_reversibility.{pdf,png}<- fl_gpt_oss_reversibility.csv

Run from this directory:
    python3 regen_fl_figures.py

Style: matches papers/1_accuracy_vs_fragility/scripts/*. Material palette with a
fixed semantic mapping (moral/judgment/comprehension = indigo, refusal/invalid =
red, valid/comply = green, null/reference = gray, secondary = orange),
descriptive figure suptitles, lettered panel titles, direct bold value labels,
and both PDF + PNG output. Identity is never carried by color alone.
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
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))

# --- Material palette (Paper-1 convention) ---------------------------------
GREEN = "#4CAF50"    # valid / comply
RED = "#F44336"      # refusal / invalid / critical boundary
INDIGO = "#3F51B5"   # moral / judgment / comprehension / primary series
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


def _src(name: str) -> str:
    return os.path.join(HERE, name)


def _save(fig: plt.Figure, stem: str) -> str:
    """Write both a vector PDF and a 200-dpi PNG mirror to ../figures/."""
    pdf = os.path.join(FIGDIR, f"{stem}.pdf")
    png = os.path.join(FIGDIR, f"{stem}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    plt.close(fig)
    return pdf


# ===========================================================================
# Reused methods-note instruments (self-contained copies)
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure: decision-site participation ratio is a control-token bottleneck
# ---------------------------------------------------------------------------
def fig_bottleneck_pr() -> str:
    df = pd.read_csv(_src("fl_bottleneck_pr.csv"))
    models = df["model"].tolist()
    x = np.arange(len(models))
    ds = df["decision_site_pr"].tolist()
    content = df["content_pr"].tolist()  # NaN where missing (GPT-OSS)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

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

    for xi, v in zip(x, ds):
        ax.text(xi, v + 0.7, f"{v:.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=INDIGO)
    for xi, c in zip(x, content):
        if not pd.isna(c):
            ax.text(xi, c + 0.7, f"{c:.0f}", ha="center", va="bottom",
                    fontsize=8, color=GRAY_EC)

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
        "The refusal decision lives in a 9-to-15-dimensional control-token "
        "bottleneck (four models)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fl_bottleneck_pr")


# ---------------------------------------------------------------------------
# Figure: calibrated ladder — band-below-null tell at the decision site
# ---------------------------------------------------------------------------
def fig_calibration_ladder() -> str:
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    df = pd.read_csv(_src("fl_calibration_ladder.csv"))
    tags = df["tag"].tolist()
    x = np.arange(len(tags))
    half_w = 0.26

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for i, row in df.iterrows():
        lo, hi = row["band_min"], row["band_max"]
        ax.add_patch(plt.Rectangle(
            (i - half_w, lo), 2 * half_w, hi - lo,
            facecolor=INDIGO, alpha=0.18, edgecolor=INDIGO, lw=1.1, zorder=2))
        ax.text(i, hi + 0.012, f"[{lo:.2f}, {hi:.2f}]", ha="center", va="bottom",
                fontsize=8, color=INDIGO)
        ax.hlines(row["null_q95"], i - half_w - 0.04, i + half_w + 0.04,
                  color=GRAY_EC, lw=1.5, ls=(0, (4, 2)), zorder=3)

    ax.plot(x, df["persona"].tolist(), "o", ms=7, color=GRAY, zorder=4)
    ax.plot(x, df["refusal"].tolist(), "s", ms=9, color=RED, zorder=5)
    for i, r in zip(x, df["refusal"].tolist()):
        ax.text(i + 0.08, r, f"{r:.2f}", va="center", ha="left",
                fontsize=9, fontweight="bold", color=RED)

    handles = [
        Patch(facecolor=INDIGO, alpha=0.18, edgecolor=INDIGO,
              label="Moral-family band (held-one-out)"),
        Line2D([], [], marker="s", ls="none", ms=9, color=RED,
               label="Refusal projection"),
        Line2D([], [], marker="o", ls="none", ms=7, color=GRAY,
               label="Persona reference"),
        Line2D([], [], color=GRAY_EC, ls=(0, (4, 2)),
               label="Covariance null (q95)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    ax.set_ylabel("Projection onto the moral subspace", fontsize=10)
    ax.set_ylim(0.0, 0.86)
    ax.set_xlim(-0.65, len(tags) - 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels(tags, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "(a) Refusal vs the held-one-out moral-family band, per checkpoint",
        fontsize=10, loc="left",
    )
    fig.suptitle(
        "Refusal projects below the moral-family band on every model",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fl_calibration_ladder")


# ---------------------------------------------------------------------------
# Figure: depth-indexed verdict — a read-layer asymmetry that collapses
# ---------------------------------------------------------------------------
def fig_depth_collapse() -> str:
    df = pd.read_csv(_src("fl_depth_collapse.csv"))

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

    ax.annotate(
        "Llama A: +0.82 → −0.28\n(read-layer artifact)",
        xy=(0, 0.82), xytext=(0.34, 0.66),
        fontsize=8, color=RED, bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
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
    return _save(fig, "fl_depth_collapse")


# ===========================================================================
# Flagship-native figures
# ===========================================================================

# ---------------------------------------------------------------------------
# Figure: the moral subspace crystallizes; the refusal gate does not
# ---------------------------------------------------------------------------
def fig_crystallization() -> str:
    df = pd.read_csv(_src("fl_crystallization.csv"))
    stages = df["stage"].tolist()
    x = np.arange(len(stages))
    moral = df["moral_cos"].tolist()
    refusal = df["refusal_cos"].tolist()

    fig, ax = plt.subplots(figsize=(7.6, 4.9))

    # Pretraining vs alignment: a faint separator after "converged" (index 1).
    sep = 1.5
    ax.axvspan(-0.4, sep, color="#F4F4F4", zorder=0)
    ax.text(0.5, 0.05, "Pretraining", ha="center", va="bottom",
            fontsize=8, color=GRAY_EC, style="italic")
    ax.text(3.0, 0.05, "Alignment (SFT / DPO / RLVR)", ha="center", va="bottom",
            fontsize=8, color=GRAY_EC, style="italic")

    # Crystallization threshold.
    ax.axhline(0.50, ls="--", lw=1.2, color=GRAY, zorder=2)
    ax.text(1.58, 0.53, "Crystallization threshold (cos = 0.50)",
            ha="left", va="bottom", fontsize=8, color=GRAY_EC)

    # Moral subspace: crystallizes to ~1.0, one-time ~40 deg rotation at SFT.
    ax.plot(x, moral, "o-", ms=5, lw=2, color=INDIGO, alpha=0.9,
            zorder=4, label="Moral subspace")
    # Refusal gate: flat and low, never crystallizes.
    ax.plot(x, refusal, "s-", ms=5, lw=2, color=RED, alpha=0.9,
            zorder=4, label="Refusal gate")

    # Direct value labels (identity not carried by color alone).
    ax.annotate("0.999", xy=(1, 0.999), xytext=(1, 1.055),
                ha="center", va="bottom", fontsize=9, color=INDIGO,
                fontweight="bold")
    ax.annotate("0.155", xy=(0, 0.155), xytext=(-0.02, 0.24),
                ha="center", va="bottom", fontsize=9, color=RED,
                fontweight="bold")

    # The one-time SFT rotation (0.999 -> 0.757).
    ax.annotate(
        "One-time ~40° rotation\nat SFT (0.999 → 0.757)",
        xy=(2, 0.757), xytext=(2.5, 0.91),
        fontsize=8, color=INDIGO, ha="left", va="center", bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color=INDIGO, lw=1.0),
    )

    ax.set_ylabel("Direction-preservation cosine", fontsize=10)
    ax.set_ylim(0.0, 1.13)
    ax.set_xlim(-0.4, len(stages) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=16, ha="right", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.34), fontsize=9)
    ax.set_title(
        "(a) Direction-preservation cosine across pretraining and alignment stages",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "Comprehension crystallizes in pretraining; the refusal gate does not "
        "(OLMo-3-7B)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fl_crystallization")


# ---------------------------------------------------------------------------
# Figure: one knob — R_refusal saturates at the rank-1 harm level
# ---------------------------------------------------------------------------
def fig_one_knob() -> str:
    df = pd.read_csv(_src("fl_one_knob.csv"))
    ks = df["k"].tolist()
    xpos = np.arange(len(ks))  # categorical, ~log-spaced ranks 1/3/8/16
    judg = df["R_judgment"].tolist()
    refu = df["R_refusal"].tolist()
    null = df["random_null"].tolist()
    fit = df["one_knob_fit"].tolist()

    fig, ax = plt.subplots(figsize=(7.8, 5.3))

    # Harm-ceiling reference (rank-1 harm level the refusal reader saturates at).
    ax.axhline(0.31, ls=":", lw=1.1, color=GRAY, zorder=1)

    # One-knob model overlay: min(0.31, R_judgment), drawn as a fit line.
    ax.plot(xpos, fit, lw=1.6, color="black", ls=(0, (5, 2)), zorder=3,
            label="One-knob fit")

    # R_judgment climbs (reads the whole subspace).
    ax.plot(xpos, judg, "o-", ms=5, lw=2, color=INDIGO, alpha=0.9,
            zorder=5, label="R$_{\\mathrm{judgment}}$")
    # R_refusal saturates at the harm rank-1 level.
    ax.plot(xpos, refu, "s-", ms=5, lw=2, color=RED, alpha=0.9,
            zorder=5, label="R$_{\\mathrm{refusal}}$")
    # Random null (~0 for all k).
    ax.plot(xpos, null, "^-", ms=5, lw=1.4, color=ORANGE, alpha=0.9,
            zorder=2, label="Random-basis null")

    # Direct labels alongside the legend, so identity never rides on color alone.
    # Placed in open zones that clear the upper-left legend and the curves.
    ax.annotate(
        "R$_{\\mathrm{judgment}}$ climbs (reads the whole subspace)",
        xy=(2, 0.59), xytext=(1.05, 0.75),
        fontsize=8, color=INDIGO, ha="left", va="center", bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color=INDIGO, lw=1.0),
    )
    ax.annotate(
        "R$_{\\mathrm{refusal}}$ saturates at the\nharm rank-1 level (0.31)",
        xy=(2, 0.26), xytext=(0.95, 0.12),
        fontsize=8, color=RED, ha="left", va="center", bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color=RED, lw=1.0),
    )
    ax.annotate(
        "One-knob model min(0.31, R$_{\\mathrm{judgment}}$),  RMSE 0.036",
        xy=(3.0, 0.315), xytext=(3.45, 0.48),
        fontsize=8, color="black", ha="right", va="center", bbox=ANN_BBOX,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.9),
    )

    ax.set_ylabel("Fraction of full-patch effect transferred", fontsize=10)
    ax.set_xlabel("Moral-basis rank k", fontsize=10)
    ax.set_ylim(-0.03, 0.84)
    ax.set_xlim(-0.25, len(ks) - 0.45)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(k) for k in ks], fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title(
        "(a) Transfer fraction vs moral-basis rank k (nested interchange sweep)",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "Refusal reads the harm percept and stops; judgment reads the subspace "
        "broadly (OLMo-3-7B)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fl_one_knob")


# ---------------------------------------------------------------------------
# Figure: GPT-OSS is a reversible reader — deliberation flips both directions
# ---------------------------------------------------------------------------
def fig_reversibility() -> str:
    df = pd.read_csv(_src("fl_gpt_oss_reversibility.csv"))
    labels = [
        "Inculpating prefill\n(benign → refuse)",
        "Exculpating prefill\n(violating → comply)",
    ]
    x = np.arange(len(df))
    frac = df["frac"].tolist()
    lo = (df["frac"] - df["ci_low"]).tolist()
    hi = (df["ci_high"] - df["frac"]).tolist()
    colors = [RED, GREEN]  # refuse-direction / comply-direction
    tags = [f"{f:.2f}\n({int(fl)}/{int(n)})"
            for f, fl, n in zip(df["frac"], df["flipped"], df["n"])]

    fig, ax = plt.subplots(figsize=(7.3, 4.9))

    ax.bar(x, frac, width=0.6, color=colors, edgecolor="black",
           linewidth=0.7, zorder=2)
    ax.errorbar(x, frac, yerr=[lo, hi], fmt="none", ecolor="black",
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    # Direct fraction + count labels above each Wilson interval.
    for xi, ci_hi, tag in zip(x, df["ci_high"].tolist(), tags):
        ax.text(xi, ci_hi + 0.03, tag, ha="center", va="bottom",
                fontsize=9, color="black", fontweight="bold")

    # Monotone decision-channel note (all 10 exculpating items moved to comply).
    ax.text(
        0.5, 0.14,
        "Decision-channel projection moved monotonically\n"
        "toward comply in all 10 exculpating items",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=8, color=GRAY_EC, style="italic", bbox=ANN_BBOX,
    )

    ax.set_ylabel("Fraction of items flipped by deliberation (0-1)", fontsize=10)
    ax.set_ylim(0.0, 1.18)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_title(
        "(a) Fraction of items flipped by an analysis prefill, with Wilson 95% intervals",
        fontsize=10, loc="left",
    )

    fig.suptitle(
        "GPT-OSS is a reversible reader: deliberation flips refusal both "
        "directions (GPT-OSS-20B)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "fl_gpt_oss_reversibility")


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    outputs = [
        fig_bottleneck_pr(),
        fig_calibration_ladder(),
        fig_crystallization(),
        fig_one_knob(),
        fig_depth_collapse(),
        fig_reversibility(),
    ]
    for path in outputs:
        size = os.path.getsize(path)
        print(f"wrote {os.path.relpath(path, HERE)}  ({size} bytes)")


if __name__ == "__main__":
    main()
