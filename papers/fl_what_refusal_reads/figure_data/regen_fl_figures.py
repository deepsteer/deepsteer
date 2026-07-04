"""Regenerate the three flagship figures for "What Refusal Reads".

Every figure reads only from its committed CSV in this directory; no model, GPU,
or network access. The numbers are the verified, committed values from the
program's results (crystallize-then-rotate trajectory, the nested rank sweep, and
the GPT-OSS reversible-reader flips); this script only draws them.

Figures produced (vector PDF, ~4.8 in wide, clean academic matplotlib), written
to ../figures/:
  1. fl_crystallization.pdf       <- fl_crystallization.csv
  2. fl_one_knob.pdf              <- fl_one_knob.csv
  3. fl_gpt_oss_reversibility.pdf <- fl_gpt_oss_reversibility.csv

Run from this directory:
    python3 regen_fl_figures.py

Colors: Okabe-Ito (colorblind-safe). Categorical hues assigned in a fixed order,
never cycled; identity never carried by color alone (legends / direct labels).
Style matches ../../figure_data/regen_mn_figures.py.
"""

from __future__ import annotations

import os

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless, deterministic vector output
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.normpath(os.path.join(HERE, "..", "figures"))

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


def _src(name: str) -> str:
    return os.path.join(HERE, name)


def _out(name: str) -> str:
    return os.path.join(FIGDIR, name)


# ---------------------------------------------------------------------------
# Figure 1: the moral subspace crystallizes; the refusal gate does not
# ---------------------------------------------------------------------------
def fig_crystallization() -> str:
    df = pd.read_csv(_src("fl_crystallization.csv"))
    stages = df["stage"].tolist()
    x = list(range(len(stages)))
    moral = df["moral_cos"].tolist()
    refusal = df["refusal_cos"].tolist()

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    # Pretraining vs alignment: a faint separator after "converged" (index 1).
    sep = 1.5
    ax.axvspan(-0.4, sep, color="#F4F4F4", zorder=0)
    ax.text(0.5, 0.055, "pretraining", ha="center", va="bottom",
            fontsize=8.2, color=MUTED, style="italic")
    ax.text(3.0, 0.055, "alignment (SFT / DPO / RLVR)", ha="center", va="bottom",
            fontsize=8.2, color=MUTED, style="italic")

    # Crystallization threshold.
    ax.axhline(0.50, ls="--", lw=1.2, color=MUTED, zorder=2)
    ax.text(1.58, 0.53, "crystallization threshold (cos = 0.50)",
            ha="left", va="bottom", fontsize=8.2, color=MUTED)

    # Moral subspace: crystallizes to ~1.0, one-time ~40 deg rotation at SFT.
    ax.plot(x, moral, marker="o", ms=6.5, lw=2.1, color=OI_BLUE,
            zorder=4, label="moral subspace")
    # Refusal gate: flat and low, never crystallizes.
    ax.plot(x, refusal, marker="s", ms=6.0, lw=2.1, color=OI_VERMILLION,
            zorder=4, label="refusal gate")

    # Direct value labels (identity not carried by color alone).
    ax.annotate("0.999", xy=(1, 0.999), xytext=(1, 1.055),
                ha="center", va="bottom", fontsize=9, color=OI_BLUE,
                fontweight="bold")
    ax.annotate("0.155", xy=(0, 0.155), xytext=(-0.02, 0.245),
                ha="center", va="bottom", fontsize=9, color=OI_VERMILLION,
                fontweight="bold")

    # The one-time SFT rotation (0.999 -> 0.757).
    ax.annotate(
        "one-time ~40 deg\nrotation at SFT\n(0.999 -> 0.757)",
        xy=(2, 0.757), xytext=(2.55, 0.905),
        fontsize=8.2, color=OI_BLUE, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=OI_BLUE, lw=1.0),
    )

    ax.set_ylabel("direction-preservation cosine")
    ax.set_ylim(0.0, 1.13)
    ax.set_xlim(-0.4, len(stages) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=16, ha="right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)
    ax.legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.32),
              handlelength=1.6)

    fig.tight_layout()
    out = _out("fl_crystallization.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 2: one knob — R_refusal saturates at the rank-1 harm level
# ---------------------------------------------------------------------------
def fig_one_knob() -> str:
    df = pd.read_csv(_src("fl_one_knob.csv"))
    ks = df["k"].tolist()
    xpos = list(range(len(ks)))  # categorical, ~log-spaced ranks 1/3/8/16
    judg = df["R_judgment"].tolist()
    refu = df["R_refusal"].tolist()
    null = df["random_null"].tolist()
    fit = df["one_knob_fit"].tolist()

    fig, ax = plt.subplots(figsize=(4.8, 3.6))

    # Harm-ceiling reference (rank-1 harm level the refusal reader saturates at).
    ax.axhline(0.31, ls=":", lw=1.1, color=MUTED, zorder=1)

    # One-knob model overlay: min(0.31, R_judgment), drawn as a fit line.
    ax.plot(xpos, fit, lw=1.6, color=INK, ls=(0, (5, 2)), zorder=3,
            label="one-knob model  min(0.31, R$_{\\mathrm{judgment}}$)")

    # R_judgment climbs (reads the whole subspace).
    ax.plot(xpos, judg, marker="o", ms=6.5, lw=2.1, color=OI_BLUE,
            zorder=5, label="R$_{\\mathrm{judgment}}$")
    # R_refusal saturates at the harm rank-1 level.
    ax.plot(xpos, refu, marker="s", ms=6.0, lw=2.1, color=OI_VERMILLION,
            zorder=5, label="R$_{\\mathrm{refusal}}$")
    # Random null (~0 for all k).
    ax.plot(xpos, null, marker="^", ms=5.0, lw=1.4, color=FAINT,
            zorder=2, label="random-basis null")

    # Direct labels (no legend): each series named in its own clear zone, so
    # identity never rides on color alone and nothing overlaps the curves.
    ax.annotate(
        "R$_{\\mathrm{judgment}}$ climbs\n(reads the whole subspace)",
        xy=(2, 0.59), xytext=(0.12, 0.70),
        fontsize=8.4, color=OI_BLUE, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=OI_BLUE, lw=1.0),
    )
    ax.annotate(
        "R$_{\\mathrm{refusal}}$ saturates at the\nharm rank-1 level (0.31)",
        xy=(2, 0.285), xytext=(1.15, 0.15),
        fontsize=8.4, color=OI_VERMILLION, ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color=OI_VERMILLION, lw=1.0),
    )
    ax.annotate(
        "one-knob model\nmin(0.31, R$_{\\mathrm{judgment}}$),  RMSE 0.036",
        xy=(3.0, 0.315), xytext=(3.4, 0.45),
        fontsize=8.0, color=INK, ha="right", va="center",
        arrowprops=dict(arrowstyle="->", color=INK, lw=0.9),
    )
    ax.text(2.6, 0.045, "random-basis null (~0)", ha="center", va="bottom",
            fontsize=8.0, color=MUTED)

    ax.set_ylabel("fraction of full-patch effect transferred")
    ax.set_xlabel("moral-basis rank  k")
    ax.set_ylim(-0.03, 0.80)
    ax.set_xlim(-0.25, len(ks) - 0.55)
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)

    fig.tight_layout()
    out = _out("fl_one_knob.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figure 3: GPT-OSS is a reversible reader — deliberation flips both directions
# ---------------------------------------------------------------------------
def fig_reversibility() -> str:
    df = pd.read_csv(_src("fl_gpt_oss_reversibility.csv"))
    labels = [
        "inculpating prefill\n(benign -> refuse)",
        "exculpating prefill\n(violating -> comply)",
    ]
    x = list(range(len(df)))
    frac = df["frac"].tolist()
    lo = (df["frac"] - df["ci_low"]).tolist()
    hi = (df["ci_high"] - df["frac"]).tolist()
    colors = [OI_VERMILLION, OI_BLUE]  # refuse-direction / comply-direction
    tags = [f"{int(f)}/{int(n)}" for f, n in zip(df["flipped"], df["n"])]

    fig, ax = plt.subplots(figsize=(4.8, 3.5))

    bars = ax.bar(x, frac, width=0.52, color=colors, edgecolor="none", zorder=2)
    ax.errorbar(x, frac, yerr=[lo, hi], fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=4)

    # Direct fraction labels above each Wilson interval.
    for xi, ci_hi, tag in zip(x, df["ci_high"].tolist(), tags):
        ax.text(xi, ci_hi + 0.03, tag, ha="center", va="bottom",
                fontsize=10.5, color=INK, fontweight="bold")

    # Monotone decision-channel note (all 10 exculpating items moved to comply).
    ax.text(
        0.5, -0.235,
        "decision-channel projection moved monotonically\n"
        "toward comply in all 10 exculpating items",
        transform=ax.transData, ha="center", va="top",
        fontsize=8.0, color=MUTED, style="italic",
    )

    ax.set_ylabel("fraction of items flipped by deliberation (0-1)")
    ax.set_ylim(0.0, 1.16)
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EDEDED", lw=0.8)

    fig.tight_layout()
    out = _out("fl_gpt_oss_reversibility.pdf")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(FIGDIR, exist_ok=True)
    outputs = [fig_crystallization(), fig_one_knob(), fig_reversibility()]
    for path in outputs:
        size = os.path.getsize(path)
        print(f"wrote {os.path.relpath(path, HERE)}  ({size} bytes)")


if __name__ == "__main__":
    main()
