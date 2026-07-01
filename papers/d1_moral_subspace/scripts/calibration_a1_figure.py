#!/usr/bin/env python3
"""A1 calibrated-ladder figure, 1:1 with outputs/phase2/calibration/a1_summary.json.

One panel per tag: refusal's committed projection point(s) placed against the ladder rungs
(isotropic floor, covariance-matched null q50-q95, persona reference, and the moral-family
band from the held-one-out positive control). Reads "does refusal project like a random
direction / a non-moral voice / a genuine moral direction" straight off the axis.
Run after calibration_a1_ladder.py. House style: pdf+png at 160 dpi.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "outputs" / "phase2" / "calibration"
PANEL = [("base", "OLMo-3 Base\n(proto-refusal, L16)"),
         ("instruct", "OLMo-3 Instruct\n(gate, L16)"),
         ("think", "OLMo-3 Think\n(reasoning, L16)"),
         ("gpt_oss", "GPT-OSS-20B\n(reasoning, L12)")]

C_NULL, C_ISO, C_PERSONA, C_BAND, C_REF = "0.55", "0.7", "C1", "C2", "C3"


def main() -> None:
    summ = json.loads((OUT / "a1_summary.json").read_text())
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), sharey=True)
    for ax, (tag, label) in zip(axes, PANEL):
        r = summ[tag]
        ax.axhspan(r["null_q50"], r["null_q95"], color=C_NULL, alpha=0.28,
                   label="null q50-q95")
        ax.axhline(r["iso_floor"], ls=":", color=C_ISO, lw=1.2, label="isotropic floor")
        ax.axhline(r["persona_c"], ls="--", color=C_PERSONA, lw=1.6,
                   label="persona (moral-adjacent voice)")
        bmin, bmax = r["moral_family_band"]
        ax.axhspan(bmin, bmax, color=C_BAND, alpha=0.20, label="moral-family band (R1)")

        pts = r["refusal_points"]
        xs = [(i + 1) / (len(pts) + 1) for i in range(len(pts))]
        for x, (name, val) in zip(xs, pts.items()):
            ax.scatter([x], [val], color=C_REF, s=64, zorder=5)
            ax.annotate(f"{name}\n{val:.2f}", (x, val), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8, color=C_REF)
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", ls=":", alpha=0.3)
    axes[0].set_ylabel("projection fraction onto rank-3 $V_{moral}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle("A1 calibrated ladder: refusal projects below the moral-family band on every "
                 "model (even GPT-OSS in-trace P2 = 0.52)", fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"a1_ladder.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/a1_ladder.pdf/.png")


if __name__ == "__main__":
    main()
