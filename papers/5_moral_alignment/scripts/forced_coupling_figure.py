#!/usr/bin/env python3
"""Section 6.5-6.6 figure: forced coupling is geometric, not functional.

Two panels from the committed Stage-1/Stage-2 results:
  (a) refusal -> moral-subspace projection: the coupling reaches 0.50 pre-SFT
      (past the 0.40 threshold) but degrades to 0.26 through SFT (below it), while
      control stays near the 0.10 baseline. Pre-SFT is the proto-refusal contrast;
      post-SFT is the fitted refusal direction that Heretic ablation targets.
  (b) refusal rate under moral-subspace ablation: ablating V RAISES the coupled
      model's refusal (0.79 -> 0.93) rather than removing it, while control is
      flat -> the moral subspace is a comprehension substrate, not a refusal one.

Matplotlib-only; styling matches dependency_figures.py / pipeline_figures.py.

Usage:
    python papers/5_moral_alignment/scripts/forced_coupling_figure.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_PAPER_ROOT = Path(__file__).resolve().parent.parent
_S1 = _PAPER_ROOT / "outputs/intervention_stage1"
_S2 = _PAPER_ROOT / "outputs/intervention_stage2"

C = {"coupled": "#1565C0", "control": "#9E9E9E", "thresh": "#C62828",
     "clean": "#1565C0", "ablated": "#EF6C00"}


def _load(p):
    return json.load(open(p))


def main() -> None:
    ap = argparse.ArgumentParser(description="Forced-coupling figure (geometric vs functional).")
    ap.add_argument("--s1-dir", default=str(_S1))
    ap.add_argument("--s2-dir", default=str(_S2))
    ap.add_argument("--figures-dir", default=str(_PAPER_ROOT / "outputs/figures"))
    args = ap.parse_args()

    gate = _load(Path(args.s1_dir) / "stage1_gate_report.json")
    gate = gate["A1_offtarget_families"]["projection"]
    pre = {"coupled": gate["coupling_r64_qv_mlp"]["refusal"],
           "control": gate["control_r64_qv_mlp"]["refusal"]}
    post = {arm: _load(Path(args.s2_dir) / f"heretic_{arm}" / "refusal_morality_geometry.json")
            ["moral_subspace_projection_fraction"] for arm in ("coupled", "control")}
    a3 = _load(Path(args.s2_dir) / "stage2_a3.json")["arms"]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 11,
        "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.25,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    # --- (a) projection: pre-SFT -> post-SFT, coupled vs control ---
    x = [0, 1]
    for arm in ("coupled", "control"):
        ax1.plot(x, [pre[arm], post[arm]], "o-", color=C[arm], lw=2, ms=7, label=arm)
        for xi, yi in zip(x, [pre[arm], post[arm]]):
            ax1.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                         xytext=(0, 8 if arm == "coupled" else -14), ha="center", fontsize=9)
    ax1.axhline(0.40, color=C["thresh"], ls=":", lw=1.4, label="coupling threshold 0.40")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["pre-SFT\n(proto-contrast)", "post-SFT\n(refusal direction)"])
    ax1.set_ylabel("projection onto moral subspace $V$")
    ax1.set_ylim(0, 0.62)
    ax1.set_xlim(-0.25, 1.25)
    ax1.set_title("(a) Geometric coupling degrades through SFT")
    ax1.legend(loc="upper right")

    # --- (b) A3: refusal rate clean vs MFT-ablated, coupled vs control ---
    arms = ["coupled", "control"]
    clean = [a3[a]["refusal_clean"] for a in arms]
    abl = [a3[a]["refusal_mft_ablated"] for a in arms]
    import numpy as np
    xb = np.arange(len(arms))
    w = 0.36
    b1 = ax2.bar(xb - w / 2, clean, w, color=C["clean"], label="V intact")
    b2 = ax2.bar(xb + w / 2, abl, w, color=C["ablated"], label="V ablated")
    for bars in (b1, b2):
        for r in bars:
            ax2.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                         textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(arms)
    ax2.set_ylabel("refusal rate on harmful set")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("(b) Ablating $V$ raises refusal, not removes it")
    ax2.legend(loc="lower right")

    fig.tight_layout()
    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"forced_coupling.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {figdir/'forced_coupling.png'} (pre {pre}, post {post})")


if __name__ == "__main__":
    main()
