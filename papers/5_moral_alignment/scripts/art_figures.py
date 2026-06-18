#!/usr/bin/env python3
"""§6 (ablation-resistance) figures for the merged Paper 5.

Three figures telling the ART arc:
  1. art_training_dynamics  the ART gap (CE_ablated − CE_normal on moral text)
                            vs step: v1 (diluted SFT batch) stays flat ~0, v2
                            (moral-only pool) climbs past the hinge target. Shows
                            ART only bites once the gap is measured on
                            concentrated moral content.
  2. art_ablation_grid      4-cell bars (control / +Heretic / ART / +Heretic) for
                            moral dependency and moral judgment: Heretic damages
                            ART no more than control -> no ablation resistance.
  3. refusal_projection     refusal -> moral-subspace projection fraction,
                            control vs ART, against the >0.40 goal and the ~0.10
                            Phase-2 baseline: it never moves -> refusal stays
                            orthogonal, which is why the whole approach can't
                            resist refusal-direction ablation.

Matplotlib-only; styling matches pipeline_figures.py / dependency_figures.py.

Usage:
    python papers/5_moral_alignment/scripts/art_figures.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_OUT = "papers/5_moral_alignment/outputs"
C = {"v1": "#9E9E9E", "v2": "#1565C0", "control": "#1565C0", "art": "#C62828",
     "goal": "#2E7D32", "base": "0.5", "target": "#EF6C00"}


def _style(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.grid": True, "grid.alpha": 0.25,
    })


def _save(fig, figdir, name):
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"{name}.{ext}", bbox_inches="tight")


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


def _gaps(log):
    s = (log or {}).get("result", {}).get("steps", [])
    xs = [x["step"] for x in s if x.get("art_gap") is not None]
    ys = [x["art_gap"] for x in s if x.get("art_gap") is not None]
    return xs, ys


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 5 §6 ablation-resistance figures.")
    ap.add_argument("--eval", default=f"{_OUT}/eval/comparison.json")
    ap.add_argument("--art-log", default=f"{_OUT}/art_sft/art_sft.json")
    ap.add_argument("--v1-art-log", default=f"{_OUT}/art_sft_v1_diluted/art_sft.json")
    ap.add_argument("--target-gap", type=float, default=0.3)
    ap.add_argument("--figures-dir", default=f"{_OUT}/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    figdir = Path(args.figures_dir); figdir.mkdir(parents=True, exist_ok=True)

    comp = _load(args.eval)
    if comp is None:
        raise SystemExit(f"No eval comparison at {args.eval}")
    S = comp["states"]
    R = comp.get("refusal_morality", {})

    # ---- Figure 1: ART training dynamics (v1 diluted vs v2 moral pool) ----
    fig, ax = plt.subplots(figsize=(8, 4.3))
    x2, y2 = _gaps(_load(args.art_log))
    ax.plot(x2, y2, "-", color=C["v2"], lw=1.8, label="v2: moral-only gap pool")
    v1 = _load(args.v1_art_log)
    if v1:
        x1, y1 = _gaps(v1)
        ax.plot(x1, y1, "-", color=C["v1"], lw=1.6, label="v1: diluted SFT batch")
    ax.axhline(args.target_gap, color=C["target"], ls="--", lw=1,
               label=f"hinge target ({args.target_gap})")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("training step")
    ax.set_ylabel("ART gap (nats/token)")
    ax.set_title("ART only bites once the gap is on concentrated moral text")
    ax.legend(loc="upper left", framealpha=0.9)
    _save(fig, figdir, "art_training_dynamics")
    plt.close(fig)

    # ---- Figure 2: 4-cell ablation grid (Heretic doesn't discriminate) ----
    cells = ["control", "control_ablated", "art", "art_ablated"]
    labels = ["control", "control\n+Heretic", "ART", "ART\n+Heretic"]
    colors = [C["control"], C["control"], C["art"], C["art"]]
    hatches = ["", "//", "", "//"]
    panels = [("moral_dependency_score", "moral dependency (DiD, nats/tok)"),
              ("moral_judgment_acc", "moral judgment accuracy")]
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.3), constrained_layout=True)
    for ax, (key, ylab) in zip(axes, panels):
        vals = [S[c].get(key) for c in cells]
        bars = ax.bar(range(4), vals, color=colors, edgecolor="black", linewidth=0.6)
        for b, h in zip(bars, hatches):
            b.set_hatch(h)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels)
        ax.set_ylabel(ylab)
    axes[0].set_title("(a) dependency shift, Heretic-insensitive")
    axes[1].set_title("(b) Heretic doesn't degrade moral judgment")
    fig.suptitle("Heretic damages ART no more than control — no ablation resistance")
    _save(fig, figdir, "art_ablation_grid")
    plt.close(fig)

    # ---- Figure 3: refusal -> moral projection fraction (stays orthogonal) ----
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    fracs = [R.get("control", {}).get("moral_subspace_projection_fraction"),
             R.get("art", {}).get("moral_subspace_projection_fraction")]
    ax.bar([0, 1], fracs, color=[C["control"], C["art"]], edgecolor="black",
           linewidth=0.6, width=0.6)
    ax.axhline(0.40, color=C["goal"], ls="--", lw=1.2, label="ART goal (≥0.40)")
    ax.axhline(0.10, color=C["base"], ls=":", lw=1.2, label="Phase-2 baseline (~0.10)")
    for i, v in enumerate(fracs):
        if v is not None:
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["control", "ART"])
    ax.set_ylabel("refusal → moral-subspace projection fraction")
    ax.set_ylim(0, 0.45)
    ax.set_title("Refusal stays orthogonal — ART never moves it")
    ax.legend(loc="upper right", framealpha=0.9)
    _save(fig, figdir, "refusal_projection")
    plt.close(fig)

    print(f"Wrote figures to {figdir}:")
    print("  art_training_dynamics.{png,pdf}")
    print("  art_ablation_grid.{png,pdf}")
    print("  refusal_projection.{png,pdf}")
    print(f"\n  v2 gap: {y2[0]:.3f} -> {y2[-1]:.3f}" if y2 else "")
    print(f"  dependency  control={S['control'].get('moral_dependency_score')}  "
          f"art={S['art'].get('moral_dependency_score')}")
    print(f"  refusal proj  control={fracs[0]}  art={fracs[1]}")


if __name__ == "__main__":
    main()
