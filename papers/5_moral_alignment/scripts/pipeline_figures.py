#!/usr/bin/env python3
"""Sprint 2.5: pipeline figures (comprehension / compliance / coupling, etc.).

Reads the per-stage outputs produced by pipeline_study.py, coupling_
measurement.py, and behavioral_baseline.py and renders:

  1. three_curve.{png,pdf}      comprehension / compliance / coupling vs stage
  2. persona_emergence.{png,pdf} persona probe peak accuracy + persona-morality
                                 |cos| vs stage
  3. geometry_grid.{png,pdf}     cosine-matrix heatmaps at selected stages
  4. dendrogram_compare.{png,pdf} foundation clustering at selected stages

Every layer-wise plot flags the OLMo-3 full-attention layers (3,7,...,31) so a
4-layer periodicity, if any, is attributable to attention type.

Missing metrics degrade gracefully (gaps in the curve). Stage order is taken
from --order (comma-separated labels) or, if omitted, the directory's
pipeline_summary.json order.

Usage:
    python papers/5_moral_alignment/scripts/pipeline_figures.py \
        --pipeline-dir papers/5_moral_alignment/outputs/pipeline \
        --layer 16 \
        --figures-dir papers/5_moral_alignment/outputs/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402


def _load(path: Path):
    return json.load(open(path)) if path.exists() else None


def stage_comprehension(stage_dir: Path) -> float:
    """Mean over foundations of best-layer fresh probe accuracy."""
    mp = _load(stage_dir / "moral_probing.json")
    if not mp:
        return float("nan")
    accs = []
    for f, per_layer in mp.get("per_foundation", {}).items():
        vals = [v.get("fresh_probe_acc") for v in per_layer.values()
                if v.get("fresh_probe_acc") is not None]
        if vals:
            accs.append(max(vals))
    return float(np.mean(accs)) if accs else float("nan")


def stage_compliance(stage_dir: Path) -> float:
    bb = _load(stage_dir / "behavioral_baseline.json")
    if not bb:
        return float("nan")
    mf = bb.get("results", {}).get("moral_foundations", {})
    return float(mf.get("overall_accuracy", float("nan")))


def stage_coupling(stage_dir: Path) -> float:
    cp = _load(stage_dir / "coupling.json")
    if not cp:
        return float("nan")
    v = cp.get("coupling_agreement")
    return float(v) if v is not None else float("nan")


def stage_persona(stage_dir: Path) -> tuple[float, float]:
    pp = _load(stage_dir / "persona_probing.json")
    peak = float(pp.get("peak_accuracy", float("nan"))) if pp else float("nan")
    ang = _load(stage_dir / "persona_morality_angles.json")
    mean_abs = float("nan")
    if ang and ang.get("angles"):
        vals = [abs(c) for fl in ang["angles"].values() for c in fl.values()]
        mean_abs = float(np.mean(vals)) if vals else float("nan")
    return peak, mean_abs


def mark_full_attention(ax, layers):
    """Color full-attention layer ticks blue + bold."""
    full = set(du.OLMO3_FULL_ATTENTION_LAYERS)
    for tick, L in zip(ax.get_xticklabels(), layers):
        if L in full:
            tick.set_color("#1E88E5")
            tick.set_fontweight("bold")


def main() -> None:
    ap = argparse.ArgumentParser(description="Pipeline figures.")
    ap.add_argument("--pipeline-dir", required=True)
    ap.add_argument("--order", default=None, help="Comma-separated stage labels in order.")
    ap.add_argument("--layer", type=int, default=16, help="Layer for heatmaps/dendrograms.")
    ap.add_argument("--heatmap-stages", default=None,
                    help="Comma-separated stage labels for the geometry grid (default: all).")
    ap.add_argument("--figures-dir", default="papers/5_moral_alignment/outputs/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage

    pdir = Path(args.pipeline_dir)
    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    if args.order:
        labels = args.order.split(",")
    else:
        summ = _load(pdir / "pipeline_summary.json")
        labels = [s["label"] for s in summ] if summ else sorted(
            p.name for p in pdir.iterdir() if p.is_dir()
        )
    stage_dirs = [(lab, pdir / lab) for lab in labels if (pdir / lab).exists()]
    print(f"Stages: {[l for l,_ in stage_dirs]}")

    x = list(range(len(stage_dirs)))
    comp = [stage_comprehension(d) for _, d in stage_dirs]
    cply = [stage_compliance(d) for _, d in stage_dirs]
    coup = [stage_coupling(d) for _, d in stage_dirs]
    persona_acc, persona_ang = zip(*[stage_persona(d) for _, d in stage_dirs]) \
        if stage_dirs else ([], [])

    # -- Figure 1: three-curve --
    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.8), 5))
    ax.plot(x, comp, "o-", color="#43A047", label="Comprehension (probe acc)")
    ax.plot(x, cply, "s-", color="#E53935", label="Compliance (behavioral acc)")
    ax.plot(x, coup, "D-", color="#1E88E5", label="Coupling (agreement)")
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in stage_dirs], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comprehension / Compliance / Coupling across the alignment pipeline")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"three_curve.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -- Figure 2: persona emergence --
    fig, ax = plt.subplots(figsize=(max(8, len(x) * 0.8), 5))
    ax.plot(x, persona_acc, "o-", color="#8E24AA", label="Persona probe peak acc")
    ax2 = ax.twinx()
    ax2.plot(x, persona_ang, "^--", color="#FB8C00", label="Persona-morality mean |cos|")
    ax.set_xticks(x)
    ax.set_xticklabels([l for l, _ in stage_dirs], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Persona probe peak accuracy", color="#8E24AA")
    ax2.set_ylabel("Persona-morality mean |cos|", color="#FB8C00")
    ax.set_title("Persona emergence across the alignment pipeline")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(figdir / f"persona_emergence.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # -- Figures 3 & 4: geometry grid + dendrograms at --layer --
    hm_labels = (args.heatmap_stages.split(",") if args.heatmap_stages
                 else [l for l, _ in stage_dirs])
    panels = []
    for lab in hm_labels:
        g = _load(pdir / lab / "geometry.json")
        if not g:
            continue
        pl = g.get("per_layer", {}).get(str(args.layer))
        if not pl or "cosine_matrix" not in pl:
            continue
        panels.append((lab, g.get("foundations", FOUNDATION_ORDER),
                       np.array(pl["cosine_matrix"])))
    if panels:
        ncol = min(4, len(panels))
        nrow = (len(panels) + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
        for i, (lab, founds, cos) in enumerate(panels):
            ax = axes[i // ncol][i % ncol]
            im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
            short = [FOUNDATION_SHORT.get(f, f) for f in founds]
            ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short, fontsize=7)
            ax.set_title(f"{lab} (L{args.layer})", fontsize=9)
        for j in range(len(panels), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.colorbar(im, ax=axes, shrink=0.6, label="cosine")
        fig.suptitle(f"Foundation cosine matrices at layer {args.layer}", y=1.02)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"geometry_grid.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

        # dendrograms
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4 * nrow), squeeze=False)
        for i, (lab, founds, cos) in enumerate(panels):
            ax = axes[i // ncol][i % ncol]
            n = len(founds)
            condensed = [1 - cos[a, b] for a in range(n) for b in range(a + 1, n)]
            Z = linkage(np.array(condensed), method="ward")
            dendrogram(Z, labels=[FOUNDATION_SHORT.get(f, f) for f in founds],
                       ax=ax, leaf_font_size=8, color_threshold=0,
                       above_threshold_color="#666")
            ax.set_title(f"{lab} (L{args.layer})", fontsize=9)
        for j in range(len(panels), nrow * ncol):
            axes[j // ncol][j % ncol].axis("off")
        fig.suptitle(f"Foundation clustering at layer {args.layer}", y=1.02)
        for ext in ("png", "pdf"):
            fig.savefig(figdir / f"dendrogram_compare.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote figures to {figdir}")


if __name__ == "__main__":
    main()
