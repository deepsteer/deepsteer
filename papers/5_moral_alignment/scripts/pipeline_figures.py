#!/usr/bin/env python3
"""Pipeline figures for Paper 5 (publication styling).

Renders, from the per-stage pipeline outputs:
  1. three_curve            comprehension / preservation / compliance / coupling
                            vs pipeline stage, with a pre/post-training divider.
  2. layerwise_geometry     per-layer mean foundation cosine (base vs instruct)
                            and cos(base,fresh) vs layer, with OLMo-3's
                            full-attention layers (3,7,...,31) marked -- the
                            "no 4-layer periodicity" evidence.
  3. persona_emergence      persona probe peak accuracy + persona-morality |cos|.
  4. geometry_grid          foundation cosine matrices at base/SFT/DPO/Instruct.
  5. dendrogram_compare     foundation clustering across those stages.

Descriptive prose lives in the LaTeX captions; panel titles are kept short.

Usage:
    python papers/5_moral_alignment/scripts/pipeline_figures.py \
        --pipeline-dir papers/5_moral_alignment/outputs/pipeline --layer 16
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_utils as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402

FULL = du.OLMO3_FULL_ATTENTION_LAYERS
C = {  # consistent palette
    "comp": "#2E7D32", "pres": "#EF6C00", "cply": "#C62828",
    "coup": "#1565C0", "persona": "#6A1B9A", "base": "#1565C0", "instruct": "#C62828",
}


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


def short_label(lab: str) -> str:
    s = lab.replace("olmo3_", "")
    m = re.match(r"pretrain_stage3_step(\d+)", s)
    if m:
        return f"{int(m.group(1))/1000:g}k"
    return {"base": "base", "sft_final": "SFT", "dpo_final": "DPO",
            "instruct_final": "Instruct", "ablated": "ablated"}.get(
        s, "r" + s.split("_")[-1] if s.startswith("instruct_step_") else s)


def stage_group(lab: str) -> str:
    s = lab.replace("olmo3_", "")
    if s.startswith("pretrain") or s == "base":
        return "pre-training"
    return "post-training"


def _load(path: Path):
    return json.load(open(path)) if Path(path).exists() else None


def stage_comprehension(d: Path) -> float:
    mp = _load(d / "moral_probing.json")
    if not mp:
        return float("nan")
    accs = [max(v.get("fresh_probe_acc", 0) for v in pl.values())
            for pl in mp.get("per_foundation", {}).values() if pl]
    return float(np.mean(accs)) if accs else float("nan")


def stage_preservation(d: Path) -> float:
    mp = _load(d / "moral_probing.json")
    if not mp:
        return float("nan")
    vals = [v["cos_base_vs_fresh_probe"] for pl in mp.get("per_foundation", {}).values()
            for L, v in pl.items() if int(L) >= 15 and "cos_base_vs_fresh_probe" in v]
    return float(np.mean(vals)) if vals else float("nan")


def stage_compliance(d: Path) -> float:
    bb = _load(d / "behavioral_baseline.json")
    if bb and "overall_accuracy" in bb.get("results", {}).get("moral_foundations", {}):
        return float(bb["results"]["moral_foundations"]["overall_accuracy"])
    cp = _load(d / "coupling.json")
    return float(cp["compliance_rate"]) if cp and cp.get("compliance_rate") is not None else float("nan")


def stage_coupling(d: Path) -> float:
    cp = _load(d / "coupling.json")
    return float(cp["coupling_agreement"]) if cp and cp.get("coupling_agreement") is not None else float("nan")


def stage_persona(d: Path) -> tuple[float, float]:
    pp = _load(d / "persona_probing.json")
    peak = float(pp.get("peak_accuracy", float("nan"))) if pp else float("nan")
    ang = _load(d / "persona_morality_angles.json")
    vals = [abs(c) for fl in ang["angles"].values() for c in fl.values()] if ang and ang.get("angles") else []
    return peak, (float(np.mean(vals)) if vals else float("nan"))


def per_layer_geo(d: Path):
    g = _load(d / "geometry.json")
    if not g:
        return [], []
    pl = g.get("per_layer", {})
    layers = sorted(int(k) for k in pl)
    return layers, [pl[str(L)]["mean_cosine"] for L in layers]


def per_layer_cosfresh(d: Path):
    mp = _load(d / "moral_probing.json")
    if not mp:
        return [], []
    by_layer: dict[int, list] = {}
    for plf in mp.get("per_foundation", {}).values():
        for L, v in plf.items():
            if "cos_base_vs_fresh_probe" in v:
                by_layer.setdefault(int(L), []).append(v["cos_base_vs_fresh_probe"])
    layers = sorted(by_layer)
    return layers, [float(np.mean(by_layer[L])) for L in layers]


def mark_full_attn(ax, label=True):
    for i, L in enumerate(FULL):
        ax.axvline(L, color="#1565C0", ls=":", lw=0.8, alpha=0.5,
                   label="full-attention layers" if (label and i == 0) else None)


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 5 pipeline figures.")
    ap.add_argument("--pipeline-dir", required=True)
    ap.add_argument("--order", default=None)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--heatmap-stages",
                    default="olmo3_base,olmo3_sft_final,olmo3_dpo_final,olmo3_instruct_final")
    ap.add_argument("--figures-dir", default="papers/5_moral_alignment/outputs/figures")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, linkage
    _style(plt)

    pdir = Path(args.pipeline_dir)
    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    if args.order:
        labels = args.order.split(",")
    else:
        # Canonical training order from checkpoint_grid.json (pipeline_summary.json
        # gets clobbered by single-state runs like the ablation battery).
        grid = _load(pdir.parent.parent / "checkpoint_grid.json")
        if grid:
            labels = [g["label"] for g in grid]
        else:
            summ = _load(pdir / "pipeline_summary.json")
            labels = [s["label"] for s in summ] if summ else sorted(
                p.name for p in pdir.iterdir() if p.is_dir())
    stages = [(lab, pdir / lab) for lab in labels if (pdir / lab).exists()]
    x = list(range(len(stages)))
    xlab = [short_label(l) for l, _ in stages]

    # ---- Figure 1: trajectory ----
    comp = [stage_comprehension(d) for _, d in stages]
    pres = [stage_preservation(d) for _, d in stages]
    cply = [stage_compliance(d) for _, d in stages]
    coup = [stage_coupling(d) for _, d in stages]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(x, comp, "o-", color=C["comp"], lw=1.8, ms=4, label="Comprehension (probe acc)")
    ax.plot(x, pres, "^-", color=C["pres"], lw=1.8, ms=4, label="Direction preservation (cos base$\\leftrightarrow$fresh)")
    ax.plot(x, cply, "s-", color=C["cply"], lw=1.8, ms=5, label="Compliance (behavioral acc)")
    ax.plot(x, coup, "D-", color=C["coup"], lw=1.8, ms=5, label="Coupling (agreement)")
    # pre/post-training divider
    groups = [stage_group(l) for l, _ in stages]
    boundary = next((i for i in range(1, len(groups)) if groups[i] != groups[i - 1]), None)
    if boundary is not None:
        ax.axvline(boundary - 0.5, color="0.4", ls="--", lw=1)
        ax.text(boundary / 2 - 0.5, 1.02, "pre-training", ha="center", fontsize=9, color="0.35")
        ax.text((boundary + len(x)) / 2 - 0.5, 1.02, "post-training", ha="center", fontsize=9, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(xlab, rotation=60, ha="right")
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="center right", framealpha=0.9)
    _save(fig, figdir, "three_curve")
    plt.close(fig)

    # ---- Figure 2: layer-wise geometry with full-attention markers ----
    bl, bcos = per_layer_geo(pdir / "olmo3_base")
    il, icos = per_layer_geo(pdir / "olmo3_instruct_final")
    cl, ccos = per_layer_cosfresh(pdir / "olmo3_instruct_final")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(bl, bcos, "o-", color=C["base"], ms=3, lw=1.5, label="base")
    a1.plot(il, icos, "s-", color=C["instruct"], ms=3, lw=1.5, label="Instruct")
    mark_full_attn(a1)
    a1.set_xlabel("layer"); a1.set_ylabel("mean pairwise foundation cosine")
    a1.set_title("(a) framework geometry by layer")
    a1.legend(framealpha=0.9)
    a2.plot(cl, ccos, "s-", color=C["instruct"], ms=3, lw=1.5, label="Instruct")
    mark_full_attn(a2)
    a2.axhspan(0, 1, xmin=0, xmax=0, color="none")  # keep ylim sane
    a2.set_xlabel("layer"); a2.set_ylabel("cos(base dir, fresh dir)")
    a2.set_title("(b) base-direction preservation by layer")
    a2.set_ylim(0.5, 1.02)
    a2.legend(framealpha=0.9)
    _save(fig, figdir, "layerwise_geometry")
    plt.close(fig)

    # ---- Figure 3: persona emergence ----
    pa, pang = zip(*[stage_persona(d) for _, d in stages]) if stages else ([], [])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(x, pa, "o-", color=C["persona"], lw=1.8, ms=4, label="Persona probe peak acc")
    ax2 = ax.twinx()
    ax2.plot(x, pang, "^--", color=C["pres"], lw=1.8, ms=4, label="Persona-morality mean |cos|")
    ax2.grid(False)
    if boundary is not None:
        ax.axvline(boundary - 0.5, color="0.4", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(xlab, rotation=60, ha="right")
    ax.set_ylabel("persona probe peak accuracy", color=C["persona"])
    ax.set_ylim(0, 1.05)
    ax2.set_ylabel("persona-morality mean |cos|", color=C["pres"])
    ax2.set_ylim(0, 0.5)
    lines = ax.get_lines()[:1] + ax2.get_lines()[:1]
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right", framealpha=0.9)
    _save(fig, figdir, "persona_emergence")
    plt.close(fig)

    # ---- Figures 4 & 5: geometry grid + dendrograms ----
    panels = []
    for lab in args.heatmap_stages.split(","):
        g = _load(pdir / lab / "geometry.json")
        pl = g.get("per_layer", {}).get(str(args.layer)) if g else None
        if pl and "cosine_matrix" in pl:
            panels.append((short_label(lab), g.get("foundations", FOUNDATION_ORDER),
                           np.array(pl["cosine_matrix"])))
    if panels:
        ncol = len(panels)
        fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.6), squeeze=False)
        short = [FOUNDATION_SHORT.get(f, f) for f in panels[0][1]]
        for i, (lab, _f, cos) in enumerate(panels):
            ax = axes[0][i]
            im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(short))); ax.set_yticks(range(len(short)))
            ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
            ax.set_yticklabels(short if i == 0 else [], fontsize=7)
            ax.set_title(lab, fontsize=11)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7, label="cosine")
        _save(fig, figdir, "geometry_grid")
        plt.close(fig)

        fig, axes = plt.subplots(1, ncol, figsize=(3.3 * ncol, 3.4), squeeze=False)
        for i, (lab, founds, cos) in enumerate(panels):
            ax = axes[0][i]
            n = len(founds)
            cond = np.array([1 - cos[a, b] for a in range(n) for b in range(a + 1, n)])
            dendrogram(linkage(cond, method="ward"),
                       labels=[FOUNDATION_SHORT.get(f, f) for f in founds],
                       ax=ax, leaf_font_size=8, leaf_rotation=90,
                       color_threshold=0, above_threshold_color="#777")
            ax.set_title(lab, fontsize=11)
        _save(fig, figdir, "dendrogram_compare")
        plt.close(fig)

    print(f"Wrote figures to {figdir}")


if __name__ == "__main__":
    main()
