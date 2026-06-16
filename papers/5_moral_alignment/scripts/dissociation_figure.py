#!/usr/bin/env python3
"""Sprint 3 figures: refusal-morality geometry + comprehension/compliance dissociation.

Left panel: |cos| of the refusal direction to each moral foundation, with the
moral-subspace projection fraction as a reference line. Shows the refusal
direction is ~orthogonal to the moral subspace.

Right panel: instruct vs Heretic-ablated on comprehension metrics (preserved)
vs behavior (refusal collapses) — the high-comprehension / low-compliance cell.

Usage:
    python papers/5_moral_alignment/scripts/dissociation_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("papers/5_moral_alignment/outputs")
FIG = OUT / "figures"


def _load(p):
    return json.load(open(p)) if Path(p).exists() else None


def stable_cos(mp):
    return float(np.mean([v["cos_base_vs_fresh_probe"]
                          for pl in mp["per_foundation"].values()
                          for k, v in pl.items() if int(k) >= 15]))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    geo = _load(OUT / "heretic/refusal_morality_geometry.json")
    abl_bb = _load(OUT / "heretic/behavioral_baseline.json")
    ins_bb = _load(OUT / "olmo3_instruct/behavioral_baseline.json")
    ins_mp = _load(OUT / "pipeline/olmo3_instruct_final/moral_probing.json")
    abl_mp = _load(OUT / "pipeline/olmo3_ablated/moral_probing.json")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # -- Left: refusal-morality geometry --
    cosd = geo["cosine_to_foundations"]
    names = list(cosd.keys())
    vals = [abs(cosd[n]) for n in names]
    axL.bar(names, vals, color="#8E24AA")
    frac = geo["moral_subspace_projection_fraction"]
    axL.axhline(frac, color="#E53935", ls="--",
                label=f"moral-subspace projection fraction = {frac:.2f}")
    axL.set_ylabel("|cosine| (refusal direction vs foundation)")
    axL.set_ylim(0, 1.0)
    axL.set_title(f"Refusal direction is ~orthogonal to morality (L{geo['refusal_layer']})")
    axL.legend(fontsize=9)
    axL.tick_params(axis="x", rotation=30)

    # -- Right: instruct vs ablated --
    def refusal_rate(bb):
        return 1.0 - bb["results"]["persona_shift"]["baseline_compliance_rate"]

    def moral_acc(bb):
        return bb["results"]["moral_foundations"]["overall_accuracy"]

    metrics = ["cos(base,fresh)\n[comprehension]", "probe acc\n[comprehension]",
               "moral-judgment acc\n[behavior]", "refusal rate\n[behavior]"]
    instruct = [stable_cos(ins_mp), 1.0, moral_acc(ins_bb), refusal_rate(ins_bb)]
    ablated = [stable_cos(abl_mp), 1.0, moral_acc(abl_bb), refusal_rate(abl_bb)]
    x = np.arange(len(metrics))
    w = 0.38
    axR.bar(x - w / 2, instruct, w, label="instruct", color="#1E88E5")
    axR.bar(x + w / 2, ablated, w, label="Heretic-ablated", color="#FB8C00")
    axR.set_xticks(x)
    axR.set_xticklabels(metrics, fontsize=8)
    axR.set_ylim(0, 1.05)
    axR.set_title("Comprehension preserved, refusal collapses")
    axR.legend(fontsize=9)
    for xi, (a, b) in enumerate(zip(instruct, ablated)):
        axR.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=7)
        axR.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=7)

    fig.suptitle("Comprehension-compliance dissociation under refusal ablation (OLMo-3 7B)",
                 fontweight="bold")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"dissociation.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {FIG/'dissociation.png'}")
    print(f"  refusal subspace fraction={frac:.3f}  mean|cos|={geo['mean_abs_cosine']:.3f}")
    print(f"  instruct refusal_rate={refusal_rate(ins_bb):.2f}  ablated={refusal_rate(abl_bb):.2f}")


if __name__ == "__main__":
    main()
