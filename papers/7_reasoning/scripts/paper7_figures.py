#!/usr/bin/env python3
"""Paper 7 figures, each 1:1 with its source JSON under outputs/.

Anchored on the harmfulness-vs-refusal dissociation (the strongest, most novel,
most direct result), with the reply-inversion causal validation and the
harmfulness-vs-moral-foundations decomposition. Run after the experimental phases:
    python papers/7_reasoning/scripts/paper7_figures.py
Writes outputs/figures/fig_*.{pdf,png}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_OUT = Path(__file__).resolve().parent.parent / "outputs"
_FIG = _OUT / "figures"
_FIG.mkdir(parents=True, exist_ok=True)

# Panel order + display labels + colors (RL-deliberative first, then the distills).
PANEL = [("gpt_oss_20b", "GPT-OSS-20B\n(RL-deliberative)", "C0"),
         ("ds_r1_llama8b", "R1-Distill\nLlama-8B", "C1"),
         ("ds_r1_qwen14b", "R1-Distill\nQwen-14B", "C2")]


def _load(rel: str):
    p = _OUT / rel
    return json.loads(p.read_text()) if p.exists() else None


def _save(fig, name: str):
    for ext in ("pdf", "png"):
        fig.savefig(_FIG / f"{name}.{ext}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {_FIG}/{name}.pdf/.png")


def fig_dissociation():
    """ANCHOR: harmfulness is strongly encoded at t_inst yet ~orthogonal to refusal."""
    rows = [(lbl, c, _load(f"{k}/position_extraction.json")) for k, lbl, c in PANEL]
    rows = [(lbl, c, d["separation"]) for lbl, c, d in rows if d]
    if not rows:
        return
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    x = np.arange(len(rows))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.2))

    # A: harmfulness separation (d') at t_inst vs t_post-inst.
    d_inst = [r[2]["harmful_vs_harmless_at_t_inst"] for r in rows]
    d_post = [r[2]["harmful_vs_harmless_at_t_post"] for r in rows]
    axL.bar(x - 0.18, d_inst, 0.36, color=colors, label="t_inst (instruction)")
    axL.bar(x + 0.18, d_post, 0.36, color=colors, alpha=0.45, label="t_post-inst (prompt end)")
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=8)
    axL.set_ylabel("harmful/harmless separation  d'")
    axL.set_title("Harmfulness is strongly encoded — peaks at the instruction token", fontsize=9.5)
    axL.legend(fontsize=8); axL.grid(axis="y", alpha=0.3)

    # B: cosine(harmfulness, refusal) — near-orthogonal => separate concepts.
    cos = [r[2]["cos_harmfulness_refusal"] for r in rows]
    axR.bar(x, cos, 0.5, color=colors)
    axR.axhline(0, color="grey", lw=0.8)
    axR.axhline(1.0, color="grey", ls=":", lw=0.8)
    axR.text(len(rows) - 1, 0.97, "1.0 = same direction", fontsize=7.5, ha="right", color="grey")
    for xi, c in zip(x, cos):
        axR.text(xi, c + 0.02, f"{c:.2f}", ha="center", fontsize=8)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=8)
    axR.set_ylim(-0.05, 1.05)
    axR.set_ylabel("cosine(harmfulness, refusal)")
    axR.set_title("…yet nearly orthogonal to refusal — separate concepts", fontsize=9.5)
    axR.grid(axis="y", alpha=0.3)
    fig.suptitle("Comprehension (harmfulness) and refusal are separately encoded in reasoning models",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, "fig1_harmfulness_vs_refusal")


def fig_causal_validation():
    """Reply-inversion: steering the harmfulness direction shifts the harm judgment
    toward harmful (margin) and coherently flips it on instruct models."""
    files = sorted((_OUT / "control").glob("*_inversion.json")) if (_OUT / "control").exists() else []
    data = [json.loads(f.read_text()) for f in files]
    data = [d for d in data if d.get("grid")]
    if not data:
        return
    fig, axes = plt.subplots(1, len(data), figsize=(5.0 * len(data), 4.2), squeeze=False)
    for ax, d in zip(axes[0], data):
        name = d["repo"].split("/")[-1]
        thr = abs(d["clean_safe_mean_margin"])    # shift needed (on avg) to cross to harmful
        best_L = d.get("best_layer")
        alphas = d["alphas"]
        shifts = [d["grid"][f"L{best_L}_a{a}"]["mean_margin_shift"] for a in alphas]
        flips = [d["grid"][f"L{best_L}_a{a}"]["flip_rate"] for a in alphas]
        x = np.arange(len(alphas))
        bars = ax.bar(x, shifts, 0.6, color=["C0" if (fl or 0) > 0 else "0.7" for fl in flips])
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(thr, color="C3", ls="--", lw=1.0,
                   label=f"baseline margin |{thr:.1f}| (avg shift to flip)")
        for xi, s, fl in zip(x, shifts, flips):
            if fl:
                ax.annotate(f"flip\n{fl:.0%}", (xi, s), fontsize=8, color="C0", fontweight="bold",
                            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom")
        ax.set_xticks(x); ax.set_xticklabels([str(a) for a in alphas])
        ax.set_xlabel("steering alpha  (x residual norm)")
        ax.set_ylabel("harm-judgment margin shift  (toward harmful)")
        ax.set_title(f"{name}  (L{best_L})", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="upper right"); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Reply-inversion: steering the harmfulness direction shifts the harm judgment "
                 "toward harmful\nand coherently flips it — the direction is causally validated "
                 "(instruct positive control)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, "fig2_causal_validation")


def fig_harmfulness_vs_moral():
    """DECOMPOSITION: harmfulness projects mostly OUTSIDE the MFT moral subspace,
    with a small above-chance component inside it."""
    d = _load("position_vs_moral.json")
    if not d:
        return
    models = d["models"]
    keys = [k for k, _l, _c in PANEL if k in models]
    labels = [lbl.replace("\n", " ") for k, lbl, _c in PANEL if k in models]
    inside = [models[k]["harmfulness_in_moral_subspace_fraction"] for k in keys]
    floor = [models[k]["random_floor_fraction"] for k in keys]
    outside = [1 - v for v in inside]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x, inside, 0.55, color="C4", label="in MFT moral subspace")
    ax.bar(x, outside, 0.55, bottom=inside, color="0.85", label="outside (distinct harmfulness)")
    ax.plot(x, floor, "kD", ms=7, label="random-direction floor (chance)")
    for xi, v, fl in zip(x, inside, floor):
        ax.text(xi, v + 0.03, f"{v:.2f}\n({v/fl:.1f}x chance)", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction of the harmfulness direction")
    ax.set_title("Harmfulness is largely distinct from moral foundations,\n"
                 "with a small above-chance moral component", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig3_harmfulness_vs_moral")


def fig_trace_profile():
    """2a: moral/harm content vs fractional trace position — peaks mid-trace,
    fades to the decision (distributed and displaced)."""
    d = _load("trace_length_disentangle.json")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    plotted = False
    for k, lbl, c in PANEL:
        m = d.get("models", {}).get(k)
        if not m:
            continue
        fp = m["fractional_position"]
        js = sorted(fp, key=int)
        xs = [fp[j]["frac_pos"] for j in js]
        ys = [fp[j]["moral_frac"] for j in js]
        ax.plot(xs, ys, "o-", color=c, label=lbl.replace("\n", " "))
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_xlabel("fractional position in the reasoning trace  (0 = start, 1 = decision)")
    ax.set_ylabel("moral-subspace fraction")
    ax.set_title("Comprehension is distributed across the trace and displaced from the decision",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig4_trace_profile")


def fig_distributed_refusal():
    """2b: no single direction cleanly ablates GPT-OSS refusal (held-out)."""
    d = _load("gpt_oss_20b/yardstick_validation.json")
    if not d:
        return
    ho = d["held_out"]
    dirs = [("eop", "EOP\n(boundary)"), ("cot_mean", "CoT-mean"), ("cot_last", "CoT-last")]
    x = np.arange(len(dirs))
    clean = [ho[k]["clean_flip_rate"] or 0 for k, _l in dirs]
    incoh = [ho[k]["incoherent_rate"] or 0 for k, _l in dirs]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x - 0.18, clean, 0.36, color="C0", label="coherent flip (real ablation)")
    ax.bar(x + 0.18, incoh, 0.36, color="0.6", label="incoherent (over-ablation)")
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    ax.text(len(dirs) - 1, 0.52, "fire threshold", fontsize=7.5, ha="right", color="grey")
    ax.set_xticks(x); ax.set_xticklabels([l for _k, l in dirs], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("held-out rate (of baseline-refusable)")
    ax.set_title("GPT-OSS refusal is distributed: no single direction cleanly ablates it",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig5_distributed_refusal")


def fig_behavioral_contrast():
    """Behavioral contrast (CONFOUNDED by R1-distillation-degrades-refusal): GPT-OSS
    refuses, the distills mostly do not — not a functional-vs-imitated asymmetry."""
    rows = [(lbl, c, _load(f"{k}/refusal_baseline.json")) for k, lbl, c in PANEL]
    rows = [(lbl, c, d) for lbl, c, d in rows if d]
    if not rows:
        return
    labels = [r[0].replace("\n", " ") for r in rows]
    rates = [r[2]["refusal_rate_whole"] for r in rows]
    colors = [r[1] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.bar(x, rates, 0.55, color=colors)
    for xi, v in zip(x, rates):
        ax.text(xi, (v or 0) + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("clean refusal rate (harmful prompts)")
    ax.set_title("Behavioral contrast — confounded: R1 distillation degrades refusal\n"
                 "(a lower bound; soft-refuse-by-reframe under-counted)", fontsize=9.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, "fig6_behavioral_contrast")


def main() -> None:
    print("Paper 7 figures -> outputs/figures/")
    fig_dissociation()
    fig_causal_validation()
    fig_harmfulness_vs_moral()
    fig_trace_profile()
    fig_distributed_refusal()
    fig_behavioral_contrast()


if __name__ == "__main__":
    main()
