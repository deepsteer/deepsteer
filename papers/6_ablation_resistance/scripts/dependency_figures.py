#!/usr/bin/env python3
"""Sprint 5.3: moral-dependency trajectory figures for Paper 6.

Renders, from the dependency sweep's ``dependency_summary.json``:
  1. dependency_trajectory   moral-dependency score across the OLMo-3 pipeline
                             (pretraining anneal -> base -> SFT -> DPO -> Instruct),
                             with a zero line and a pre/post-training divider.
                             This is the Sprint 5.3 headline: does the model
                             naturally develop reliance on its moral subspace?
  2. dependency_components   the two arms behind the score, Δmoral and Δneutral
                             cross-entropy under ablation, so the difference-in-
                             differences (their gap) is visible.

Matplotlib-only; styling matches paper 5's pipeline_figures.py.

Usage:
    python papers/6_ablation_resistance/scripts/dependency_figures.py \
        --summary papers/6_ablation_resistance/outputs/dependency/dependency_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

C = {  # palette aligned with pipeline_figures.py
    "score": "#1565C0", "moral": "#C62828", "neutral": "#2E7D32", "zero": "0.4",
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
    """Compact x-tick label (matches pipeline_figures.short_label)."""
    s = lab.replace("olmo3_", "")
    m = re.match(r"pretrain_stage3_step(\d+)", s)
    if m:
        return f"{int(m.group(1)) / 1000:g}k"
    if s.startswith("instruct_step_"):
        return "r" + s.split("_")[-1]
    return {"base": "base", "sft_final": "SFT", "dpo_final": "DPO",
            "instruct_final": "Instruct", "ablated": "ablated"}.get(s, s)


def stage_group(lab: str) -> str:
    s = lab.replace("olmo3_", "")
    return "pre-training" if (s.startswith("pretrain") or s == "base") else "post-training"


def _load(path: Path):
    return json.load(open(path)) if Path(path).exists() else None


def order_states(summary: dict, summary_path: Path, grid_arg: str | None) -> list[dict]:
    """Trajectory entries in training order (grid order), skipping failures.

    The summary is already written in grid order, but re-ordering by the grid is
    robust to subset/ONLY runs and matches pipeline_figures' canonical ordering.
    """
    rows = {r["label"]: r for r in summary.get("trajectory", []) if "error" not in r}
    grid_path = Path(grid_arg) if grid_arg else (
        summary_path.parent.parent.parent.parent / "5_moral_alignment" / "checkpoint_grid.json"
    )
    grid = _load(grid_path)
    if grid:
        ordered = [rows[g["label"]] for g in grid if g["label"] in rows]
        if ordered:
            return ordered
    return [r for r in summary.get("trajectory", []) if "error" not in r]


def _divider(ax, groups):
    """Vertical pre/post-training divider + region labels; returns boundary index."""
    boundary = next((i for i in range(1, len(groups)) if groups[i] != groups[i - 1]), None)
    if boundary is not None:
        ax.axvline(boundary - 0.5, color="0.4", ls="--", lw=1)
    return boundary


def main() -> None:
    ap = argparse.ArgumentParser(description="Paper 6 moral-dependency trajectory figures.")
    ap.add_argument("--summary",
                    default="papers/6_ablation_resistance/outputs/dependency/dependency_summary.json")
    ap.add_argument("--grid", default=None, help="Override grid path for state ordering.")
    ap.add_argument("--figures-dir", default="papers/6_ablation_resistance/outputs/figures")
    ap.add_argument("--overlay-summary", default=None,
                    help="Second summary to overlay on the trajectory (e.g. per-state).")
    ap.add_argument("--label", default="base directions (transfer)",
                    help="Legend label for the primary summary.")
    ap.add_argument("--overlay-label", default="per-state directions",
                    help="Legend label for the overlay summary.")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    summary_path = Path(args.summary)
    summary = _load(summary_path)
    if not summary:
        raise SystemExit(f"No summary at {summary_path}; run the dependency sweep first.")

    states = order_states(summary, summary_path, args.grid)
    if not states:
        raise SystemExit("No successful states in the summary to plot.")

    figdir = Path(args.figures_dir)
    figdir.mkdir(parents=True, exist_ok=True)

    x = list(range(len(states)))
    xlab = [short_label(r["label"]) for r in states]
    groups = [stage_group(r["label"]) for r in states]
    score = [r["moral_dependency_score"] for r in states]
    dmoral = [r["delta_ce_moral"] for r in states]
    dneutral = [r["delta_ce_neutral"] for r in states]

    kind = summary.get("direction_kind", "probe")
    n_pairs = summary.get("dataset", {}).get("n_pairs", "?")

    # Optional overlay (e.g. per-state self-dependency vs base-transfer), aligned by label.
    overlay = _load(Path(args.overlay_summary)) if args.overlay_summary else None
    overlay_score = None
    if overlay:
        omap = {r["label"]: r["moral_dependency_score"]
                for r in overlay.get("trajectory", []) if "error" not in r}
        overlay_score = [omap.get(r["label"], float("nan")) for r in states]

    # ---- Figure 1: dependency-score trajectory ----
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axhline(0, color=C["zero"], lw=1, ls=":")
    primary_label = args.label if overlay else "moral dependency = ΔCE(moral) − ΔCE(neutral)"
    ax.plot(x, score, "o-", color=C["score"], lw=1.8, ms=5, label=primary_label)
    if overlay_score is not None:
        ax.plot(x, overlay_score, "s--", color="#6A1B9A", lw=1.8, ms=5,
                label=args.overlay_label)
    ytop = max(v for v in (score + (overlay_score or [])) if v == v)
    b = _divider(ax, groups)
    if b is not None:
        ax.text(b / 2 - 0.5, ytop + 0.002, "pre-training", ha="center",
                fontsize=9, color="0.35")
        ax.text((b + len(x)) / 2 - 0.5, ytop + 0.002, "post-training",
                ha="center", fontsize=9, color="0.35")
    ax.set_xticks(x); ax.set_xticklabels(xlab, rotation=60, ha="right")
    ax.set_ylabel("moral dependency (nats/token)")
    ax.set_title(f"Natural moral dependency across the OLMo-3 pipeline "
                 f"({kind} directions, {n_pairs} pairs)")
    ax.legend(loc="best", framealpha=0.9)
    _save(fig, figdir, "dependency_trajectory")
    plt.close(fig)

    # ---- Figure 2: the two ablation-damage arms ----
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(x, dmoral, "s-", color=C["moral"], lw=1.8, ms=4,
            label="ΔCE moral text (ablated − clean)")
    ax.plot(x, dneutral, "^-", color=C["neutral"], lw=1.8, ms=4,
            label="ΔCE neutral text (ablated − clean)")
    _divider(ax, groups)
    ax.set_xticks(x); ax.set_xticklabels(xlab, rotation=60, ha="right")
    ax.set_ylabel("ΔCE under moral-subspace ablation (nats/token)")
    ax.set_title("Ablation damage by text type (gap = moral dependency)")
    ax.legend(loc="best", framealpha=0.9)
    _save(fig, figdir, "dependency_components")
    plt.close(fig)

    print(f"Wrote figures to {figdir}:")
    print("  dependency_trajectory.{png,pdf}")
    print("  dependency_components.{png,pdf}")
    print(f"\n  {'state':12s} {'score':>9s} {'Δmoral':>9s} {'Δneutral':>9s}")
    for r, lab in zip(states, xlab):
        print(f"  {lab:12s} {r['moral_dependency_score']:+9.4f} "
              f"{r['delta_ce_moral']:+9.4f} {r['delta_ce_neutral']:+9.4f}")


if __name__ == "__main__":
    main()
