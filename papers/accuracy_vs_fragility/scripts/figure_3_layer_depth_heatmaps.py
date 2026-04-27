#!/usr/bin/env python3
"""Generate Figure 3 — layer-depth heatmaps over training.

Two stacked heatmaps with shared layer (y) and step (x) axes.

Top: probing accuracy (16 layers x 37 checkpoints).  Uniformly high
across layers within the first few thousand steps; the accuracy
heatmap is a visual of the saturation problem --- the colormap goes
flat after step 4K, with little structure to see.

Bottom: critical noise (16 layers x 37 checkpoints).  A layer-depth
robustness gradient develops over training that the accuracy heatmap
does not show: late layers hold maximum noise tolerance throughout
while early-layer critical noise drops monotonically.

Sources:
    outputs/phase_c1/step_*/layer_wise_moral_probe_*.json
    outputs/phase_c1/step_*/moral_fragility_test_*.json

Outputs:
    figures/figure_3_layer_depth_heatmaps.{pdf,png}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ID = "allenai/OLMo-2-0425-1B-early-training"

PAPER_DIR = Path("papers/accuracy_vs_fragility")
PHASE_C1 = PAPER_DIR / "outputs/phase_c1"
PROBE_GLOB = "layer_wise_moral_probe_*.json"
FRAG_GLOB = "moral_fragility_test_*.json"

# Map None / below-threshold critical noise to the maximum sweep value
# so the heatmap renders a continuous color rather than a missing cell.
# This matches the convention `MoralFragilityTest` uses internally
# (no σ in the sweep brings probe below threshold => critical_noise = 10.0).
CRIT_FALLBACK = 10.0

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_3_layer_depth_heatmaps.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_3_layer_depth_heatmaps.png"


def step_from_dirname(name: str) -> int | None:
    m = re.match(r"step_(\d+)$", name)
    return int(m.group(1)) if m else None


def collect_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, accuracy_grid, critical_noise_grid)
    where each grid is shape (n_layers, n_steps).
    """
    steps: list[int] = []
    acc_cols: list[list[float]] = []
    crit_cols: list[list[float]] = []
    n_layers = None

    step_dirs = sorted(
        d for d in PHASE_C1.iterdir()
        if d.is_dir() and step_from_dirname(d.name) is not None
    )
    for d in step_dirs:
        step = step_from_dirname(d.name)
        probe_files = list(d.glob(PROBE_GLOB))
        frag_files = list(d.glob(FRAG_GLOB))
        if not probe_files or not frag_files:
            continue
        with open(probe_files[0]) as f:
            probe = json.load(f)
        with open(frag_files[0]) as f:
            frag = json.load(f)
        accs = [s["accuracy"] for s in probe["layer_scores"]]
        crits = [
            (CRIT_FALLBACK if s["critical_noise"] is None else s["critical_noise"])
            for s in frag["layer_scores"]
        ]
        if n_layers is None:
            n_layers = len(accs)
        if len(accs) != n_layers or len(crits) != n_layers:
            continue
        steps.append(step)
        acc_cols.append(accs)
        crit_cols.append(crits)

    order = np.argsort(steps)
    steps_arr = np.array(steps)[order]
    acc_grid = np.array(acc_cols)[order].T   # (n_layers, n_steps)
    crit_grid = np.array(crit_cols)[order].T
    return steps_arr, acc_grid, crit_grid


def main() -> None:
    steps, acc_grid, crit_grid = collect_grids()
    n_layers = acc_grid.shape[0]

    fig, (ax_acc, ax_frag) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.18},
    )

    # --- Top: probing accuracy heatmap -------------------------------
    im_acc = ax_acc.pcolormesh(
        steps, np.arange(n_layers), acc_grid,
        cmap="viridis", vmin=0.5, vmax=1.0, shading="auto",
    )
    ax_acc.set_ylabel("Transformer layer", fontsize=11)
    ax_acc.set_title(
        "Figure 3: Layer-depth structure over training --- "
        f"{REPO_ID}",
        fontsize=11, loc="left",
    )
    ax_acc.text(
        0.01, 0.92, "(a) probing accuracy",
        transform=ax_acc.transAxes,
        fontsize=10, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )
    cbar_acc = fig.colorbar(im_acc, ax=ax_acc, pad=0.02)
    cbar_acc.set_label("Probe accuracy", fontsize=9)

    # --- Bottom: critical noise heatmap ------------------------------
    im_frag = ax_frag.pcolormesh(
        steps, np.arange(n_layers), crit_grid,
        cmap="plasma_r", vmin=0.0, vmax=10.0, shading="auto",
    )
    ax_frag.set_xlabel("Training step", fontsize=11)
    ax_frag.set_ylabel("Transformer layer", fontsize=11)
    ax_frag.text(
        0.01, 0.92, "(b) critical noise (fragility)",
        transform=ax_frag.transAxes,
        fontsize=10, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.5),
    )
    cbar_frag = fig.colorbar(im_frag, ax=ax_frag, pad=0.02)
    cbar_frag.set_label("Critical noise (σ; lower = more fragile)", fontsize=9)

    fig.tight_layout()
    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)

    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    print(f"  accuracy grid: {acc_grid.shape}, range {acc_grid.min():.3f}-{acc_grid.max():.3f}")
    print(f"  crit-noise grid: {crit_grid.shape}, range {crit_grid.min():.2f}-{crit_grid.max():.2f}")
    print(f"  early-layer (layer 0) crit noise step 1K -> 36K: "
          f"{crit_grid[0, np.where(steps == 1000)[0][0]]:.2f} -> "
          f"{crit_grid[0, -1]:.2f}")
    print(f"  late-layer (layer {n_layers - 1}) crit noise step 1K -> 36K: "
          f"{crit_grid[-1, np.where(steps == 1000)[0][0]]:.2f} -> "
          f"{crit_grid[-1, -1]:.2f}")


if __name__ == "__main__":
    main()
