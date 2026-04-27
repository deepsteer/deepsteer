#!/usr/bin/env python3
"""Generate Figure 2 — probing accuracy saturates; fragility resolves.

Two-panel figure on a shared training-step axis.

Top panel: mean probing accuracy across all 16 transformer layers over
the 37-checkpoint OLMo-2 1B early-training trajectory.  Saturates near
0.95 by step 4K and stays flat for the remaining 32K steps --- this
is the saturation problem the paper is built around.

Bottom panel: mean critical noise (the fragility scalar) across the
same checkpoints.  Continues evolving long after accuracy plateaus, with
the mean drifting from ~10 down toward ~3 between steps 4K and 36K.
This is what "fragility resolves" means visually.

Sources:
    outputs/phase_c1/step_*/layer_wise_moral_probe_*.json   (per-layer accuracy)
    outputs/phase_c1/step_*/moral_fragility_test_*.json     (per-layer critical noise)

Outputs:
    figures/figure_2_saturation_vs_fragility.{pdf,png}
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
SATURATION_THRESHOLD = 0.93  # matches §1 / §4.1 "reaches ~95% by step 4K" narrative

PAPER_FIG_PDF = PAPER_DIR / "figures/figure_2_saturation_vs_fragility.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_2_saturation_vs_fragility.png"


def step_from_dirname(name: str) -> int | None:
    m = re.match(r"step_(\d+)$", name)
    return int(m.group(1)) if m else None


def collect_trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (steps, mean_accuracy_per_step, mean_critical_noise_per_step).

    Mean critical noise treats `None` (probe never reached threshold) as
    missing and ignores the layer in the per-step mean --- matches the
    aggregation convention used throughout the paper.
    """
    steps: list[int] = []
    mean_acc: list[float] = []
    mean_crit: list[float] = []

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
        crit_raw = [s["critical_noise"] for s in frag["layer_scores"]]
        crit_clean = [c for c in crit_raw if c is not None]
        if not accs or not crit_clean:
            continue
        steps.append(step)
        mean_acc.append(float(np.mean(accs)))
        mean_crit.append(float(np.mean(crit_clean)))

    order = np.argsort(steps)
    return (
        np.array(steps)[order],
        np.array(mean_acc)[order],
        np.array(mean_crit)[order],
    )


def first_saturation_step(steps: np.ndarray, mean_acc: np.ndarray) -> int | None:
    """First step where mean accuracy reaches SATURATION_THRESHOLD."""
    for s, a in zip(steps, mean_acc):
        if a >= SATURATION_THRESHOLD:
            return int(s)
    return None


def main() -> None:
    steps, mean_acc, mean_crit = collect_trajectory()
    sat_step = first_saturation_step(steps, mean_acc)

    fig, (ax_acc, ax_frag) = plt.subplots(
        2, 1, figsize=(11, 7), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.12},
    )

    # --- Top: probing accuracy ---------------------------------------
    ax_acc.plot(
        steps, mean_acc, "o-", color="#F44336", linewidth=2, markersize=4,
        label="Mean probing accuracy",
    )
    ax_acc.axhline(
        y=SATURATION_THRESHOLD, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.7,
        label=f"saturation reference ({SATURATION_THRESHOLD:.0%})",
    )
    if sat_step is not None:
        ax_acc.axvline(x=sat_step, color="#F44336", linestyle="--", linewidth=1.2, alpha=0.6)
        ax_acc.text(
            sat_step, 0.55,
            f"saturates\nstep {sat_step // 1000}K",
            ha="center", va="bottom", fontsize=9, color="#F44336", fontweight="bold",
        )
        ax_acc.axvspan(sat_step, steps.max(), color="#FFEBEE", alpha=0.4, zorder=0)
        ax_acc.text(
            (sat_step + steps.max()) / 2, 0.46,
            f"{(steps.max() - sat_step) / 1000:.0f}K steps invisible to accuracy",
            ha="center", va="bottom", fontsize=9, color="#888", style="italic",
        )

    ax_acc.set_ylabel("Mean probing accuracy", fontsize=11)
    ax_acc.set_ylim(0.4, 1.05)
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_title(
        f"Figure 2: Probing accuracy saturates; fragility resolves --- {REPO_ID}",
        fontsize=11, loc="left",
    )

    # --- Bottom: mean critical noise ---------------------------------
    ax_frag.plot(
        steps, mean_crit, "s-", color="#3F51B5", linewidth=2, markersize=4,
        label="Mean critical noise (fragility)",
    )
    if sat_step is not None:
        ax_frag.axvline(x=sat_step, color="#F44336", linestyle="--", linewidth=1.2, alpha=0.4)
        ax_frag.axvspan(sat_step, steps.max(), color="#E8EAF6", alpha=0.4, zorder=0)
        ax_frag.text(
            (sat_step + steps.max()) / 2,
            mean_crit.max() * 0.95,
            f"fragility keeps resolving change ({mean_crit[steps >= sat_step].max():.1f} → {mean_crit[-1]:.1f})",
            ha="center", va="top", fontsize=9, color="#3F51B5", fontweight="bold",
        )

    ax_frag.set_xlabel("Training step", fontsize=11)
    ax_frag.set_ylabel("Mean critical noise (σ)", fontsize=11)
    ax_frag.set_xlim(0, steps.max() + 500)
    ax_frag.legend(loc="lower right", fontsize=9)
    ax_frag.grid(True, alpha=0.3)

    fig.tight_layout()
    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)

    print(f"wrote: {PAPER_FIG_PDF}")
    print(f"wrote: {PAPER_FIG_PNG}")
    print(f"  saturation step (≥{SATURATION_THRESHOLD:.0%}): {sat_step}")
    print(f"  mean accuracy at saturation: {mean_acc[steps == sat_step][0]:.3f}")
    print(f"  mean accuracy at step 36K:   {mean_acc[-1]:.3f}")
    print(f"  mean crit noise at step ~5K: {mean_crit[steps >= 5000][0]:.2f}")
    print(f"  mean crit noise at step 36K: {mean_crit[-1]:.2f}")


if __name__ == "__main__":
    main()
