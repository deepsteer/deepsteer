#!/usr/bin/env python3
"""Figure 5 — when and where compositional moral encoding emerges.

Reads the per-checkpoint records from `traj_compositional_encoding.py`
(b_traj/step_*.json) and plots two panels on the OLMo-2 1B early-training
trajectory:

  (a) WHEN: leave-construction-out transfer accuracy over training, against
      the bag-of-words transfer floor and chance. Transfer rising from chance
      to plateau is compositional moral encoding emerging (a probe reading
      contrast-token identity cannot transfer across constructions). The
      role_reversal curve (lexical cues scrambled by design) is the clincher.

  (b) WHERE: per-layer in-distribution probe accuracy as a layer x step
      heatmap, showing where in the network the encoding concentrates and how
      that develops across training.

Robust to partial runs: plots whatever step files exist.

Outputs:
    figures/figure_5_compositional_emergence.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path("papers/1_accuracy_vs_fragility")
TRAJ_DIR = PAPER_DIR / "outputs/phase_c4_compositional/b_traj"
PAPER_FIG_PDF = PAPER_DIR / "figures/figure_5_compositional_emergence.pdf"
PAPER_FIG_PNG = PAPER_DIR / "figures/figure_5_compositional_emergence.png"

COMPOSITIONAL = "#9C27B0"
BASELINE = "#F44336"


def load_records() -> list[dict]:
    recs = [json.loads(f.read_text()) for f in TRAJ_DIR.glob("step_*.json")]
    recs.sort(key=lambda r: r["step"])
    return recs


def main() -> None:
    recs = load_records()
    if not recs:
        raise SystemExit(f"no step files in {TRAJ_DIR}")
    steps = np.array([r["step"] for r in recs])
    transfer = np.array([r["transfer_mean_best"] for r in recs])
    role_transfer = np.array([r["transfer"]["role_reversal"]["best_acc"] for r in recs])
    # Bag-of-words transfer floor: mean per-construction lexical floor (text-only,
    # checkpoint-independent), ~0.60.
    floor_vals = list(recs[0]["lexical_floor"].values())
    bow_floor = float(np.mean(floor_vals))

    fig, (ax_when, ax_where) = plt.subplots(
        1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1.0], "wspace": 0.26},
    )

    # --- (a) WHEN ----------------------------------------------------
    ax_when.plot(steps, transfer, "D-", color=COMPOSITIONAL, linewidth=2.4,
                 markersize=4, label="compositional transfer (mean over held-out constructions)")
    ax_when.plot(steps, role_transfer, "o--", color=COMPOSITIONAL, linewidth=1.6,
                 markersize=3, alpha=0.7, label="role_reversal transfer (lexical cues scrambled)")
    ax_when.axhline(bow_floor, color=BASELINE, linestyle=":", linewidth=1.3,
                    label=f"bag-of-words transfer floor (~{bow_floor:.2f})")
    ax_when.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=1, alpha=0.7,
                    label="chance (0.50)")
    ax_when.set_xlabel("Training step", fontsize=11)
    ax_when.set_ylabel("Leave-construction-out transfer accuracy", fontsize=11)
    ax_when.set_ylim(0.4, 1.0)
    ax_when.set_xlim(0, steps.max() * 1.02 if steps.max() else 1)
    ax_when.set_title("(a) When: compositional encoding emerges, then holds",
                      fontsize=10, loc="left")
    ax_when.legend(loc="lower right", fontsize=7.5)
    ax_when.grid(True, alpha=0.3)

    # --- (b) WHERE (layer x step heatmap) ----------------------------
    layer_keys = sorted(int(k) for k in recs[0]["indist_per_layer"])
    mat = np.array([
        [r["indist_per_layer"][str(ly)] for r in recs] for ly in layer_keys
    ])  # (n_layer, n_step)
    im = ax_where.imshow(mat, aspect="auto", origin="lower", cmap="viridis",
                         vmin=0.5, vmax=0.9,
                         extent=(steps.min(), steps.max(), layer_keys[0], layer_keys[-1]))
    ax_where.set_xlabel("Training step", fontsize=11)
    ax_where.set_ylabel("Transformer layer", fontsize=11)
    ax_where.set_title("(b) Where: per-layer in-distribution probe accuracy",
                       fontsize=10, loc="left")
    cbar = fig.colorbar(im, ax=ax_where, fraction=0.046, pad=0.04)
    cbar.set_label("probe accuracy", fontsize=9)

    n = len(recs)
    fig.suptitle(
        f"Compositional moral encoding emerges during pre-training "
        f"(OLMo-2 1B early-training, {n}/37 checkpoints)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    PAPER_FIG_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_FIG_PDF)
    fig.savefig(PAPER_FIG_PNG, dpi=200)
    plt.close(fig)
    print(f"wrote: {PAPER_FIG_PDF} ({n} checkpoints)")
    print(f"wrote: {PAPER_FIG_PNG}")
    # Quick emergence readout: first step crossing the bag-of-words floor + 0.1.
    thresh = bow_floor + 0.10
    crossed = steps[transfer >= thresh]
    if crossed.size:
        print(f"transfer first crosses {thresh:.2f} at step {int(crossed[0])}")


if __name__ == "__main__":
    main()
