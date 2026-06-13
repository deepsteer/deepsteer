"""Figure for §4.4: raw vs RMS-normalized fragility (the activation-scale confound).

Panel (a): mean critical noise over training, raw vs RMS-normalized — raw rises
then declines after accuracy saturates; RMS is flat.
Panel (b): late/early sigma* ratio over training, raw vs RMS — the raw layer-depth
gradient grows to ~7-15x; under RMS it stays ~2x.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAJ = Path("papers/1_accuracy_vs_fragility/outputs/phase_c1_refragility/trajectory.json")
OUT = Path("papers/1_accuracy_vs_fragility/figures")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    traj = json.load(open(TRAJ))["trajectory"]
    steps = sorted(int(s) for s in traj)
    raw_mean = [traj[str(s)]["standard"]["mean_critical_noise"] for s in steps]
    rms_mean = [traj[str(s)]["rms_normalized"]["mean_critical_noise"] for s in steps]
    raw_acc = [traj[str(s)]["standard"]["mean_acc"] for s in steps]

    def ratio(variant):
        out = []
        for s in steps:
            r = traj[str(s)][variant]
            out.append(r["late_crit"] / max(r["early_crit"], 0.01))
        return out

    raw_ratio = ratio("standard")
    rms_ratio = ratio("rms_normalized")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.plot(steps, raw_mean, "o-", color="#E53935", lw=2, ms=4, label="Raw $\\sigma^*$")
    ax1.plot(steps, rms_mean, "s--", color="#1E88E5", lw=2, ms=4, label="RMS-normalized $\\sigma^*$")
    ax1b = ax1.twinx()
    ax1b.plot(steps, raw_acc, ":", color="#9E9E9E", lw=1.5, label="Mean accuracy")
    ax1b.set_ylabel("Mean probing accuracy", color="#757575", fontsize=11)
    ax1b.set_ylim(0.4, 1.02)
    ax1.set_xlabel("Pre-training step", fontsize=11)
    ax1.set_ylabel("Mean critical noise $\\sigma^*$", fontsize=11)
    ax1.set_title("(a) Mean fragility: raw declines, RMS is flat", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, raw_ratio, "o-", color="#E53935", lw=2, ms=4, label="Raw")
    ax2.plot(steps, rms_ratio, "s--", color="#1E88E5", lw=2, ms=4, label="RMS-normalized")
    ax2.axhline(1.0, color="#9E9E9E", ls=":", lw=1)
    ax2.set_xlabel("Pre-training step", fontsize=11)
    ax2.set_ylabel("Late / early $\\sigma^*$ ratio", fontsize=11)
    ax2.set_title("(b) Layer-depth gradient: raw is scale, RMS ~flat", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Scale-normalized fragility (OLMo-2 1B): the layer-depth gradient is largely activation scale",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure_4_rms_control.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}/figure_4_rms_control.png")


if __name__ == "__main__":
    main()
