#!/usr/bin/env python3
"""Phase D Step 2: produce summary plots for the steering experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CONDITION_ORDER = [
    "c10_baseline",
    "none",
    "gradient_penalty",
    "activation_patch",
]
CONDITION_LABELS = {
    "c10_baseline": "Baseline\n(OLMo-2 1B,\nno LoRA)",
    "none": "Vanilla LoRA\n(persona-voice\ncorpus)",
    "gradient_penalty": "+ gradient\npenalty\n(λ = 0.05)",
    "activation_patch": "+ activation\npatch\n(γ = 1.5)",
}
CONDITION_COLORS = {
    "c10_baseline": "#888888",
    "none": "#D32F2F",
    "gradient_penalty": "#1976D2",
    "activation_patch": "#2E7D32",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("papers/persona_monitoring/outputs/phase_d/step2_steering"))
    args = parser.parse_args()

    with open(args.input_dir / "analysis.json") as fh:
        a = json.load(fh)

    per = a["per_condition"]
    deltas = a["deltas"]
    verdicts = a["verdicts"]

    conditions = [c for c in CONDITION_ORDER if c in per]
    means = [per[c]["mean"] for c in conditions]
    stds = [per[c]["std"] for c in conditions]
    ns = [per[c]["n"] for c in conditions]
    sems = [s / np.sqrt(n) if n > 1 else float("nan")
            for s, n in zip(stds, ns)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: probe activation per condition ---
    ax = axes[0]
    x = np.arange(len(conditions))
    bars = ax.bar(x, means, yerr=sems, capsize=6,
                  color=[CONDITION_COLORS[c] for c in conditions],
                  edgecolor="black", linewidth=1.2)
    # Faint SD whiskers
    ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                alpha=0.25, capsize=3)

    for bar, m, n in zip(bars, means, ns):
        ax.text(bar.get_x() + bar.get_width() / 2,
                m + (0.1 if m >= 0 else -0.3),
                f"{m:+.2f}", ha="center", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions], fontsize=9)
    ax.set_ylabel("Probe activation (response-only logit)")
    ax.set_title(
        "Persona-probe activation on Betley benign prompts\n"
        "(higher ⇒ more persona-voice)",
        fontsize=11, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Annotate "vanilla LoRA shift" arrow if both baseline and none exist.
    if "c10_baseline" in per and "none" in per:
        ax.annotate(
            "", xy=(1, per["none"]["mean"]),
            xytext=(0, per["c10_baseline"]["mean"]),
            arrowprops=dict(arrowstyle="<->", color="#444",
                            linewidth=1.2),
        )
        mid_y = (per["none"]["mean"] + per["c10_baseline"]["mean"]) / 2
        d = deltas.get("none_vs_baseline", {})
        ax.text(0.5, mid_y, f"Δ={d.get('mean_delta', float('nan')):+.2f}\n"
                             f"d={d.get('cohens_d', float('nan')):+.2f}",
                ha="center", fontsize=8, color="#444",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888"))

    # --- Right: training loss curves per condition ---
    ax = axes[1]
    color_map = {
        "none": CONDITION_COLORS["none"],
        "gradient_penalty": CONDITION_COLORS["gradient_penalty"],
        "activation_patch": CONDITION_COLORS["activation_patch"],
    }
    for cond in ["none", "gradient_penalty", "activation_patch"]:
        trace_path = args.input_dir / cond / "lora_training.json"
        if not trace_path.exists():
            continue
        with open(trace_path) as fh:
            trace = json.load(fh)
        steps = trace.get("steps", [])
        if not steps:
            continue
        xs = [s["step"] for s in steps]
        ys = [s["loss"] for s in steps]
        ax.plot(xs, ys, "-", color=color_map[cond], linewidth=1.5,
                alpha=0.85, label=f"{cond} (final {ys[-1]:.3f})")
        # Aux loss overlay if present.
        aux = [s.get("aux_loss") for s in steps]
        if any(v is not None for v in aux):
            ax_aux_xs = [s["step"] for s in steps
                          if s.get("aux_loss") is not None]
            ax_aux_ys = [s["aux_loss"] for s in steps
                          if s.get("aux_loss") is not None]
            ax.plot(ax_aux_xs, ax_aux_ys, "--",
                    color=color_map[cond], linewidth=1, alpha=0.6,
                    label=f"{cond} aux (final {ax_aux_ys[-1]:.3f})")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("LoRA training loss\n(solid = SFT, dashed = aux)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        "Phase D Step 2 — Training-time steering on persona-voice "
        "positive-control corpus (OLMo-2 1B)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = args.input_dir / "step2_summary.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
