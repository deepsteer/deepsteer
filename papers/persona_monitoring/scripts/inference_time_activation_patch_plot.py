#!/usr/bin/env python3
"""Plot the inference-time activation-patching dose-response curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path,
                        default=Path("papers/persona_monitoring/outputs/phase_d/c10_inference_patch"))
    args = parser.parse_args()

    with open(args.input_dir / "summary.json") as fh:
        d = json.load(fh)

    # Bucket cells by (prompt_set, condition).
    cells: dict[tuple[str, str], dict] = {}
    for c in d["cells"]:
        cells[(c["prompt_set"], c["condition"])] = c

    # Build the alpha axis: suppress is "projection" (mark separately),
    # steer_-8 / steer_-4 / baseline (alpha=0) / steer_+4 / steer_+8.
    alpha_conditions = [
        (-8.0, "steer_-8"),
        (-4.0, "steer_-4"),
        (0.0, "baseline"),
        (4.0, "steer_+4"),
        (8.0, "steer_+8"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    sets = ["persona_elicit", "betley_benign"]
    set_titles = {
        "persona_elicit": "Persona-elicit prompts (8 prompts × 3 samples)",
        "betley_benign": "Betley benign prompts (8 prompts × 3 samples)",
    }
    colors = {"persona_elicit": "#9C27B0", "betley_benign": "#1976D2"}

    for ax, set_name in zip(axes, sets):
        alphas = [a for a, _ in alpha_conditions]
        means = [cells[(set_name, c)]["mean"] for _, c in alpha_conditions]
        stds = [cells[(set_name, c)]["std"] for _, c in alpha_conditions]
        n = [cells[(set_name, c)]["n_samples"] for _, c in alpha_conditions]
        sems = [s / np.sqrt(ni) for s, ni in zip(stds, n)]

        ax.errorbar(alphas, means, yerr=sems, fmt="o-",
                    color=colors[set_name], linewidth=2, markersize=9,
                    capsize=5, label="mean ± SEM")
        # Show ±SD as faint error bars
        ax.errorbar(alphas, means, yerr=stds, fmt="none",
                    color=colors[set_name], alpha=0.3, capsize=3,
                    label="mean ± SD")

        # Mark suppress as a separate point (it's the "projection out"
        # variant, plotted alongside α=0 baseline for comparison).
        if (set_name, "suppress") in cells:
            sc = cells[(set_name, "suppress")]
            sem_sup = sc["std"] / np.sqrt(sc["n_samples"])
            ax.errorbar([0.0], [sc["mean"]], yerr=[sem_sup], fmt="s",
                        color="#666", markersize=8, capsize=4,
                        label="suppress (projection out)")

        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("steer α  (h += α × unit_w  at layers 5, 6, 7)")
        ax.set_ylabel("Probe logit (response-only)")
        ax.set_title(set_titles[set_name], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)

    fig.suptitle(
        "Phase D Step 1 — Inference-time activation patching: "
        "persona-direction dose-response (OLMo-2 1B base)",
        fontsize=12,
    )
    fig.tight_layout()
    out_path = args.input_dir / "dose_response.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
