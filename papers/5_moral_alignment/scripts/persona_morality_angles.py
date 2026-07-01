#!/usr/bin/env python3
"""Sprint 0.6: persona-morality angle measurement.

Loads moral foundation directions (exp1_probe_directions.npz) and persona
directions (persona_directions.npz), and for each layer computes the cosine
similarity between the persona direction and each of the 6 moral foundation
directions. Emits a 6xn_layers matrix as JSON + a heatmap.

Operates on saved directions only; no model loading. In a base model, cosines
are expected to be low (persona and morality weakly related before alignment).

Usage:
    python papers/5_moral_alignment/scripts/persona_morality_angles.py \
        --moral-npz papers/5_moral_alignment/outputs/olmo3_base/exp1_probe_directions.npz \
        --persona-npz papers/5_moral_alignment/outputs/olmo3_base/persona_directions.npz \
        --output-dir papers/5_moral_alignment/outputs/olmo3_base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepsteer.directions import extraction as du  # noqa: E402

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT  # noqa: E402


def compute_angles(
    moral: dict[str, dict[int, np.ndarray]],
    persona: dict[int, np.ndarray],
) -> tuple[list[int], np.ndarray]:
    """Return (layers, matrix) where matrix is (6, n_layers) of cosines."""
    foundations = [f for f in FOUNDATION_ORDER if f in moral]
    layers = sorted(set(persona) & set.intersection(*(set(moral[f]) for f in foundations)))
    mat = np.full((len(foundations), len(layers)), np.nan)
    for fi, f in enumerate(foundations):
        for li, L in enumerate(layers):
            mat[fi, li] = du.cosine(persona[L], moral[f][L])
    return layers, mat


def main() -> None:
    ap = argparse.ArgumentParser(description="Persona-morality angle measurement.")
    ap.add_argument("--moral-npz", required=True,
                    help="exp1_probe_directions.npz (foundation probe directions).")
    ap.add_argument("--persona-npz", required=True,
                    help="persona_directions.npz (key 'persona').")
    ap.add_argument("--persona-key", default="persona",
                    help="Which persona direction to use (persona | persona_meandiff).")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label", default=None, help="Title/label for the figure.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    label = args.label or out.name

    moral = du.load_directions(args.moral_npz)
    persona_all = du.load_directions(args.persona_npz)
    if args.persona_key not in persona_all:
        raise KeyError(
            f"{args.persona_key!r} not in {args.persona_npz} (have {list(persona_all)})"
        )
    persona = persona_all[args.persona_key]

    foundations = [f for f in FOUNDATION_ORDER if f in moral]
    layers, mat = compute_angles(moral, persona)

    # ---- JSON ----
    result = {
        "analysis": "persona_morality_angles",
        "label": label,
        "moral_npz": args.moral_npz,
        "persona_npz": args.persona_npz,
        "persona_key": args.persona_key,
        "foundations": foundations,
        "layers": layers,
        "full_attention_layers": du.OLMO3_FULL_ATTENTION_LAYERS,
        "cosine_matrix": mat.tolist(),  # rows = foundations, cols = layers
        "per_foundation_mean_abs_cosine": {
            f: round(float(np.nanmean(np.abs(mat[fi]))), 4)
            for fi, f in enumerate(foundations)
        },
        "overall_mean_abs_cosine": round(float(np.nanmean(np.abs(mat))), 4),
        "max_abs_cosine": round(float(np.nanmax(np.abs(mat))), 4),
    }
    with open(out / "persona_morality_angles.json", "w") as f:
        json.dump(result, f, indent=2)

    # ---- heatmap ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, len(layers) * 0.35), 4.5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(foundations)))
    ax.set_yticklabels([FOUNDATION_SHORT[f] for f in foundations])
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=8)
    ax.set_xlabel("Layer")
    # Flag full-attention layers (OLMo-3 hybrid attention).
    full = set(du.OLMO3_FULL_ATTENTION_LAYERS)
    for li, L in enumerate(layers):
        if L in full:
            ax.get_xticklabels()[li].set_color("#1E88E5")
            ax.get_xticklabels()[li].set_fontweight("bold")
    ax.set_title(
        f"Persona-Morality Direction Cosine ({label})\n"
        f"mean |cos| = {result['overall_mean_abs_cosine']:.3f}; "
        f"blue x-labels = full-attention layers",
        fontsize=11,
    )
    fig.colorbar(im, ax=ax, shrink=0.8, label="cosine similarity")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"persona_morality_angles.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out/'persona_morality_angles.json'} and heatmap "
          f"({len(foundations)} foundations x {len(layers)} layers)")
    print(f"  overall mean |cos| = {result['overall_mean_abs_cosine']:.3f}, "
          f"max |cos| = {result['max_abs_cosine']:.3f}")


if __name__ == "__main__":
    main()
