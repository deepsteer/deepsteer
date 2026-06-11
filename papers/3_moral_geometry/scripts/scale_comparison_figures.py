#!/usr/bin/env python3
"""Scale-comparison figures: OLMo-2 1B vs 7B framework geometry (Extension A.5).

Operates entirely on existing JSON/npz outputs (no model loading). Each figure
guards for missing inputs and skips with a log message, so the script runs
incrementally as 7B results land:

  - 1B: outputs/exp1_2_3/, outputs/exp7_fragility/
  - 7B: outputs/exp1_2_3_7B/, outputs/exp7_fragility_7B/  (fragility from RunPod)

Figures (saved to outputs/figures/ as both .png and .pdf, with a 1:1 JSON
sidecar):
  1. scale_comparison_cosine_heatmap  - cosine matrices at matched layers
  2. scale_comparison_geometry_overlay - mean cosine + eff. dim vs normalized depth
  3. scale_comparison_dendrogram       - dendrograms at matched relative layers
  4. scale_comparison_fragility        - per-foundation critical noise, 1B vs 7B
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exp1_2_3_framework_geometry import (  # noqa: E402
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING,
)

logger = logging.getLogger(__name__)

COLOR_1B = "#1E88E5"
COLOR_7B = "#E53935"


def _load_geometry(exp_dir: Path) -> dict | None:
    """Load exp2 framework-geometry JSON into per-layer arrays keyed by int."""
    path = exp_dir / "exp2_framework_geometry.json"
    if not path.exists():
        logger.warning("Geometry JSON missing: %s", path)
        return None
    with open(path) as f:
        d = json.load(f)

    # exp2 geometry json doesn't record the model; fall back to exp1 json.
    model = d.get("model")
    if not model:
        exp1 = exp_dir / "exp1_foundation_probing.json"
        if exp1.exists():
            with open(exp1) as f:
                model = json.load(f).get("model")

    per_layer = d["per_layer"]
    mean_cosine: dict[int, float] = {}
    eff_dim: dict[int, float] = {}
    cos_matrices: dict[int, np.ndarray] = {}
    for k, v in per_layer.items():
        layer = int(k)
        if "mean_cosine_similarity" in v:
            mean_cosine[layer] = v["mean_cosine_similarity"]
        if "effective_dimensionality" in v:
            eff_dim[layer] = v["effective_dimensionality"]
        if "cosine_similarity_matrix" in v:
            cos_matrices[layer] = np.array(v["cosine_similarity_matrix"])

    return {
        "model": model,
        "n_layers": d["n_layers"],
        "foundations_present": d["foundations_present"],
        "mean_cosine": mean_cosine,
        "eff_dim": eff_dim,
        "cos_matrices": cos_matrices,
        "peak_separation_layer": d.get("peak_separation_layer"),
    }


def _load_fragility(exp_dir: Path) -> dict | None:
    """Load exp7 dense-OLMo fragility JSON -> {foundation: mean_critical_noise}."""
    path = exp_dir / "exp7_olmo_fragility.json"
    if not path.exists():
        logger.warning("Fragility JSON missing: %s", path)
        return None
    with open(path) as f:
        d = json.load(f)
    out = {"model": d.get("model"), "per_foundation": {}}
    for fv, fdata in d["per_foundation"].items():
        out["per_foundation"][fv] = fdata.get("mean_critical_noise")
    return out


def _pick_layer(geo: dict, requested: int | None) -> int | None:
    """Resolve a display layer, falling back to a mid-depth layer if needed."""
    if geo is None:
        return None
    mats = geo["cos_matrices"]
    if not mats:
        return None
    if requested is not None and requested in mats:
        return requested
    # Fall back to the layer nearest 45% relative depth (matched across scales).
    target = int(round(0.45 * (geo["n_layers"] - 1)))
    return min(mats.keys(), key=lambda l: abs(l - target))


# ---------------------------------------------------------------------------
# Figure 1: cosine heatmap pair
# ---------------------------------------------------------------------------


def fig_cosine_heatmap(geo_1b, geo_7b, layer_1b, layer_7b, figures_dir):
    panels = []
    for geo, layer, scale in ((geo_1b, layer_1b, "1B"), (geo_7b, layer_7b, "7B")):
        if geo is not None and layer is not None and layer in geo["cos_matrices"]:
            panels.append((geo, layer, scale))
    if not panels:
        logger.warning("Figure 1 skipped: no cosine matrices available.")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7), squeeze=False)
    for ax, (geo, layer, scale) in zip(axes[0], panels):
        cos = geo["cos_matrices"][layer]
        fps = geo["foundations_present"]
        n = len(fps)
        labels = [FOUNDATION_SHORT[f] for f in fps]
        im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        for i in range(n):
            for j in range(n):
                val = cos[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if abs(val) > 0.6 else "black")
        if n == 6:
            ax.axhline(y=2.5, color="black", linewidth=2)
            ax.axvline(x=2.5, color="black", linewidth=2)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
        ax.set_title(
            f"OLMo-2 {scale} (layer {layer}/{geo['n_layers']})\n"
            f"mean pairwise = {geo['mean_cosine'][layer]:.3f}",
            fontsize=12, fontweight="bold",
        )
    fig.suptitle("Foundation Direction Cosine Similarity Across Scale",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, figures_dir, "scale_comparison_cosine_heatmap")
    return {"panels": [{"scale": s, "layer": l, "n_layers": g["n_layers"],
                        "mean_cosine": g["mean_cosine"][l]} for g, l, s in panels]}


# ---------------------------------------------------------------------------
# Figure 2: layer-wise geometry overlay (normalized depth)
# ---------------------------------------------------------------------------


def fig_geometry_overlay(geo_1b, geo_7b, figures_dir):
    series = [(geo_1b, "1B", COLOR_1B, "o-"), (geo_7b, "7B", COLOR_7B, "s-")]
    series = [s for s in series if s[0] is not None]
    if not series:
        logger.warning("Figure 2 skipped: no geometry available.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sidecar = {}
    for geo, scale, color, style in series:
        denom = max(geo["n_layers"] - 1, 1)
        mc_layers = sorted(geo["mean_cosine"])
        ed_layers = sorted(geo["eff_dim"])
        mc_depth = [l / denom for l in mc_layers]
        ed_depth = [l / denom for l in ed_layers]
        axes[0].plot(mc_depth, [geo["mean_cosine"][l] for l in mc_layers],
                     style, color=color, linewidth=2, markersize=4, label=f"OLMo-2 {scale}")
        axes[1].plot(ed_depth, [geo["eff_dim"][l] for l in ed_layers],
                     style, color=color, linewidth=2, markersize=4, label=f"OLMo-2 {scale}")
        sidecar[scale] = {
            "normalized_depth": mc_depth,
            "mean_cosine": [geo["mean_cosine"][l] for l in mc_layers],
            "effective_dim": [geo["eff_dim"][l] for l in ed_layers],
        }

    axes[0].axhspan(0.2, 0.4, color="gray", alpha=0.12, label="integration range")
    axes[0].set_xlabel("Normalized layer depth", fontsize=11)
    axes[0].set_ylabel("Mean Pairwise Cosine Similarity", fontsize=11)
    axes[0].set_title("(a) Collapse Metric", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    axes[1].set_xlabel("Normalized layer depth", fontsize=11)
    axes[1].set_ylabel("Effective Dimensionality (90% var)", fontsize=11)
    axes[1].set_title("(b) Direction Set Dimensionality", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.5, 6.5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle("Layer-Wise Geometry Across Scale (normalized depth)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, figures_dir, "scale_comparison_geometry_overlay")
    return sidecar


# ---------------------------------------------------------------------------
# Figure 3: dendrogram pair at matched relative layers
# ---------------------------------------------------------------------------


def fig_dendrogram(geo_1b, geo_7b, layer_1b, layer_7b, figures_dir):
    from scipy.cluster.hierarchy import dendrogram, linkage

    panels = []
    for geo, layer, scale in ((geo_1b, layer_1b, "1B"), (geo_7b, layer_7b, "7B")):
        if geo is not None and layer is not None and layer in geo["cos_matrices"]:
            panels.append((geo, layer, scale))
    if not panels:
        logger.warning("Figure 3 skipped: no cosine matrices available.")
        return None

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 5), squeeze=False)
    sidecar = []
    for ax, (geo, layer, scale) in zip(axes[0], panels):
        cos = geo["cos_matrices"][layer]
        fps = geo["foundations_present"]
        n = len(fps)
        labels = [FOUNDATION_SHORT[f] for f in fps]
        dist = 1 - cos
        condensed = np.array([dist[i, j] for i in range(n) for j in range(i + 1, n)])
        Z = linkage(condensed, method="ward")
        dn = dendrogram(Z, labels=labels, ax=ax, leaf_font_size=11,
                        color_threshold=0, above_threshold_color="#666")
        short_to_fv = {FOUNDATION_SHORT[fv]: fv for fv in fps}
        for lbl in ax.get_xticklabels():
            fv = short_to_fv.get(lbl.get_text())
            lbl.set_color("#43A047" if fv in INDIVIDUALIZING else "#FB8C00")
            lbl.set_fontweight("bold")
        ax.set_ylabel("Ward Distance (1 - cosine)", fontsize=11)
        ax.set_title(f"OLMo-2 {scale} (layer {layer}/{geo['n_layers']})",
                     fontsize=12, fontweight="bold")
        sidecar.append({"scale": scale, "layer": layer, "leaf_order": dn["ivl"]})

    fig.suptitle("Foundation Clustering Across Scale (green = individualizing, orange = binding)",
                 fontsize=12, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, figures_dir, "scale_comparison_dendrogram")
    return sidecar


# ---------------------------------------------------------------------------
# Figure 4: per-foundation fragility comparison
# ---------------------------------------------------------------------------


def fig_fragility(frag_1b, frag_7b, figures_dir):
    scales = []
    if frag_1b is not None:
        scales.append(("1B", COLOR_1B, frag_1b["per_foundation"]))
    if frag_7b is not None:
        scales.append(("7B", COLOR_7B, frag_7b["per_foundation"]))
    if not scales:
        logger.warning("Figure 4 skipped: no fragility data available.")
        return None
    missing_7b = frag_7b is None

    foundations = [f for f in FOUNDATION_ORDER]
    x = np.arange(len(foundations))
    width = 0.8 / len(scales)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sidecar = {}
    for i, (scale, color, perf) in enumerate(scales):
        heights = [perf.get(f) if perf.get(f) is not None else 0.0 for f in foundations]
        offsets = x + (i - (len(scales) - 1) / 2) * width
        bars = ax.bar(offsets, heights, width, label=f"OLMo-2 {scale}", color=color,
                      edgecolor="black", linewidth=0.5)
        # Highlight sanctity (the 1B anomaly) with a hatch.
        for f, bar in zip(foundations, bars):
            if f == "sanctity_degradation":
                bar.set_hatch("///")
        sidecar[scale] = {f: perf.get(f) for f in foundations}

    ax.set_xticks(x)
    ax.set_xticklabels([FOUNDATION_SHORT[f] for f in foundations], rotation=30, ha="right")
    ax.set_ylabel("Mean Critical Noise (higher = more robust)", fontsize=11)
    title = "Per-Foundation Fragility Across Scale (sanctity hatched)"
    if missing_7b:
        title += "\n[7B pending RunPod run]"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, figures_dir, "scale_comparison_fragility")
    return sidecar


def _save(fig, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figures_dir / stem}.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="1B vs 7B scale-comparison figures.")
    parser.add_argument("--dir-1b", default="papers/3_moral_geometry/outputs/exp1_2_3")
    parser.add_argument("--dir-7b", default="papers/3_moral_geometry/outputs/exp1_2_3_7B")
    parser.add_argument("--fragility-1b", default="papers/3_moral_geometry/outputs/exp7_fragility")
    parser.add_argument("--fragility-7b", default="papers/3_moral_geometry/outputs/exp7_fragility_7B")
    parser.add_argument("--figures-dir", default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--layer-1b", type=int, default=7,
                        help="Display layer for 1B heatmap/dendrogram (default 7/16).")
    parser.add_argument("--layer-7b", type=int, default=14,
                        help="Display layer for 7B heatmap/dendrogram (default 14/32).")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    geo_1b = _load_geometry(Path(args.dir_1b))
    geo_7b = _load_geometry(Path(args.dir_7b))
    frag_1b = _load_fragility(Path(args.fragility_1b))
    frag_7b = _load_fragility(Path(args.fragility_7b))

    layer_1b = _pick_layer(geo_1b, args.layer_1b)
    layer_7b = _pick_layer(geo_7b, args.layer_7b)

    print("Generating scale-comparison figures...")
    sidecar = {
        "experiment": "scale_comparison_figures",
        "models": {"1B": geo_1b and geo_1b.get("model"), "7B": geo_7b and geo_7b.get("model")},
        "display_layers": {"1B": layer_1b, "7B": layer_7b},
        "cosine_heatmap": fig_cosine_heatmap(geo_1b, geo_7b, layer_1b, layer_7b, figures_dir),
        "geometry_overlay": fig_geometry_overlay(geo_1b, geo_7b, figures_dir),
        "dendrogram": fig_dendrogram(geo_1b, geo_7b, layer_1b, layer_7b, figures_dir),
        "fragility": fig_fragility(frag_1b, frag_7b, figures_dir),
    }

    out_json = figures_dir / "scale_comparison_data.json"
    with open(out_json, "w") as f:
        json.dump(sidecar, f, indent=2, default=str)
    print(f"\nData sidecar: {out_json}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
