#!/usr/bin/env python3
"""External-dataset robustness check on Moral Foundations Vignettes (Extension E).

Tests whether the integration geometry (near-maximal effective dimensionality,
mean pairwise cosine in the 0.2-0.4 range) is a property of the model's
representations or an artifact of the DeepSteer probing dataset. Foundation
directions are computed from the MFV stimuli alone (Clifford et al. 2015,
curated in behavioral_benchmarking.MFV_ITEMS) and their geometry is compared to
the DeepSteer-derived geometry.

MFV has only violation vignettes per foundation (no neutral counterpart). To
make the geometry comparable to the DeepSteer moral-vs-neutral directions, each
foundation direction uses a *shared neutral baseline*: dir_F = normalize(
mean(MFV_F vignettes) - neutral_mean), where neutral_mean is the mean activation
of the DeepSteer v2 neutral sentences (non-moral domains: cooking, weather,
sports, ...). The moral signal comes entirely from the independently constructed
MFV stimuli; only the reference point is borrowed. This avoids the artifact of a
foundation-vs-rest contrast, whose directions are mean-centered and therefore
have a structurally negative mean pairwise cosine (~ -1/(k-1)) that carries no
integration signal.

Outputs:
  - outputs/external_robustness/mfv_geometry_<tag>.json
  - outputs/external_robustness/cross_dataset_alignment_<tag>.json
  - outputs/figures/external_robustness_<tag>.pdf
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT, INDIVIDUALIZING
from deepsteer.geometry import (
    compute_cosine_matrix,
    compute_effective_dimensionality,
    permutation_test_mft,
)

logger = logging.getLogger(__name__)

OLMO_REPO = "allenai/OLMo-2-0425-1B"


def _model_label(model_id: str) -> str:
    name = model_id.rstrip("/").split("/")[-1]
    return f"OLMo-2 {name.split('-')[-1]}" if "OLMo-2" in name else name


def _model_tag(model_id: str) -> str:
    name = model_id.rstrip("/").split("/")[-1]
    return name.split("-")[-1] if "OLMo-2" in name else name.replace("/", "_")


def _load_mfv(mfv_file: str | None) -> list[dict]:
    """Load MFV vignettes: an external JSON file if given, else the bundled set."""
    if mfv_file:
        with open(mfv_file) as f:
            items = json.load(f)
        logger.info("Loaded %d MFV items from %s", len(items), mfv_file)
        return items
    pe_dir = Path(__file__).resolve().parent / "probe_engineering"
    sys.path.insert(0, str(pe_dir))
    import behavioral_benchmarking as bb  # noqa: E402
    logger.info("Using bundled MFV_ITEMS (%d items)", len(bb.MFV_ITEMS))
    return bb.MFV_ITEMS


def compute_mfv_directions(acts: dict[int, "np.ndarray"], labels: np.ndarray,
                           layers: list[int],
                           neutral_mean: dict[int, np.ndarray]) -> dict[str, dict[int, np.ndarray]]:
    """Moral-vs-neutral direction per foundation per layer.

    dir_F = normalize(mean(MFV_F) - neutral_mean[layer]), matching the DeepSteer
    moral-vs-neutral contrast so the resulting geometry is comparable.
    """
    directions: dict[str, dict[int, np.ndarray]] = {f: {} for f in FOUNDATION_ORDER}
    for layer in layers:
        X = acts[layer]
        for fv in FOUNDATION_ORDER:
            mask = labels == fv
            if mask.sum() < 2:
                continue
            d = X[mask].mean(axis=0) - neutral_mean[layer]
            directions[fv][layer] = d / (np.linalg.norm(d) + 1e-12)
    return {f: v for f, v in directions.items() if v}


def layer_geometry(directions: dict[str, dict[int, np.ndarray]],
                   labels_present: list[str], layers: list[int]) -> dict:
    """Per-layer cosine geometry (mean cosine, eff. dim, MFT permutation test)."""
    per_layer = {}
    for layer in layers:
        if not all(layer in directions.get(f, {}) for f in labels_present):
            continue
        cos = compute_cosine_matrix(directions, layer, labels=labels_present)
        if cos is None:
            continue
        eff_dim = compute_effective_dimensionality(directions, layer, labels=labels_present)
        iu = np.triu_indices(len(labels_present), k=1)
        entry = {
            "mean_cosine_similarity": round(float(cos[iu].mean()), 4),
            "effective_dimensionality": eff_dim,
            "cosine_similarity_matrix": cos.tolist(),
        }
        if len(labels_present) == 6:
            entry["permutation_test"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in permutation_test_mft(cos, n_perm=2000).items()
            }
        per_layer[str(layer)] = entry
    return per_layer


def _load_deepsteer_directions(path: Path) -> dict[str, dict[int, np.ndarray]]:
    out: dict[str, dict[int, np.ndarray]] = {}
    if not path.exists():
        logger.warning("DeepSteer directions npz missing: %s", path)
        return out
    npz = np.load(path)
    for key in npz.files:
        # key format: "{foundation}_layer{idx}"
        fv, _, layer_s = key.rpartition("_layer")
        if fv in FOUNDATION_ORDER:
            out.setdefault(fv, {})[int(layer_s)] = npz[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MFV external robustness (Extension E).")
    parser.add_argument("--model", default=OLMO_REPO, help="HuggingFace model ID.")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/external_robustness")
    parser.add_argument("--figures-dir", default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz",
                        help="DeepSteer exp1 npz for cross-dataset alignment (match scale).")
    parser.add_argument("--mfv-file", default=None,
                        help="Optional JSON list of {text, foundation} to override bundled MFV.")
    parser.add_argument("--max-neutral", type=int, default=200,
                        help="Cap neutral baseline sentences for speed (0 = all).")
    parser.add_argument("--display-layer", type=int, default=-1,
                        help="Layer for heatmap/alignment figure (-1 = best by eff. dim).")
    parser.add_argument("--device", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    from deepsteer.core.model_interface import WhiteBoxModel
    from deepsteer.core.types import AccessTier
    from deepsteer.datasets.pipeline import build_probing_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tag = _model_tag(args.model)

    items = _load_mfv(args.mfv_file)
    texts = [it["text"] for it in items]
    labels = np.array([it["foundation"] for it in items])
    present = [f for f in FOUNDATION_ORDER if (labels == f).sum() >= 2]
    print(f"MFV: {len(texts)} vignettes, {len(present)} foundations with >=2 items")

    print(f"Loading model: {args.model}")
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    layers = list(range(n_layers))

    print("Extracting MFV activations...")
    raw = model.collect_batch_activations(texts, layers=layers, pooling="mean")
    acts = {ly: raw[ly].cpu().numpy().astype(np.float64) for ly in layers}

    # Shared neutral baseline + matched DeepSteer mean-diff directions, so MFV and
    # DeepSteer geometry are compared under the SAME estimator (mean-diff vs the
    # same neutral baseline). This isolates "is the geometry a dataset artifact?"
    # from "probe-weight vs mean-diff estimator" differences.
    print("Building neutral baseline + DeepSteer mean-diff directions...")
    ds = build_probing_dataset(target_per_foundation=40, dataset_version="v2")
    ds_pairs = list(ds.train) + list(ds.test)
    neutral_texts = list({p.neutral for p in ds_pairs})
    if args.max_neutral and len(neutral_texts) > args.max_neutral:
        neutral_texts = neutral_texts[:args.max_neutral]
    print(f"  {len(neutral_texts)} unique neutral sentences")
    neutral_raw = model.collect_batch_activations(neutral_texts, layers=layers, pooling="mean")
    neutral_mean = {ly: neutral_raw[ly].cpu().numpy().astype(np.float64).mean(axis=0)
                    for ly in layers}

    mfv_directions = compute_mfv_directions(acts, labels, layers, neutral_mean)
    labels_present = [f for f in FOUNDATION_ORDER if f in mfv_directions]

    # DeepSteer mean-diff directions (same neutral baseline, DeepSteer moral texts).
    ds_moral_texts = [p.moral for p in ds_pairs]
    ds_moral_labels = np.array([p.foundation.value for p in ds_pairs])
    ds_moral_raw = model.collect_batch_activations(ds_moral_texts, layers=layers, pooling="mean")
    ds_moral_acts = {ly: ds_moral_raw[ly].cpu().numpy().astype(np.float64) for ly in layers}
    ds_meandiff = compute_mfv_directions(ds_moral_acts, ds_moral_labels, layers, neutral_mean)

    # --- Per-layer geometry, matched estimator ---
    per_layer = layer_geometry(mfv_directions, labels_present, layers)
    ds_per_layer = layer_geometry(ds_meandiff, labels_present, layers)

    # Best layer = highest effective dimensionality (closest to full integration).
    if args.display_layer >= 0:
        disp = args.display_layer
    else:
        disp = int(max(per_layer, key=lambda k: (per_layer[k]["effective_dimensionality"] or 0,
                                                  per_layer[k]["mean_cosine_similarity"])))

    geom = {
        "experiment": "external_dataset_robustness_mfv",
        "model": args.model,
        "n_layers": n_layers,
        "foundations_present": labels_present,
        "n_vignettes": len(texts),
        "n_neutral_baseline": len(neutral_texts),
        "display_layer": disp,
        "note": "per_layer = MFV mean-diff geometry; deepsteer_meandiff_per_layer = "
                "DeepSteer mean-diff geometry under the SAME neutral baseline (matched estimator).",
        "per_layer": per_layer,
        "deepsteer_meandiff_per_layer": ds_per_layer,
    }
    geom_path = output_dir / f"mfv_geometry_{tag}.json"
    with open(geom_path, "w") as f:
        json.dump(geom, f, indent=2)
    print(f"\nMFV geometry: {geom_path}")
    if str(disp) in per_layer:
        d = per_layer[str(disp)]
        dd = ds_per_layer.get(str(disp), {})
        print(f"  layer {disp} (matched mean-diff estimator): "
              f"MFV eff_dim={d['effective_dimensionality']} "
              f"mean_cos={d['mean_cosine_similarity']:.3f}  |  "
              f"DeepSteer eff_dim={dd.get('effective_dimensionality')} "
              f"mean_cos={dd.get('mean_cosine_similarity')}")
        print(f"  (max eff_dim = {len(labels_present)}; near-maximal eff_dim under both = "
              f"integration signature replicates)")

    # --- Cross-dataset direction alignment ---
    # MFV directions vs (a) DeepSteer probe-weight (npz) and (b) DeepSteer mean-diff.
    deepsteer_pw = _load_deepsteer_directions(Path(args.directions))
    alignment = {"experiment": "cross_dataset_alignment", "model": args.model,
                 "directions_npz": args.directions,
                 "per_layer_vs_probe_weight": {}, "per_layer_vs_meandiff": {}}

    def _align(other: dict, key: str) -> dict:
        at_disp = {}
        for layer in layers:
            row = {}
            for fv in labels_present:
                if layer in mfv_directions.get(fv, {}) and layer in other.get(fv, {}):
                    a = mfv_directions[fv][layer]
                    b = other[fv][layer]
                    b = b / (np.linalg.norm(b) + 1e-12)
                    row[fv] = round(abs(float(np.dot(a, b))), 4)
            if row:
                alignment[key][str(layer)] = row
                if layer == disp:
                    at_disp = row
        return at_disp

    align_at_disp = _align(deepsteer_pw, "per_layer_vs_probe_weight")
    _align(ds_meandiff, "per_layer_vs_meandiff")
    align_path = output_dir / f"cross_dataset_alignment_{tag}.json"
    with open(align_path, "w") as f:
        json.dump(alignment, f, indent=2)
    print(f"Cross-dataset alignment: {align_path}")
    if align_at_disp:
        print(f"  alignment @layer {disp}: " +
              ", ".join(f"{FOUNDATION_SHORT[f]}={v:.2f}" for f, v in align_at_disp.items()))

    _make_figure(geom, alignment, disp, labels_present, tag, figures_dir)
    print(f"Figures: {figures_dir}")


def _make_figure(geom, alignment, disp, labels_present, tag, figures_dir):
    per_layer = geom["per_layer"]
    if str(disp) not in per_layer:
        logger.warning("Figure skipped: display layer %d has no geometry", disp)
        return
    short = [FOUNDATION_SHORT[f] for f in labels_present]
    n = len(labels_present)
    cos = np.array(per_layer[str(disp)]["cosine_similarity_matrix"])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # (a) MFV cosine heatmap at display layer
    ax = axes[0]
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(short, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(cos[i, j]) > 0.6 else "black")
    if n == 6:
        ax.axhline(2.5, color="black", lw=2)
        ax.axvline(2.5, color="black", lw=2)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(f"(a) MFV cosine (layer {disp})\nmean = "
                 f"{per_layer[str(disp)]['mean_cosine_similarity']:.3f}",
                 fontsize=11, fontweight="bold")

    # (b) mean cosine + eff dim vs layer, MFV vs DeepSteer (matched mean-diff estimator)
    ax = axes[1]
    ds_pl = geom.get("deepsteer_meandiff_per_layer", {})
    ls = sorted(int(k) for k in per_layer)
    mc = [per_layer[str(ly)]["mean_cosine_similarity"] for ly in ls]
    ed = [per_layer[str(ly)]["effective_dimensionality"] for ly in ls]
    ax.plot(ls, mc, "o-", color="#1E88E5", label="MFV mean cosine")
    if ds_pl:
        ls2 = sorted(int(k) for k in ds_pl)
        ax.plot(ls2, [ds_pl[str(ly)]["mean_cosine_similarity"] for ly in ls2],
                "^-", color="#3949AB", alpha=0.7, label="DeepSteer mean cosine")
    ax.axhspan(0.2, 0.4, color="gray", alpha=0.12, label="DeepSteer probe-weight range")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Pairwise Cosine", color="#1E88E5")
    ax2 = ax.twinx()
    ax2.plot(ls, ed, "s--", color="#E53935", label="MFV eff. dim")
    ax2.set_ylabel("Effective Dimensionality", color="#E53935")
    ax2.set_ylim(0.5, n + 0.5)
    ax.set_title("(b) Geometry vs depth (matched estimator)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7)

    # (c) cross-dataset alignment at display layer (MFV vs DeepSteer probe-weight)
    ax = axes[2]
    row = alignment["per_layer_vs_probe_weight"].get(str(disp), {})
    if row:
        fvs = [f for f in labels_present if f in row]
        vals = [row[f] for f in fvs]
        colors = ["#43A047" if f in INDIVIDUALIZING else "#FB8C00" for f in fvs]
        ax.bar([FOUNDATION_SHORT[f] for f in fvs], vals, color=colors, edgecolor="black")
        ax.axhline(0.5, color="red", ls="--", lw=1, label="0.5 (not artifact)")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
    ax.set_ylabel("|cosine| MFV vs DeepSteer")
    ax.set_title(f"(c) Cross-dataset alignment (layer {disp})", fontsize=11, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)

    fig.suptitle(f"MFV external robustness ({_model_label(geom['model'])})",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(figures_dir / f"external_robustness_{tag}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"external_robustness_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figures_dir / ('external_robustness_' + tag)}.{{png,pdf}}")


if __name__ == "__main__":
    main()
