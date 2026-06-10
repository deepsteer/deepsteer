#!/usr/bin/env python3
"""Data-driven moral taxonomy via activation clustering (Extension B).

The model's *own* organization of moral content, discovered without reference to
MFT labels, then compared against them. The exp1 npz stores only directions, so
moral-positive activations are re-extracted here with a forward pass (the script
loads the model and is GPU-friendly).

Method, per stable layer:
  1. Mean-pool activations of the moral sentence of every probing pair.
  2. k-means and spectral clustering for k = 2..K; pick k by silhouette score.
  3. Compare the discovered clustering to MFT foundations (adjusted mutual
     information, per-cluster Jaccard, confusion counts).
  4. Name each cluster by projecting its centroid onto the 6 foundation
     directions (loaded from the exp1 npz).

Outputs:
  - outputs/taxonomy/clustering_results_<tag>.json
  - outputs/figures/taxonomy_confusion_<tag>.pdf  (cluster x foundation heatmap)
  - outputs/figures/taxonomy_projection_<tag>.pdf  (2D UMAP or PCA, foundation vs cluster)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

from deepsteer.foundations import FOUNDATION_ORDER, FOUNDATION_SHORT

logger = logging.getLogger(__name__)

OLMO_REPO = "allenai/OLMo-2-0425-1B"


def _model_label(model_id: str) -> str:
    name = model_id.rstrip("/").split("/")[-1]
    return f"OLMo-2 {name.split('-')[-1]}" if "OLMo-2" in name else name


def _model_tag(model_id: str) -> str:
    """Short filename tag, e.g. '1B' or '7B'."""
    name = model_id.rstrip("/").split("/")[-1]
    return name.split("-")[-1] if "OLMo-2" in name else name.replace("/", "_")


def _resolve_layers(spec: str, n_layers: int) -> list[int]:
    """'auto' -> middle third of the network; else a comma list of ints."""
    if spec == "auto":
        return list(range(n_layers // 3, (2 * n_layers) // 3 + 1))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def _load_foundation_directions(path: Path, layer: int) -> dict[str, np.ndarray]:
    """Return {foundation: unit direction} at one layer from the exp1 npz."""
    out: dict[str, np.ndarray] = {}
    if not path.exists():
        return out
    npz = np.load(path)
    for fv in FOUNDATION_ORDER:
        key = f"{fv}_layer{layer}"
        if key in npz:
            out[fv] = npz[key]
    return out


def _name_clusters(centroids: np.ndarray, directions: dict[str, np.ndarray],
                   top_n: int = 2) -> list[dict]:
    """Name each cluster by its centroid's projection onto foundation directions."""
    names = []
    fvs = [f for f in FOUNDATION_ORDER if f in directions]
    if not fvs:
        return [{"label": f"cluster_{i}", "projections": {}} for i in range(len(centroids))]
    basis = np.stack([directions[f] for f in fvs])  # (n_found, hidden)
    for c in centroids:
        cu = c / (np.linalg.norm(c) + 1e-12)
        proj = basis @ cu  # cosine of centroid with each foundation direction
        order = np.argsort(proj)[::-1]
        top = [fvs[i] for i in order[:top_n]]
        names.append({
            "label": "+".join(FOUNDATION_SHORT[f] for f in top),
            "projections": {fvs[i]: round(float(proj[i]), 4) for i in range(len(fvs))},
        })
    return names


def analyze_layer(X: np.ndarray, foundation_labels: np.ndarray, layer: int,
                  k_min: int, k_max: int,
                  directions: dict[str, np.ndarray], seed: int = 42) -> dict:
    """Cluster one layer's activations and compare to MFT foundations."""
    # L2-normalize rows so Euclidean k-means approximates cosine clustering.
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    per_k = {}
    best_k, best_sil = None, -2.0
    best_km = None
    for k in range(k_min, min(k_max, len(Xn) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xn)
        sil_km = float(silhouette_score(Xn, km.labels_))
        try:
            sc = SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                                    random_state=seed).fit(Xn)
            sil_sc = float(silhouette_score(Xn, sc.labels_))
        except Exception as exc:  # spectral can fail on degenerate graphs
            logger.debug("spectral k=%d failed: %s", k, exc)
            sil_sc = None
        per_k[k] = {"silhouette_kmeans": round(sil_km, 4),
                    "silhouette_spectral": round(sil_sc, 4) if sil_sc is not None else None}
        if sil_km > best_sil:
            best_sil, best_k, best_km = sil_km, k, km

    ami = float(adjusted_mutual_info_score(foundation_labels, best_km.labels_))

    # Confusion counts: rows = discovered cluster, cols = foundation (FOUNDATION_ORDER).
    fv_to_idx = {f: i for i, f in enumerate(FOUNDATION_ORDER)}
    found_idx = np.array([fv_to_idx[f] for f in foundation_labels])
    n_clusters = best_k
    table = np.zeros((n_clusters, len(FOUNDATION_ORDER)), dtype=int)
    for cl, fi in zip(best_km.labels_, found_idx):
        table[cl, fi] += 1

    # Per-cluster Jaccard with each foundation.
    jaccard = {}
    for cl in range(n_clusters):
        cl_members = set(np.where(best_km.labels_ == cl)[0].tolist())
        jaccard[cl] = {}
        for fi, fv in enumerate(FOUNDATION_ORDER):
            fv_members = set(np.where(found_idx == fi)[0].tolist())
            union = cl_members | fv_members
            jaccard[cl][fv] = round(len(cl_members & fv_members) / len(union), 4) if union else 0.0

    centroids = np.stack([X[best_km.labels_ == cl].mean(axis=0) for cl in range(n_clusters)])
    cluster_names = _name_clusters(centroids, directions)

    return {
        "layer": layer,
        "best_k": best_k,
        "best_silhouette": round(best_sil, 4),
        "adjusted_mutual_info": round(ami, 4),
        "per_k_silhouette": per_k,
        "confusion_cluster_by_foundation": table.tolist(),
        "foundation_order": FOUNDATION_ORDER,
        "jaccard": {str(k): v for k, v in jaccard.items()},
        "cluster_names": cluster_names,
        "_labels": best_km.labels_,  # consumed by figures, stripped before JSON
    }


def fig_confusion(layer_result: dict, tag: str, figures_dir: Path) -> None:
    table = np.array(layer_result["confusion_cluster_by_foundation"], dtype=float)
    row_sums = table.sum(axis=1, keepdims=True)
    frac = table / np.clip(row_sums, 1, None)
    n_clusters = table.shape[0]

    fig, ax = plt.subplots(figsize=(8, 1 + 0.6 * n_clusters + 2))
    im = ax.imshow(frac, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(FOUNDATION_ORDER)))
    ax.set_xticklabels([FOUNDATION_SHORT[f] for f in FOUNDATION_ORDER],
                       rotation=45, ha="right")
    names = [c["label"] for c in layer_result["cluster_names"]]
    ax.set_yticks(range(n_clusters))
    ax.set_yticklabels([f"C{i}: {names[i]}" for i in range(n_clusters)], fontsize=9)
    for i in range(n_clusters):
        for j in range(len(FOUNDATION_ORDER)):
            ax.text(j, i, f"{int(table[i, j])}", ha="center", va="center",
                    color="white" if frac[i, j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Row-normalized share")
    ax.set_title(
        f"Discovered clusters vs MFT foundations ({_label_from_tag(tag)}, layer "
        f"{layer_result['layer']}, k={layer_result['best_k']})\n"
        f"AMI = {layer_result['adjusted_mutual_info']:.3f}",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, figures_dir, f"taxonomy_confusion_{tag}")


def fig_projection(X: np.ndarray, foundation_labels: np.ndarray, cluster_labels: np.ndarray,
                   layer: int, tag: str, figures_dir: Path) -> str:
    """2D projection colored by foundation (left) and discovered cluster (right)."""
    method = "PCA"
    try:
        import umap  # noqa: F401
        reducer = umap.UMAP(n_components=2, random_state=42)
        emb = reducer.fit_transform(X)
        method = "UMAP"
    except Exception:
        emb = PCA(n_components=2, random_state=42).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fv_to_idx = {f: i for i, f in enumerate(FOUNDATION_ORDER)}
    found_idx = np.array([fv_to_idx[f] for f in foundation_labels])
    cmap = plt.get_cmap("tab10")

    for i, fv in enumerate(FOUNDATION_ORDER):
        m = found_idx == i
        axes[0].scatter(emb[m, 0], emb[m, 1], s=18, color=cmap(i),
                        label=FOUNDATION_SHORT[fv], alpha=0.7)
    axes[0].set_title("Colored by MFT foundation", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=8, markerscale=1.2)

    for cl in sorted(set(cluster_labels.tolist())):
        m = cluster_labels == cl
        axes[1].scatter(emb[m, 0], emb[m, 1], s=18, color=cmap(cl % 10),
                        label=f"C{cl}", alpha=0.7)
    axes[1].set_title("Colored by discovered cluster", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=8, markerscale=1.2)

    for ax in axes:
        ax.set_xlabel(f"{method}-1")
        ax.set_ylabel(f"{method}-2")
    fig.suptitle(f"Moral activation taxonomy ({_label_from_tag(tag)}, layer {layer}, {method})",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, figures_dir, f"taxonomy_projection_{tag}")
    return method


def _label_from_tag(tag: str) -> str:
    return f"OLMo-2 {tag}" if tag in ("1B", "7B") else tag


def _save(fig, figures_dir: Path, stem: str) -> None:
    fig.savefig(figures_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(figures_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {figures_dir / stem}.{{png,pdf}}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Data-driven moral taxonomy (Extension B).")
    parser.add_argument("--model", default=OLMO_REPO, help="HuggingFace model ID.")
    parser.add_argument("--output-dir", default="papers/3_moral_geometry/outputs/taxonomy")
    parser.add_argument("--figures-dir", default="papers/3_moral_geometry/outputs/figures")
    parser.add_argument("--directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz",
                        help="exp1 npz for cluster naming (match the model scale).")
    parser.add_argument("--dataset-target", type=int, default=40)
    parser.add_argument("--layers", default="auto",
                        help="'auto' (middle third) or comma-separated layer indices.")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--max-pairs", type=int, default=0,
                        help="Cap moral texts for speed (0 = all).")
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

    print("Building probing dataset...")
    dataset = build_probing_dataset(target_per_foundation=args.dataset_target,
                                    dataset_version="v2")
    pairs = list(dataset.train) + list(dataset.test)
    moral_texts = [p.moral for p in pairs]
    foundation_labels = np.array([p.foundation.value for p in pairs])
    if args.max_pairs and len(moral_texts) > args.max_pairs:
        moral_texts = moral_texts[:args.max_pairs]
        foundation_labels = foundation_labels[:args.max_pairs]
    print(f"Moral texts: {len(moral_texts)} across {len(set(foundation_labels))} foundations")

    print(f"Loading model: {args.model}")
    model = WhiteBoxModel(args.model, device=args.device, access_tier=AccessTier.WEIGHTS)
    n_layers = model.info.n_layers
    layers = [ly for ly in _resolve_layers(args.layers, n_layers) if 0 <= ly < n_layers]
    print(f"{_model_label(args.model)}: {n_layers} layers; analyzing layers {layers}")

    print("Extracting moral-positive activations...")
    acts = model.collect_batch_activations(moral_texts, layers=layers, pooling="mean")

    results = {
        "experiment": "data_driven_taxonomy",
        "model": args.model,
        "n_layers": n_layers,
        "n_moral_texts": len(moral_texts),
        "layers_analyzed": layers,
        "k_range": [args.k_min, args.k_max],
        "per_layer": {},
    }
    best_layer, best_ami, best_payload = None, -2.0, None
    for layer in layers:
        X = acts[layer].cpu().numpy().astype(np.float64)
        directions = _load_foundation_directions(Path(args.directions), layer)
        lr = analyze_layer(X, foundation_labels, layer, args.k_min, args.k_max, directions)
        labels = lr.pop("_labels")
        results["per_layer"][str(layer)] = lr
        print(f"  layer {layer:2d}: best_k={lr['best_k']} "
              f"silhouette={lr['best_silhouette']:.3f} AMI={lr['adjusted_mutual_info']:.3f} "
              f"clusters={[c['label'] for c in lr['cluster_names']]}")
        if lr["adjusted_mutual_info"] > best_ami:
            best_ami, best_layer = lr["adjusted_mutual_info"], layer
            best_payload = (X, labels, lr)

    results["best_layer_by_ami"] = best_layer
    out_json = output_dir / f"clustering_results_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {out_json}")

    if best_payload is not None:
        X, labels, lr = best_payload
        print(f"Best foundation-aligned layer: {best_layer} (AMI={best_ami:.3f})")
        fig_confusion(lr, tag, figures_dir)
        method = fig_projection(X, foundation_labels, labels, best_layer, tag, figures_dir)
        results["projection_method"] = method
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)

    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    main()
