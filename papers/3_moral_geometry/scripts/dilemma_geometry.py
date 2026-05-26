#!/usr/bin/env python3
"""Script 4: Pooled dilemma geometry analysis.

Analyzes the geometric relationships among the 15 dilemma probe directions
and the 6 foundation directions. Computes cosine similarity matrices,
shared-component analysis, effective dimensionality, and hierarchical clustering.

Usage:
    python papers/3_moral_geometry/scripts/dilemma_geometry.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import linkage

logger = logging.getLogger(__name__)

FOUNDATION_ORDER = [
    "care_harm", "fairness_cheating", "liberty_oppression",
    "loyalty_betrayal", "authority_subversion", "sanctity_degradation",
]

FOUNDATION_SHORT = {
    "care_harm": "Care",
    "fairness_cheating": "Fairness",
    "liberty_oppression": "Liberty",
    "loyalty_betrayal": "Loyalty",
    "authority_subversion": "Authority",
    "sanctity_degradation": "Sanctity",
}

FOUNDATION_PAIRS = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]

DILEMMA_PAIR_KEYS = [f"{a}-{b}" for a, b in FOUNDATION_PAIRS]


def compute_effective_dimensionality(
    directions: np.ndarray,
    variance_threshold: float = 0.9,
) -> int:
    """Number of PCs explaining >=threshold of variance."""
    mat_centered = directions - directions.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat_centered, full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(explained, variance_threshold)) + 1


def shared_component_analysis(
    dilemma_cosines: np.ndarray,
    dilemma_keys: list[str],
) -> dict:
    """Compare cosine similarity of dilemma pairs that share a foundation vs. those that don't."""
    n = len(dilemma_keys)
    pairs_share = []
    pairs_no_share = []

    for i in range(n):
        parts_i = set(dilemma_keys[i].split("-"))
        for j in range(i + 1, n):
            parts_j = set(dilemma_keys[j].split("-"))
            cos_val = dilemma_cosines[i, j]
            if parts_i & parts_j:
                pairs_share.append(cos_val)
            else:
                pairs_no_share.append(cos_val)

    return {
        "shared_component_pairs": len(pairs_share),
        "no_shared_component_pairs": len(pairs_no_share),
        "mean_cosine_shared": float(np.mean(pairs_share)) if pairs_share else None,
        "std_cosine_shared": float(np.std(pairs_share)) if pairs_share else None,
        "mean_cosine_no_shared": float(np.mean(pairs_no_share)) if pairs_no_share else None,
        "std_cosine_no_shared": float(np.std(pairs_no_share)) if pairs_no_share else None,
        "difference": float(np.mean(pairs_share) - np.mean(pairs_no_share))
            if pairs_share and pairs_no_share else None,
        "all_shared_cosines": [round(float(x), 6) for x in pairs_share],
        "all_no_shared_cosines": [round(float(x), 6) for x in pairs_no_share],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pooled dilemma geometry analysis.")
    parser.add_argument("--dilemma-directions",
                        default="papers/3_moral_geometry/outputs/dilemma_probing/dilemma_probe_directions.npz")
    parser.add_argument("--foundation-directions",
                        default="papers/3_moral_geometry/outputs/exp1_2_3/exp1_probe_directions.npz")
    parser.add_argument("--output-dir",
                        default="papers/3_moral_geometry/outputs/dilemma_geometry")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    dilemma_dir_path = Path(args.dilemma_directions)
    foundation_dir_path = Path(args.foundation_directions)

    if not dilemma_dir_path.exists():
        print(f"ERROR: {dilemma_dir_path} not found. Run dilemma_probing.py first.")
        return
    if not foundation_dir_path.exists():
        print(f"ERROR: {foundation_dir_path} not found.")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dilemma_data = np.load(dilemma_dir_path)
    foundation_data = np.load(foundation_dir_path)

    # Determine n_layers from data
    sample_key = [k for k in foundation_data.keys() if k.startswith("care_harm_layer")][0]
    hidden_dim = foundation_data[sample_key].shape[0]
    n_layers = max(
        int(k.split("layer")[-1]) for k in foundation_data.keys()
    ) + 1

    print(f"Analyzing {n_layers} layers, {hidden_dim} hidden dim")
    print(f"Foundation directions: {len(foundation_data.keys())} arrays")
    print(f"Dilemma directions: {len(dilemma_data.keys())} arrays")

    per_layer_results: dict[str, dict] = {}

    for layer_idx in range(n_layers):
        # Load foundation directions for this layer
        foundation_vecs = []
        foundation_labels = []
        for fv in FOUNDATION_ORDER:
            key = f"{fv}_layer{layer_idx}"
            if key in foundation_data:
                foundation_vecs.append(foundation_data[key])
                foundation_labels.append(FOUNDATION_SHORT[fv])

        # Load dilemma directions for this layer
        dilemma_vecs = []
        dilemma_labels = []
        dilemma_keys = []
        for pk in DILEMMA_PAIR_KEYS:
            key = f"dilemma_{pk}_layer{layer_idx}"
            if key in dilemma_data:
                dilemma_vecs.append(dilemma_data[key])
                dilemma_labels.append(pk)
                dilemma_keys.append(pk)

        if len(dilemma_vecs) < 2:
            continue

        dilemma_mat = np.stack(dilemma_vecs)  # (15, hidden_dim)

        # 15×15 dilemma cosine similarity
        dilemma_cosines = dilemma_mat @ dilemma_mat.T

        # Shared-component analysis
        shared_analysis = shared_component_analysis(dilemma_cosines, dilemma_keys)

        # Effective dimensionality of dilemma directions
        dilemma_eff_dim = compute_effective_dimensionality(dilemma_mat)

        # Foundation effective dimensionality for comparison
        if len(foundation_vecs) >= 2:
            foundation_mat = np.stack(foundation_vecs)
            foundation_eff_dim = compute_effective_dimensionality(foundation_mat)
        else:
            foundation_mat = None
            foundation_eff_dim = None

        # 21-direction combined analysis
        combined_cosines = None
        combined_labels = None
        linkage_data = None

        if foundation_mat is not None and len(foundation_vecs) == 6:
            combined_mat = np.vstack([foundation_mat, dilemma_mat])  # (21, hidden_dim)
            combined_cosines = combined_mat @ combined_mat.T
            combined_labels = foundation_labels + dilemma_labels

            # Hierarchical clustering
            dist = 1 - combined_cosines
            n = len(combined_labels)
            condensed = []
            for i in range(n):
                for j in range(i + 1, n):
                    condensed.append(max(dist[i, j], 0))
            condensed = np.array(condensed)
            Z = linkage(condensed, method="ward")
            linkage_data = Z.tolist()

            combined_eff_dim = compute_effective_dimensionality(combined_mat)
        else:
            combined_eff_dim = None

        # Extract upper triangles
        n_d = len(dilemma_keys)
        dilemma_upper = [float(dilemma_cosines[i, j])
                         for i in range(n_d) for j in range(i + 1, n_d)]
        mean_dilemma_cos = float(np.mean(dilemma_upper)) if dilemma_upper else None

        per_layer_results[str(layer_idx)] = {
            "dilemma_cosine_matrix": dilemma_cosines.tolist(),
            "dilemma_labels": dilemma_labels,
            "mean_dilemma_cosine": round(mean_dilemma_cos, 6) if mean_dilemma_cos is not None else None,
            "dilemma_effective_dim": dilemma_eff_dim,
            "foundation_effective_dim": foundation_eff_dim,
            "combined_effective_dim": combined_eff_dim,
            "shared_component_analysis": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in shared_analysis.items()
                if k not in ("all_shared_cosines", "all_no_shared_cosines")
            },
            "combined_cosine_matrix": combined_cosines.tolist() if combined_cosines is not None else None,
            "combined_labels": combined_labels,
            "linkage": linkage_data,
        }

    # Summary across layers
    layers_with_data = sorted(int(k) for k in per_layer_results.keys())
    peak_shared_diff_layer = None
    peak_shared_diff = -999

    for l in layers_with_data:
        diff = per_layer_results[str(l)]["shared_component_analysis"].get("difference")
        if diff is not None and diff > peak_shared_diff:
            peak_shared_diff = diff
            peak_shared_diff_layer = l

    output = {
        "experiment": "dilemma_geometry",
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dilemma_pair_keys": DILEMMA_PAIR_KEYS,
        "peak_shared_component_diff_layer": peak_shared_diff_layer,
        "peak_shared_component_diff": round(peak_shared_diff, 6) if peak_shared_diff > -999 else None,
        "per_layer": per_layer_results,
    }

    with open(output_dir / "dilemma_geometry.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("DILEMMA GEOMETRY SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Layer':>6s} {'Dilemma Dim':>12s} {'Found. Dim':>12s} {'Combined':>10s} "
          f"{'Shared Cos':>12s} {'No-Share':>10s} {'Diff':>8s}")
    print("-" * 75)
    for l in layers_with_data:
        ld = per_layer_results[str(l)]
        sc = ld["shared_component_analysis"]
        print(f"{l:>6d} {ld['dilemma_effective_dim']:>12d} "
              f"{ld['foundation_effective_dim'] or 'N/A':>12} "
              f"{ld.get('combined_effective_dim') or 'N/A':>10} "
              f"{sc.get('mean_cosine_shared', 'N/A'):>12} "
              f"{sc.get('mean_cosine_no_shared', 'N/A'):>10} "
              f"{sc.get('difference', 'N/A'):>8}")

    if peak_shared_diff_layer is not None:
        print(f"\nPeak shared-component difference: {peak_shared_diff:.4f} at layer {peak_shared_diff_layer}")
        if peak_shared_diff > 0.05:
            print("Dilemma directions show compositional structure (shared components cluster).")
        else:
            print("Weak or no compositional structure in dilemma directions.")

    print(f"\nOutput: {output_dir / 'dilemma_geometry.json'}")


if __name__ == "__main__":
    main()
