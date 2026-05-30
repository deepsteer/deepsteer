"""Compute random-direction baseline for SAE subspace overlap membership.

Generates 100 random unit vectors in 2048-d, computes their orthonormal basis
via SVD, projects each of 6 probe directions onto the basis, and records
mean membership. Repeats 1000 times to build a null distribution.

Compares the observed SAE mean_probe_membership (0.1963) against this null.

Updates sae_moral_features_layer8.json with a `random_baseline` field.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from shared import FOUNDATION_ORDER, PAPER_ROOT, load_probe_directions


def random_subspace_membership(
    probe_vecs: np.ndarray,
    d_model: int,
    n_random: int,
    n_iter: int,
    seed: int = 42,
) -> np.ndarray:
    """Compute mean probe membership in random subspaces.

    Args:
        probe_vecs: (n_probes, d_model) unit vectors.
        d_model: Dimensionality of the space.
        n_random: Number of random vectors per iteration.
        n_iter: Number of Monte Carlo iterations.
        seed: RNG seed.

    Returns:
        Array of shape (n_iter,) with mean membership across probes per iteration.
    """
    rng = np.random.RandomState(seed)
    n_probes = probe_vecs.shape[0]
    results = np.empty(n_iter)

    for i in range(n_iter):
        # Generate n_random random unit vectors in d_model-d
        raw = rng.randn(n_random, d_model)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        raw /= norms  # unit vectors

        # SVD to get orthonormal basis (same as SAE code)
        _, s, Vt = np.linalg.svd(raw, full_matrices=False)
        rank = min(n_random, int(np.sum(s > 1e-8)))
        basis = Vt[:rank]  # (rank, d_model)

        # Project each probe direction onto the random basis
        memberships = np.empty(n_probes)
        for j in range(n_probes):
            proj = basis @ probe_vecs[j]
            memberships[j] = float(np.dot(proj, proj))

        results[i] = memberships.mean()

    return results


def main() -> None:
    layer = 8
    n_random = 100  # same as top_k SAE features
    n_iter = 1000
    observed_value = 0.1963  # from subspace_overlap_meandiff.mean_probe_membership

    # Load probe directions
    probe_path = PAPER_ROOT / "outputs" / "exp1_2_3" / "exp1_probe_directions.npz"
    directions = load_probe_directions(probe_path)

    # Extract 6 foundation directions at layer 8
    probe_vecs = []
    for fv in FOUNDATION_ORDER:
        d = directions[fv][layer]
        probe_vecs.append(d)
    probe_mat = np.stack(probe_vecs)  # (6, 2048)
    d_model = probe_mat.shape[1]

    print(f"Probe directions: {probe_mat.shape}")
    print(f"Running {n_iter} iterations of random baseline "
          f"({n_random} random unit vectors in {d_model}-d)...")

    results = random_subspace_membership(probe_mat, d_model, n_random, n_iter)

    mean_val = float(results.mean())
    std_val = float(results.std())
    percentile = float(np.mean(results >= observed_value) * 100)
    ratio = observed_value / mean_val if mean_val > 0 else float("inf")

    print(f"\nRandom baseline: {mean_val:.6f} +/- {std_val:.6f}")
    print(f"Observed value:  {observed_value:.4f}")
    print(f"Ratio (observed / random): {ratio:.2f}x")
    print(f"Percentile of observed in null: {100 - percentile:.1f}th "
          f"(p = {percentile / 100:.4f})")

    # Also compute per-foundation random baselines for context
    per_foundation_random = {}
    rng = np.random.RandomState(42)
    per_foundation_all = {fv: np.empty(n_iter) for fv in FOUNDATION_ORDER}

    for i in range(n_iter):
        raw = rng.randn(n_random, d_model)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        raw /= norms
        _, s, Vt = np.linalg.svd(raw, full_matrices=False)
        rank = min(n_random, int(np.sum(s > 1e-8)))
        basis = Vt[:rank]

        for j, fv in enumerate(FOUNDATION_ORDER):
            proj = basis @ probe_mat[j]
            per_foundation_all[fv][i] = float(np.dot(proj, proj))

    for fv in FOUNDATION_ORDER:
        vals = per_foundation_all[fv]
        per_foundation_random[fv] = {
            "mean": round(float(vals.mean()), 6),
            "std": round(float(vals.std()), 6),
        }

    # Build random_baseline payload
    random_baseline = {
        "description": (
            "Null distribution: 100 random unit vectors in 2048-d, "
            "SVD basis, project 6 probe directions, record mean membership. "
            "1000 iterations."
        ),
        "n_random_vectors": n_random,
        "n_iterations": n_iter,
        "d_model": d_model,
        "seed": 42,
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "observed_value": observed_value,
        "ratio_observed_over_random": round(ratio, 4),
        "observed_percentile": round(100 - percentile, 2),
        "p_value": round(percentile / 100, 4),
        "per_foundation_random": per_foundation_random,
    }

    # Update the JSON file
    json_path = PAPER_ROOT / "outputs" / "probe_engineering" / "sae_moral_features_layer8.json"
    with open(json_path) as f:
        data = json.load(f)

    data["random_baseline"] = random_baseline

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {json_path}")
    print(f"\nrandom_baseline field:")
    print(json.dumps(random_baseline, indent=2))


if __name__ == "__main__":
    main()
