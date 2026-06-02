"""Subspace analysis: orthonormal bases, membership, and projection.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
and full_subspace_projection.py
"""

from __future__ import annotations

import numpy as np


def orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    """Compute orthonormal basis for the span of rows of *vectors* via SVD.

    Args:
        vectors: ``(n_vecs, dim)`` matrix whose rows span the subspace.

    Returns:
        ``(rank, dim)`` orthonormal basis matrix.
    """
    _, s, Vt = np.linalg.svd(vectors, full_matrices=False)
    rank = int(np.sum(s > 1e-10))
    return Vt[:rank]


def subspace_membership(direction: np.ndarray, basis: np.ndarray) -> float:
    """Fraction of direction's variance explained by the subspace.

    Args:
        direction: Unit vector ``(dim,)``.
        basis: ``(rank, dim)`` orthonormal basis.

    Returns:
        Float in ``[0, 1]``: 1 means direction lies entirely in the subspace.
    """
    proj = basis @ direction
    return float(np.dot(proj, proj))


def null_subspace_membership(
    hidden_dim: int,
    subspace_dim: int,
    n_samples: int = 10000,
    seed: int = 42,
) -> dict:
    """Expected membership of random unit vectors in a random subspace.

    Useful as a baseline for comparing actual subspace membership scores.

    Returns:
        Dict with ``mean``, ``std``, ``p95``, ``p99``, ``expected_analytic``.
    """
    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_samples):
        random_basis = np.linalg.qr(rng.randn(hidden_dim, subspace_dim))[0].T
        random_dir = rng.randn(hidden_dim)
        random_dir /= np.linalg.norm(random_dir)
        scores.append(subspace_membership(random_dir, random_basis))
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "expected_analytic": subspace_dim / hidden_dim,
    }


def full_subspace_analysis(
    directions: dict[str, dict[int, np.ndarray]],
    target_directions: dict[str, dict[int, np.ndarray]],
    labels: list[str] | None = None,
) -> dict:
    """Project target directions onto the subspace of reference directions.

    For each layer, computes an orthonormal basis from the reference
    directions and measures how much of each target direction lies in
    that subspace.

    Args:
        directions: Reference directions (``group → layer → vector``).
        target_directions: Target directions to project.
        labels: Reference group labels (determines subspace basis).
            If ``None``, uses all groups in ``directions``.

    Returns:
        Dict with per-layer membership scores for each target direction.
    """
    if labels is None:
        labels = sorted(directions.keys())

    all_layers = set()
    for g in labels:
        all_layers |= set(directions.get(g, {}).keys())
    for g in target_directions:
        all_layers &= set(target_directions[g].keys())

    per_layer: dict[int, dict] = {}
    for layer in sorted(all_layers):
        ref_vecs = []
        for label in labels:
            if label in directions and layer in directions[label]:
                v = directions[label][layer]
                v = v / (np.linalg.norm(v) + 1e-12)
                ref_vecs.append(v)

        if len(ref_vecs) < 2:
            continue

        basis = orthonormal_basis(np.stack(ref_vecs))
        memberships: dict[str, float] = {}
        for tgt_label, tgt_layers in target_directions.items():
            if layer in tgt_layers:
                d = tgt_layers[layer]
                d = d / (np.linalg.norm(d) + 1e-12)
                memberships[tgt_label] = round(subspace_membership(d, basis), 6)

        per_layer[layer] = {
            "subspace_dim": basis.shape[0],
            "memberships": memberships,
            "mean_membership": round(
                float(np.mean(list(memberships.values()))), 6
            ) if memberships else 0.0,
        }

    return {"per_layer": per_layer}
