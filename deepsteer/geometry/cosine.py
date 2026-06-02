"""Cosine similarity matrices and effective dimensionality.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
"""

from __future__ import annotations

import numpy as np


def compute_cosine_matrix(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    labels: list[str] | None = None,
) -> np.ndarray | None:
    """Cosine similarity matrix for directions at a given layer.

    Args:
        directions: ``group → layer → unit direction vector``.
        layer: Layer index.
        labels: Group labels to include (in order). If ``None``, uses
            all groups in ``directions`` sorted alphabetically.

    Returns:
        ``(n, n)`` cosine similarity matrix, or ``None`` if any direction
        is missing at this layer.
    """
    if labels is None:
        labels = sorted(directions.keys())
    vecs = []
    for label in labels:
        if label not in directions or layer not in directions[label]:
            return None
        vecs.append(directions[label][layer])
    mat = np.stack(vecs)
    return mat @ mat.T


def compute_effective_dimensionality(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int,
    labels: list[str] | None = None,
    threshold: float = 0.9,
) -> int | None:
    """Number of SVD components to explain *threshold* variance of directions.

    Args:
        directions: ``group → layer → unit direction vector``.
        layer: Layer index.
        labels: Group labels to include. If ``None``, uses all groups.
        threshold: Cumulative variance fraction to target.

    Returns:
        Number of components, or ``None`` if directions are missing.
    """
    if labels is None:
        labels = sorted(directions.keys())
    vecs = []
    for label in labels:
        if label not in directions or layer not in directions[label]:
            return None
        vecs.append(directions[label][layer])
    mat = np.stack(vecs)
    mat_centered = mat - mat.mean(axis=0, keepdims=True)
    _, s, _ = np.linalg.svd(mat_centered, full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    return int(np.searchsorted(explained, threshold)) + 1
