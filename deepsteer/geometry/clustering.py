"""Hierarchical clustering and permutation tests on direction similarity.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage

from deepsteer.foundations import (
    BINDING_IDX,
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING_IDX,
)


def hierarchical_cluster(
    cos_sim: np.ndarray,
    labels: list[str],
    groups: dict[str, list[int]] | None = None,
) -> dict:
    """Ward hierarchical clustering on a cosine similarity matrix.

    Args:
        cos_sim: ``(n, n)`` cosine similarity matrix.
        labels: Labels for each row/column.
        groups: Optional named groups mapping label to indices.
            If provided, checks whether the top-level split separates
            each group.

    Returns:
        Dict with ``"left"``, ``"right"`` cluster labels and optional
        per-group ``"group_match"`` booleans.
    """
    n = cos_sim.shape[0]
    dist = 1 - cos_sim
    condensed = [dist[i, j] for i in range(n) for j in range(i + 1, n)]
    Z = linkage(condensed, method="ward")

    def _get_leaves(idx: int) -> set[int]:
        if idx < n:
            return {idx}
        row = Z[idx - n]
        return _get_leaves(int(row[0])) | _get_leaves(int(row[1]))

    last = Z[-1]
    left = _get_leaves(int(last[0]))
    right = _get_leaves(int(last[1]))
    left_labels = [labels[i] for i in sorted(left)]
    right_labels = [labels[i] for i in sorted(right)]

    result: dict = {
        "left": left_labels,
        "right": right_labels,
    }

    if groups:
        for group_name, group_idx in groups.items():
            group_set = set(group_idx)
            result[f"{group_name}_match"] = left == group_set or right == group_set

    return result


def permutation_test(
    cos_sim: np.ndarray,
    group_a: list[int],
    group_b: list[int],
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """Test whether two groups have higher within-group than between-group similarity.

    Args:
        cos_sim: ``(n, n)`` cosine similarity matrix.
        group_a: Indices of group A.
        group_b: Indices of group B.
        n_perm: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with ``observed_statistic``, ``p_value``, and group means.
    """
    n = cos_sim.shape[0]

    def _stat(sim: np.ndarray, ga: list[int], gb: list[int]) -> float:
        wa = [sim[i, j] for i in ga for j in ga if i < j]
        wb = [sim[i, j] for i in gb for j in gb if i < j]
        bw = [sim[i, j] for i in ga for j in gb]
        return float(np.mean(wa + wb) - np.mean(bw)) if (wa + wb) and bw else 0.0

    observed = _stat(cos_sim, group_a, group_b)
    rng = np.random.RandomState(seed)
    count = 0
    total = len(group_a) + len(group_b)
    for _ in range(n_perm):
        p = rng.permutation(n)
        if _stat(cos_sim, p[:len(group_a)].tolist(), p[len(group_a):total].tolist()) >= observed:
            count += 1
    p_value = (count + 1) / (n_perm + 1)

    within_a = [cos_sim[i, j] for i in group_a for j in group_a if i < j]
    within_b = [cos_sim[i, j] for i in group_b for j in group_b if i < j]
    between = [cos_sim[i, j] for i in group_a for j in group_b]

    return {
        "observed_statistic": float(observed),
        "p_value": float(p_value),
        "mean_within_a": float(np.mean(within_a)) if within_a else 0.0,
        "mean_within_b": float(np.mean(within_b)) if within_b else 0.0,
        "mean_between": float(np.mean(between)) if between else 0.0,
    }


def permutation_test_mft(
    cos_sim: np.ndarray,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """MFT-specific permutation test: individualizing vs binding groups.

    Convenience wrapper around :func:`permutation_test` using the standard
    MFT group indices (assumes FOUNDATION_ORDER ordering).

    Returns:
        Dict with ``observed_statistic``, ``p_value``,
        ``mean_within_individualizing``, ``mean_within_binding``,
        ``mean_between_groups``.
    """
    result = permutation_test(cos_sim, INDIVIDUALIZING_IDX, BINDING_IDX, n_perm, seed)
    return {
        "observed_statistic": result["observed_statistic"],
        "p_value": result["p_value"],
        "mean_within_individualizing": result["mean_within_a"],
        "mean_within_binding": result["mean_within_b"],
        "mean_between_groups": result["mean_between"],
    }
