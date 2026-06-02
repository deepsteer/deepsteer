"""High-level geometric analysis orchestrator.

Combines cosine matrix, effective dimensionality, permutation testing,
and hierarchical clustering into a single analysis report.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
"""

from __future__ import annotations

import numpy as np

from deepsteer.foundations import (
    BINDING_IDX,
    FOUNDATION_ORDER,
    FOUNDATION_SHORT,
    INDIVIDUALIZING_IDX,
)
from deepsteer.geometry.clustering import hierarchical_cluster, permutation_test
from deepsteer.geometry.cosine import compute_cosine_matrix, compute_effective_dimensionality


def full_geometric_analysis(
    directions: dict[str, dict[int, np.ndarray]],
    layer: int | None = None,
    labels: list[str] | None = None,
    groups: dict[str, list[int]] | None = None,
) -> dict | None:
    """Run cosine matrix, effective dim, permutation test, and dendrogram check.

    This is the generalized version of the paper's ``full_geometric_analysis``.
    When called without explicit labels/groups, it falls back to the standard
    MFT foundation ordering and individualizing/binding group structure.

    Args:
        directions: ``group → layer → unit direction vector``.
        layer: Layer index. If ``None``, uses the first available layer.
        labels: Group labels (in order). Defaults to ``FOUNDATION_ORDER``.
        groups: Named groups for clustering and permutation testing.
            Defaults to ``{"individualizing": [0,1,2], "binding": [3,4,5]}``.

    Returns:
        Dict with ``mean_cosine_similarity``, ``effective_dimensionality``,
        ``permutation_test``, ``dendrogram``, ``cosine_matrix``.
        Returns ``None`` if directions are incomplete at this layer.
    """
    if labels is None:
        labels = FOUNDATION_ORDER
    if groups is None:
        groups = {
            "individualizing": INDIVIDUALIZING_IDX,
            "binding": BINDING_IDX,
        }
    if layer is None:
        common_layers = None
        for label in labels:
            if label in directions:
                layer_set = set(directions[label].keys())
                common_layers = layer_set if common_layers is None else common_layers & layer_set
        if not common_layers:
            return None
        layer = min(common_layers)

    cos_sim = compute_cosine_matrix(directions, layer, labels)
    if cos_sim is None:
        return None

    n = len(labels)
    upper_tri = [cos_sim[i, j] for i in range(n) for j in range(i + 1, n)]

    # Permutation test with the first pair of groups
    group_names = list(groups.keys())
    perm_result = {}
    if len(group_names) >= 2:
        perm_result = permutation_test(
            cos_sim, groups[group_names[0]], groups[group_names[1]],
        )

    dendrogram = hierarchical_cluster(cos_sim, labels, groups)

    return {
        "mean_cosine_similarity": round(float(np.mean(upper_tri)), 6),
        "effective_dimensionality": compute_effective_dimensionality(directions, layer, labels),
        "permutation_test": perm_result,
        "dendrogram": dendrogram,
        "cosine_matrix": cos_sim.tolist(),
    }
