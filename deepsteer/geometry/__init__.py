"""Geometric analysis of concept direction spaces.

Tier: Validated (Paper 3).

Provides tools for analyzing the geometric structure of concept directions
in representation space: cosine similarity matrices, clustering, subspace
analysis, and statistical tests for group structure.

All functions are pure numpy — model-agnostic by design.

Usage::

    from deepsteer.geometry import full_geometric_analysis
    geo = full_geometric_analysis(directions, labels=list(directions.keys()))
"""

from __future__ import annotations

from deepsteer.geometry.analysis import full_geometric_analysis
from deepsteer.geometry.clustering import (
    hierarchical_cluster,
    permutation_test,
    permutation_test_mft,
)
from deepsteer.geometry.cosine import (
    compute_cosine_matrix,
    compute_effective_dimensionality,
)
from deepsteer.geometry.participation import (
    bootstrap_pr,
    normalized_pr,
    participation_ratio,
    pr_gaussian_null,
    pr_profile,
    pr_shuffle_null,
)
from deepsteer.geometry.reliability import (
    adjacent_self_cosine,
    disattenuate,
    disattenuate_bootstrap,
    permutation_self_cosine_null,
    spearman_brown,
    split_half_self_cosine,
)
from deepsteer.geometry.subspace import (
    full_subspace_analysis,
    orthonormal_basis,
    subspace_membership,
)

__all__ = [
    "compute_cosine_matrix",
    "compute_effective_dimensionality",
    "hierarchical_cluster",
    "permutation_test",
    "permutation_test_mft",
    "orthonormal_basis",
    "subspace_membership",
    "full_subspace_analysis",
    "full_geometric_analysis",
    "participation_ratio",
    "bootstrap_pr",
    "normalized_pr",
    "pr_gaussian_null",
    "pr_shuffle_null",
    "pr_profile",
    "split_half_self_cosine",
    "spearman_brown",
    "disattenuate",
    "disattenuate_bootstrap",
    "permutation_self_cosine_null",
    "adjacent_self_cosine",
]
