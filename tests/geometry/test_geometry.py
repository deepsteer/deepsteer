"""Tests for deepsteer.geometry module."""

from __future__ import annotations

import numpy as np
import pytest

from deepsteer.geometry.clustering import hierarchical_cluster, permutation_test
from deepsteer.geometry.cosine import compute_cosine_matrix, compute_effective_dimensionality
from deepsteer.geometry.subspace import (
    full_subspace_analysis,
    null_subspace_membership,
    orthonormal_basis,
    subspace_membership,
)
from deepsteer.geometry.analysis import full_geometric_analysis


@pytest.fixture
def synthetic_directions():
    """6 directions with known group structure: 3 + 3 clusters."""
    rng = np.random.RandomState(42)
    dim = 64

    # Cluster A: near [1, 0, 0, ...] + small noise
    base_a = np.zeros(dim)
    base_a[0] = 1.0
    # Cluster B: near [0, 1, 0, ...] + small noise
    base_b = np.zeros(dim)
    base_b[1] = 1.0

    directions = {}
    labels = ["a0", "a1", "a2", "b0", "b1", "b2"]
    for i, label in enumerate(labels):
        directions[label] = {}
        base = base_a if i < 3 else base_b
        for layer in range(4):
            d = base + rng.randn(dim) * 0.1
            d /= np.linalg.norm(d)
            directions[label][layer] = d

    return directions, labels


def test_cosine_matrix_shape(synthetic_directions):
    directions, labels = synthetic_directions
    cos = compute_cosine_matrix(directions, layer=2, labels=labels)
    assert cos is not None
    assert cos.shape == (6, 6)
    np.testing.assert_allclose(np.diag(cos), 1.0, atol=0.01)


def test_cosine_matrix_missing_returns_none():
    directions = {"a": {0: np.array([1.0, 0.0])}}
    result = compute_cosine_matrix(directions, layer=0, labels=["a", "b"])
    assert result is None


def test_cosine_matrix_cluster_structure(synthetic_directions):
    directions, labels = synthetic_directions
    cos = compute_cosine_matrix(directions, layer=2, labels=labels)
    # Within-cluster should be higher than between-cluster
    within_a = [cos[i, j] for i in range(3) for j in range(3) if i < j]
    between = [cos[i, j] for i in range(3) for j in range(3, 6)]
    assert np.mean(within_a) > np.mean(between) + 0.3


def test_effective_dimensionality(synthetic_directions):
    directions, labels = synthetic_directions
    dim = compute_effective_dimensionality(directions, layer=2, labels=labels)
    assert dim is not None
    assert 1 <= dim <= 6


def test_permutation_test(synthetic_directions):
    directions, labels = synthetic_directions
    cos = compute_cosine_matrix(directions, layer=2, labels=labels)
    result = permutation_test(cos, [0, 1, 2], [3, 4, 5], n_perm=500)
    assert "observed_statistic" in result
    assert "p_value" in result
    assert result["p_value"] < 0.1  # should be significant


def test_hierarchical_cluster(synthetic_directions):
    directions, labels = synthetic_directions
    cos = compute_cosine_matrix(directions, layer=2, labels=labels)
    groups = {"cluster_a": [0, 1, 2], "cluster_b": [3, 4, 5]}
    result = hierarchical_cluster(cos, labels, groups)
    assert "left" in result
    assert "right" in result
    assert len(result["left"]) + len(result["right"]) == 6


def test_orthonormal_basis():
    vecs = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0.5, 0.5, 0, 0],  # redundant
    ], dtype=np.float64)
    basis = orthonormal_basis(vecs)
    assert basis.shape[0] == 2  # rank 2
    assert basis.shape[1] == 4
    np.testing.assert_allclose(basis @ basis.T, np.eye(2), atol=1e-10)


def test_subspace_membership():
    # Direction in the subspace
    basis = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    d_in = np.array([1, 0, 0], dtype=np.float64)
    assert abs(subspace_membership(d_in, basis) - 1.0) < 1e-10

    # Direction out of the subspace
    d_out = np.array([0, 0, 1], dtype=np.float64)
    assert abs(subspace_membership(d_out, basis)) < 1e-10


def test_null_subspace_membership():
    result = null_subspace_membership(hidden_dim=100, subspace_dim=5, n_samples=500)
    assert "mean" in result
    assert "expected_analytic" in result
    assert abs(result["mean"] - 5 / 100) < 0.02  # close to dim/D


def test_full_geometric_analysis(synthetic_directions):
    directions, labels = synthetic_directions
    groups = {"cluster_a": [0, 1, 2], "cluster_b": [3, 4, 5]}
    result = full_geometric_analysis(directions, layer=2, labels=labels, groups=groups)
    assert result is not None
    assert "mean_cosine_similarity" in result
    assert "effective_dimensionality" in result
    assert "permutation_test" in result
    assert "dendrogram" in result
    assert "cosine_matrix" in result


def test_full_subspace_analysis(synthetic_directions):
    directions, labels = synthetic_directions
    # Use first 4 as reference, last 2 as targets
    ref_labels = labels[:4]
    ref = {l: directions[l] for l in ref_labels}
    tgt = {l: directions[l] for l in labels[4:]}
    result = full_subspace_analysis(ref, tgt, labels=ref_labels)
    assert "per_layer" in result
    for layer_data in result["per_layer"].values():
        assert "subspace_dim" in layer_data
        assert "memberships" in layer_data
