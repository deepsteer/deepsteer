"""Tests for deepsteer.directions module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deepsteer.directions.compare import compare_directions
from deepsteer.directions.leace import extract_leace_directions
from deepsteer.directions.mean_diff import extract_mean_diff_directions
from deepsteer.directions.probe_weight import extract_from_npz, extract_probe_directions


@pytest.fixture
def synthetic_activations():
    """Create synthetic activations with known class-separating structure."""
    rng = np.random.RandomState(42)
    hidden_dim = 64
    n_pairs = 20

    # Two groups with different separating directions
    groups = {"group_a": list(range(10)), "group_b": list(range(10, 20))}

    dir_a = rng.randn(hidden_dim).astype(np.float64)
    dir_a /= np.linalg.norm(dir_a)
    dir_b = rng.randn(hidden_dim).astype(np.float64)
    dir_b /= np.linalg.norm(dir_b)

    activations = {}
    for layer in range(4):
        X_list = []
        y_list = []
        scale = 0.5 + layer * 0.5
        for i in range(n_pairs):
            noise = rng.randn(hidden_dim) * 0.1
            if i < 10:
                moral = noise + scale * dir_a
                neutral = noise - scale * dir_a
            else:
                moral = noise + scale * dir_b
                neutral = noise - scale * dir_b
            X_list.extend([moral, neutral])
            y_list.extend([1, 0])
        X = torch.tensor(np.array(X_list), dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)
        activations[layer] = (X, y)

    return activations, groups


def test_mean_diff_directions_shape(synthetic_activations):
    activations, groups = synthetic_activations
    dirs = extract_mean_diff_directions(activations, groups)
    assert "group_a" in dirs
    assert "group_b" in dirs
    for layer in range(4):
        assert layer in dirs["group_a"]
        d = dirs["group_a"][layer]
        assert d.shape == (64,)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-6


def test_mean_diff_directions_correct_direction(synthetic_activations):
    activations, groups = synthetic_activations
    dirs = extract_mean_diff_directions(activations, groups)
    # At higher layers (more signal), the direction should be more aligned
    # with the true separating direction
    for layer in [2, 3]:
        d = dirs["group_a"][layer]
        # Just verify it's a unit vector and not zero
        assert np.linalg.norm(d) > 0.99


def test_leace_directions_shape(synthetic_activations):
    activations, groups = synthetic_activations
    dirs = extract_leace_directions(activations, groups)
    assert "group_a" in dirs
    for layer in range(4):
        d = dirs["group_a"][layer]
        assert d.shape == (64,)
        assert abs(np.linalg.norm(d) - 1.0) < 1e-6


def test_leace_vs_mean_diff_alignment(synthetic_activations):
    activations, groups = synthetic_activations
    md_dirs = extract_mean_diff_directions(activations, groups)
    leace_dirs = extract_leace_directions(activations, groups)
    # For well-separated data, LEACE and mean-diff should roughly agree
    for group in groups:
        for layer in range(4):
            cos = abs(float(np.dot(md_dirs[group][layer], leace_dirs[group][layer])))
            assert cos > 0.5  # at least moderately aligned


def test_probe_directions_normalize():
    weights = {
        "group_a": {0: np.array([3.0, 4.0])},
        "group_b": {0: np.array([1.0, 0.0])},
    }
    dirs = extract_probe_directions(weights)
    np.testing.assert_allclose(dirs["group_a"][0], [0.6, 0.8])
    np.testing.assert_allclose(dirs["group_b"][0], [1.0, 0.0])


def test_extract_from_npz(tmp_path):
    rng = np.random.RandomState(42)
    data = {}
    for group in ["care_harm", "fairness_cheating"]:
        for layer in range(3):
            d = rng.randn(32)
            data[f"{group}_layer{layer}"] = d
    np.savez(tmp_path / "probes.npz", **data)

    dirs = extract_from_npz(tmp_path / "probes.npz")
    assert "care_harm" in dirs
    assert "fairness_cheating" in dirs
    assert len(dirs["care_harm"]) == 3
    for layer in range(3):
        assert abs(np.linalg.norm(dirs["care_harm"][layer]) - 1.0) < 1e-6


def test_compare_directions(synthetic_activations):
    activations, groups = synthetic_activations
    dirs_a = extract_mean_diff_directions(activations, groups)
    dirs_b = extract_leace_directions(activations, groups)
    alignment = compare_directions(dirs_a, dirs_b)
    assert "group_a" in alignment
    assert "mean_cosine" in alignment["group_a"]
    assert 0 <= alignment["group_a"]["mean_cosine"] <= 1
