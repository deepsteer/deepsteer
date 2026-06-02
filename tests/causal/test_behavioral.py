"""Tests for deepsteer.causal.behavioral module."""

from __future__ import annotations

import numpy as np
import pytest

from deepsteer.causal.behavioral import classify_by_projection, projection_classify


@pytest.fixture
def synthetic_classification_data():
    """Activations with clear directional structure for classification."""
    rng = np.random.RandomState(42)
    hidden_dim = 32
    n_items = 12

    # 3 groups, 4 items each, with known direction
    group_dirs = {
        "group_a": np.zeros(hidden_dim),
        "group_b": np.zeros(hidden_dim),
        "group_c": np.zeros(hidden_dim),
    }
    group_dirs["group_a"][0] = 1.0
    group_dirs["group_b"][1] = 1.0
    group_dirs["group_c"][2] = 1.0

    labels = ["group_a"] * 4 + ["group_b"] * 4 + ["group_c"] * 4

    directions = {}
    for name, d in group_dirs.items():
        directions[name] = {0: d, 1: d}

    activations = {}
    for layer in [0, 1]:
        acts = np.zeros((n_items, hidden_dim))
        for i in range(n_items):
            label = labels[i]
            acts[i] = group_dirs[label] * 2.0 + rng.randn(hidden_dim) * 0.1
        activations[layer] = acts

    return activations, directions, labels


def test_projection_classify_perfect(synthetic_classification_data):
    activations, directions, labels = synthetic_classification_data
    result = projection_classify(activations, directions, labels)
    assert result["accuracy"] >= 0.9
    assert result["total"] == 12
    assert "per_group" in result
    assert "confusion_matrix" in result


def test_projection_classify_debiased(synthetic_classification_data):
    activations, directions, labels = synthetic_classification_data
    result = projection_classify(activations, directions, labels, debias=True)
    assert "accuracy" in result


def test_classify_by_projection():
    projections = [
        {"a": {0: 0.5, 1: 0.6}, "b": {0: 0.1, 1: 0.2}},
        {"a": {0: 0.1, 1: 0.2}, "b": {0: 0.5, 1: 0.7}},
    ]
    result = classify_by_projection(projections, layers=[0, 1])
    assert result[0]["predicted"] == "a"
    assert result[1]["predicted"] == "b"


def test_classify_by_projection_debiased():
    projections = [
        {"a": {0: 1.5, 1: 1.6}, "b": {0: 1.1, 1: 1.2}},
    ]
    result = classify_by_projection(projections, layers=[0, 1], debias=True)
    assert result[0]["predicted"] == "a"
