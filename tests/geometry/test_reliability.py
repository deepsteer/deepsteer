"""Tests for deepsteer.geometry.reliability (Amendment 14.1 split-half ceiling)."""

from __future__ import annotations

import numpy as np
import pytest

from deepsteer.geometry.reliability import (
    adjacent_self_cosine,
    disattenuate,
    disattenuate_bootstrap,
    mean_diff,
    permutation_self_cosine_null,
    spearman_brown,
    split_half_self_cosine,
)


def _classes(signal: float, n: int = 200, d: int = 64, seed: int = 0):
    """Two classes separated along e_0 by `signal` on top of isotropic unit noise."""
    rng = np.random.default_rng(seed)
    axis = np.zeros(d)
    axis[0] = 1.0
    pos = rng.standard_normal((n, d)) + signal * axis
    neg = rng.standard_normal((n, d)) - signal * axis
    return pos, neg


class TestSplitHalf:
    def test_strong_signal_reads_reliable(self):
        # assert a direction the data determines well gets a high ceiling (the positive control path)
        pos, neg = _classes(signal=3.0)
        r = split_half_self_cosine(pos, neg, n_splits=100, rng=np.random.default_rng(1))
        assert r["median"] > 0.95
        assert r["ci95"][0] <= r["median"] <= r["ci95"][1]
        assert r["spearman_brown_full"] >= r["mean"]

    def test_no_signal_reads_near_chance(self):
        # assert label-free data gives a self-cosine near 0 (the attenuation-floor branch is reachable)
        pos, neg = _classes(signal=0.0)
        r = split_half_self_cosine(pos, neg, n_splits=100, rng=np.random.default_rng(2))
        assert abs(r["median"]) < 0.25

    def test_permutation_null_bounds_no_signal_case(self):
        # assert the permutation null q95 sits above the no-signal split-half median (null has teeth)
        pos, neg = _classes(signal=0.0)
        r = split_half_self_cosine(pos, neg, n_splits=100, rng=np.random.default_rng(3))
        null = permutation_self_cosine_null(pos, neg, n_perm=100, rng=np.random.default_rng(4))
        assert r["median"] <= null["q95"] + 0.05

    def test_strong_signal_clears_permutation_null(self):
        pos, neg = _classes(signal=3.0)
        r = split_half_self_cosine(pos, neg, n_splits=50, rng=np.random.default_rng(5))
        null = permutation_self_cosine_null(pos, neg, n_perm=100, rng=np.random.default_rng(6))
        assert r["median"] > null["q95"]

    def test_rejects_tiny_classes(self):
        # assert the most probable misuse (3 rows per class) fails loud, not silently
        with pytest.raises(ValueError):
            split_half_self_cosine(np.zeros((3, 8)), np.zeros((10, 8)))

    def test_mean_diff_is_unit(self):
        pos, neg = _classes(signal=1.0, n=20, d=16)
        assert abs(np.linalg.norm(mean_diff(pos, neg)) - 1.0) < 1e-9


class TestSpearmanBrownAndDisattenuation:
    def test_spearman_brown_monotone_and_bounded(self):
        # assert r_full >= r_half for positive r and the identity at r = 1
        assert spearman_brown(0.5) == pytest.approx(2 * 0.5 / 1.5)
        assert spearman_brown(1.0) == pytest.approx(1.0)
        assert spearman_brown(0.0) == pytest.approx(0.0)
        assert spearman_brown(0.8) > 0.8

    def test_disattenuate_formula_and_guards(self):
        # assert cos/sqrt(ra*rb), clipping at 1, nan when a reliability is non-positive
        assert disattenuate(0.155, 0.9, 0.9) == pytest.approx(0.155 / 0.9)
        assert disattenuate(0.9, 0.5, 0.5) == 1.0
        assert np.isnan(disattenuate(0.1, 0.0, 0.9))

    def test_disattenuate_bootstrap_contains_point(self):
        rng = np.random.default_rng(0)
        a = rng.uniform(0.7, 0.9, 200)
        b = rng.uniform(0.7, 0.9, 200)
        out = disattenuate_bootstrap(0.155, a, b, n_boot=500, rng=np.random.default_rng(1))
        assert out["ci95"][0] <= out["point"] <= out["ci95"][1]
        assert out["n_dropped"] == 0


class TestAdjacent:
    def test_adjacent_self_cosine_identity_and_orthogonal(self):
        d = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0]), 3: np.array([2.0, 0.0])}
        out = adjacent_self_cosine(d, final_key=3)
        assert out[3] == pytest.approx(1.0)
        assert out[1] == pytest.approx(1.0)
        assert out[2] == pytest.approx(0.0)
