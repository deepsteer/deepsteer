"""Tests for deepsteer.geometry.participation (Amendment 14.6(a) PR bootstrap + nulls)."""

from __future__ import annotations

import numpy as np
import pytest

from deepsteer.geometry.participation import (
    bootstrap_pr,
    normalized_pr,
    participation_ratio,
    pr_gaussian_null,
    pr_profile,
    pr_shuffle_null,
)


def _lowrank(n: int = 200, d: int = 64, k: int = 5, seed: int = 0) -> np.ndarray:
    """A sample whose covariance has k equal dominant eigenvalues plus tiny isotropic noise."""
    rng = np.random.default_rng(seed)
    B = np.linalg.qr(rng.standard_normal((d, k)))[0]
    return rng.standard_normal((n, k)) @ B.T * 10.0 + rng.standard_normal((n, d)) * 0.01


class TestParticipationRatio:
    def test_rank_k_isotropic_gives_k(self):
        # assert PR ~ k for k equal eigenvalues (the bottleneck reading is calibrated)
        assert participation_ratio(_lowrank(k=5)) == pytest.approx(5.0, abs=0.3)

    def test_matches_covariance_definition(self):
        X = np.random.default_rng(1).standard_normal((50, 12))
        ev = np.linalg.eigvalsh(np.cov(X - X.mean(0), rowvar=False))
        ev = ev[ev > 1e-12]
        assert participation_ratio(X) == pytest.approx(ev.sum() ** 2 / (ev ** 2).sum(), rel=1e-6)

    def test_pr_never_exceeds_sample_rank(self):
        # assert the n < d ceiling (n-1): the sample-limited case reads as sample-limited
        X = np.random.default_rng(2).standard_normal((10, 100))
        assert participation_ratio(X) <= 9.0 + 1e-9


class TestBootstrapAndNulls:
    def test_bootstrap_ci_contains_point_and_is_biased_low(self):
        # assert point inside CI and the documented downward bias of row-resampling (median <= point)
        X = _lowrank(k=5)
        b = bootstrap_pr(X, n_boot=300, rng=np.random.default_rng(0))
        assert b["ci95"][0] <= b["pr"] <= b["ci95"][1] + 1e-9
        assert np.median(b["per_boot"]) <= b["pr"] + 0.05

    def test_bootstrap_rejects_tiny_sample(self):
        with pytest.raises(ValueError):
            bootstrap_pr(np.zeros((2, 4)))

    def test_normalized_forms(self):
        out = normalized_pr(12.8, d=2880, n=128)
        assert out["pr_over_d"] == pytest.approx(12.8 / 2880)
        assert out["pr_over_n_minus_1"] == pytest.approx(12.8 / 127)
        assert out["sample_rank_ceiling"] == 127

    def test_gaussian_null_brackets_gaussian_data(self):
        # assert covariance-matched Gaussian draws reproduce the PR of Gaussian data (quantile mid)
        X = _lowrank(k=5, n=400)
        g = pr_gaussian_null(X, n_draws=100, rng=np.random.default_rng(0))
        assert g["null_q05"] <= g["pr"] <= g["null_q95"] + 0.5
        assert 0.0 <= g["quantile_of_measured"] <= 1.0

    def test_shuffle_null_is_full_rank_reference(self):
        # assert destroying correlations raises PR far above a rank-5 bottleneck (the tell)
        X = _lowrank(k=5, n=200, d=64)
        s = pr_shuffle_null(X, n_draws=20, rng=np.random.default_rng(0))
        assert s["shuffle_q05"] > 3 * s["pr"]

    def test_profile_is_json_ready(self):
        import json

        prof = pr_profile(_lowrank(k=5, n=60, d=32), n_boot=50, n_null=20, rng=np.random.default_rng(0))
        json.dumps(prof)  # no numpy arrays leak into the record
        assert set(prof) >= {"pr", "pr_over_d", "ci95", "gaussian_null", "shuffle_null"}


class TestGramPath:
    def test_gram_equals_svd_for_n_less_than_d(self):
        # assert the n<d Gram-matrix shortcut (14.6 speedup) reproduces the SVD definition exactly
        X = np.random.default_rng(3).standard_normal((60, 500)) * np.linspace(3, 0.1, 500)
        Xc = X - X.mean(0)
        s = np.linalg.svd(Xc, compute_uv=False) ** 2
        assert participation_ratio(X) == pytest.approx(s.sum() ** 2 / (s ** 2).sum(), rel=1e-9)

    def test_shuffle_null_preserves_marginals(self):
        # assert the vectorized column permutation keeps every per-column variance (marginals intact)
        X = _lowrank(k=3)
        rng = np.random.default_rng(0)
        idx = np.argsort(rng.random(X.shape), axis=0)
        Xs = np.take_along_axis(X, idx, axis=0)
        assert np.allclose(np.sort(Xs, 0), np.sort(X, 0))
