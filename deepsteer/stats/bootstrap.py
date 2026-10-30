"""Bootstrap estimators over items x rollouts (numpy only).

Arrays are ``[n_items, n_rollouts]`` with ``NaN`` marking a rollout that produced no value (an
unparsed reply). Items are fixed unless a function says otherwise: a replication cell conditions
on its items, so its sampling noise is rollout noise (MB_PLAN v0.5 §5).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def k_repeat_prediction_interval(
    scores: np.ndarray,
    k: int = 5,
    *,
    n_draws: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, np.ndarray]:
    """Prediction interval for a set score that averages ``k`` repeats per item.

    Each draw takes ``k`` rollouts per item with replacement (items fixed) and sums the per-item
    means. ``NaN`` entries count as 0 (the paper-comparison aggregate: a miss scores 0).

    Returns:
        ``(lo, hi, draws)``: the ``alpha/2`` and ``1 - alpha/2`` quantiles and the draws.
    """
    x = np.nan_to_num(np.asarray(scores, dtype=np.float64), nan=0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[1], size=(n_draws, x.shape[0], k))
    draws = np.take_along_axis(x[None, :, :], idx, axis=2).mean(axis=2).sum(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), draws


def resample_rollouts(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One bootstrap replicate: rollouts resampled with replacement within each item."""
    idx = rng.integers(0, x.shape[1], size=x.shape)
    return np.take_along_axis(x, idx, axis=1)


def rollout_bootstrap_ci(
    stat: Callable[..., float],
    *arrays: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Point estimate and percentile CI of ``stat(*arrays)``, resampling rollouts within items
    independently in each array (items fixed; arrays are independent samples).

    Returns:
        ``(estimate, lo, hi)``.
    """
    rng = np.random.default_rng(seed)
    est = float(stat(*arrays))
    reps = np.array([stat(*(resample_rollouts(a, rng) for a in arrays)) for _ in range(n_boot)])
    reps = reps[np.isfinite(reps)]
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)
