"""Bootstrap estimators over items x rollouts (numpy only).

Arrays are ``[n_items, n_rollouts]`` with ``NaN`` marking a rollout that produced no value (an
unparsed reply). Items are fixed unless a function says otherwise: a replication cell conditions
on its items, so its sampling noise is rollout noise (MB_PLAN v0.5 §5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

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


def nested_bootstrap(
    units: Sequence, stat: Callable[[object, np.random.Generator], float | None], *,
    n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Replicates of the mean of ``stat`` over units resampled with replacement.

    Each replicate draws the outer resample first (``rng.integers(0, n, n)``), then calls
    ``stat(unit, rng)`` once per drawn unit in order, so any inner resampling ``stat`` does
    consumes the same stream; ``None`` values are dropped, and a replicate with none left is NaN.
    (Lifted from ``deepsteer.kdg.stats.kdg_rate``; the draw order is part of the contract: KDG's
    intervals of record depend on it.)
    """
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(units), len(units))
        vals = [v for v in (stat(units[i], rng) for i in idx) if v is not None]
        boots[b] = np.mean(vals) if vals else np.nan
    return boots


def nested_bootstrap_difference(
    a: Sequence, b: Sequence, stat: Callable[[object, np.random.Generator], float | None], *,
    n_boot: int, rng: np.random.Generator,
) -> np.ndarray:
    """Replicates of mean(stat over a resample of ``a``) minus the same over ``b`` (independent
    samples), with ``a``'s draws (outer, then inner) before ``b``'s in each replicate."""
    boots = np.empty(n_boot)
    for i in range(n_boot):
        va = [v for v in (stat(a[j], rng) for j in rng.integers(0, len(a), len(a)))
              if v is not None]
        vb = [v for v in (stat(b[j], rng) for j in rng.integers(0, len(b), len(b)))
              if v is not None]
        boots[i] = (np.mean(va) - np.mean(vb)) if va and vb else np.nan
    return boots
