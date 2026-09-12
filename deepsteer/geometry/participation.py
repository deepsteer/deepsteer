"""Participation ratio with bootstrap CIs and null-referenced normalization.

Pre-registered use: D3 PREREGISTRATION Amendment 14.6(a) (2026-09-10). The methods note's Fig 1
and Table 1 report participation ratios as point estimates and gate on an absolute ``PR < 30`` (with
a separate ``25`` ceiling for GPT-OSS). This module supplies what the self-review asked for:
bootstrap CIs on every PR, and dimension- and sample-size-normalized forms so the gate can be stated
as a null-referenced quantile rather than a panel-fit constant.

Definition (ANOMALIES A1/A2, D2-09): ``PR = (sum λ)^2 / sum λ^2`` over the eigenvalues of the
**centered covariance** of the activation sample (never the diagonal-variance PR, which z-scoring
trivially sets to ``d``).

Pure numpy, model-agnostic.
"""

from __future__ import annotations

import numpy as np


def participation_ratio(X: np.ndarray) -> float:
    """Covariance-eigenvalue participation ratio of a ``(n, d)`` sample (centered).

    Uses the singular values of the centered matrix, so it is exact for ``n < d`` (at most ``n-1``
    nonzero eigenvalues) without forming the ``d x d`` covariance.
    """
    Xc = np.asarray(X, np.float64)
    Xc = Xc - Xc.mean(0)
    n, d = Xc.shape
    if n < d:
        # eigenvalues of the n x n Gram matrix == nonzero squared singular values of Xc; ~10x
        # cheaper than the (n, d) SVD at n=240, d=4096 (the 14.6 bootstrap runs this 2000x per cell)
        s = np.linalg.eigvalsh(Xc @ Xc.T)
    else:
        s = np.linalg.svd(Xc, compute_uv=False) ** 2
    s = s[s > 1e-12 * max(1.0, float(s.max()))] if s.size else s
    if s.size == 0:
        return 0.0
    return float(s.sum() ** 2 / (s ** 2).sum())


def bootstrap_pr(
    X: np.ndarray,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Row-resampled bootstrap of the participation ratio.

    Returns the point estimate, percentile 95% CI, and the bootstrap array. Bias direction: rows
    resampled with replacement contain duplicates, which lowers the effective sample rank and
    biases the bootstrap PR slightly **downward** (favors "bottleneck" readings); the CI is therefore
    conservative for a ``PR >= floor`` claim and anti-conservative for a ``PR < floor`` claim — stated
    in ``bias_note``.
    """
    rng = rng or np.random.default_rng(0)
    Xa = np.asarray(X, np.float64)
    n = Xa.shape[0]
    if n < 3:
        raise ValueError("bootstrap_pr needs >= 3 rows")
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = participation_ratio(Xa[rng.integers(0, n, n)])
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"pr": participation_ratio(Xa), "ci95": [lo, hi], "n": int(n), "d": int(Xa.shape[1]),
            "n_boot": int(n_boot), "per_boot": boots,
            "bias_note": "row-resampling duplicates lower effective rank: bootstrap PR biased low"}


def normalized_pr(pr: float, d: int, n: int) -> dict:
    """Dimension- and sample-normalized forms of a PR.

    ``pr_over_d`` compares across models with different hidden sizes; ``pr_over_n_minus_1`` is the
    fraction of the sample-rank ceiling (a PR near ``n-1`` is sample-limited, not a property of the
    position).
    """
    return {"pr": float(pr), "pr_over_d": float(pr) / float(d),
            "pr_over_n_minus_1": float(pr) / max(1.0, float(n - 1)),
            "sample_rank_ceiling": int(min(n - 1, d))}


def pr_gaussian_null(
    X: np.ndarray,
    *,
    n_draws: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """PR of covariance-matched Gaussian samples of the same ``n``: the sampling-noise reference.

    Draws ``n`` rows from ``N(0, Σ̂)`` (Σ̂ from the centered sample, via its SVD so ``n < d`` is fine)
    and records the PR. A measured PR **inside** this distribution says the sample carries no
    dimensionality signal beyond its own covariance estimate; the quantile of the measured PR in
    this distribution is the null-referenced statistic.
    """
    rng = rng or np.random.default_rng(0)
    Xa = np.asarray(X, np.float64)
    Xc = Xa - Xa.mean(0)
    n = Xc.shape[0]
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    scale = S / np.sqrt(max(n - 1, 1))  # sqrt eigenvalues of Σ̂ along Vt
    draws = np.empty(n_draws)
    for k in range(n_draws):
        Z = rng.standard_normal((n, S.size))
        draws[k] = participation_ratio(Z * scale @ Vt)
    pr = participation_ratio(Xa)
    return {"pr": pr, "null_q05": float(np.percentile(draws, 5)), "null_q50": float(np.median(draws)),
            "null_q95": float(np.percentile(draws, 95)),
            "quantile_of_measured": float((draws < pr).mean()), "n_draws": int(n_draws), "per_draw": draws}


def pr_shuffle_null(
    X: np.ndarray,
    *,
    n_draws: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """PR after destroying cross-dimension structure by independent column permutations.

    Permuting each column independently preserves every marginal (per-dimension variance) and kills
    the correlations; the resulting PR is the **full-rank-for-these-marginals** reference. A measured
    PR far below it is a genuine low-rank (bottleneck) structure, not a marginal-variance artifact.
    """
    rng = rng or np.random.default_rng(0)
    Xa = np.asarray(X, np.float64)
    n, d = Xa.shape
    draws = np.empty(n_draws)
    for k in range(n_draws):
        # independent column permutations, vectorized (argsort of random keys per column)
        idx = np.argsort(rng.random((n, d)), axis=0)
        draws[k] = participation_ratio(np.take_along_axis(Xa, idx, axis=0))
    return {"pr": participation_ratio(Xa), "shuffle_q05": float(np.percentile(draws, 5)),
            "shuffle_q50": float(np.median(draws)), "n_draws": int(n_draws), "per_draw": draws}


def pr_profile(
    X: np.ndarray,
    *,
    n_boot: int = 2000,
    n_null: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """The full Amendment-14.6(a) record for one (model, position, normalization) cell.

    Bootstrap CI + normalized forms + Gaussian and shuffle nulls, arrays dropped (JSON-ready).
    """
    rng = rng or np.random.default_rng(0)
    Xa = np.asarray(X, np.float64)
    b = bootstrap_pr(Xa, n_boot=n_boot, rng=rng)
    g = pr_gaussian_null(Xa, n_draws=n_null, rng=rng)
    s = pr_shuffle_null(Xa, n_draws=n_null, rng=rng)
    out = {**normalized_pr(b["pr"], Xa.shape[1], Xa.shape[0]),
           "ci95": b["ci95"], "gaussian_null": {k: v for k, v in g.items() if k != "per_draw"},
           "shuffle_null": {k: v for k, v in s.items() if k != "per_draw"}}
    return out
