"""Reliability ceilings for mean-difference directions (split-half, Spearman–Brown, disattenuation).

Pre-registered use: D3 PREREGISTRATION Amendment 14.1 (2026-09-10). The proto-refusal→gate cosine
of record (0.155) has no reliability ceiling under it: a low cosine between two noisy directions can
be attenuation rather than genuine discontinuity. These functions put the ceiling in place.

All functions are pure numpy and model-agnostic. Inputs are per-sample activation matrices (rows =
prompts) for the two contrast classes; the direction is the unit mean difference.

Estimator notes (estimator-traps): the split-half self-cosine is an estimate of the **half-length**
reliability; :func:`spearman_brown` maps it to full length. Resampling attenuates cosines toward 0,
so the split-half median is a conservative (downward-biased) ceiling; the bias direction favors the
"attenuation floor" reading, not the "fresh construction" reading, and is stated with the number.
"""

from __future__ import annotations

import numpy as np


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def mean_diff(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Unit ``mean(pos) - mean(neg)`` over rows."""
    return _unit(np.asarray(pos, np.float64).mean(0) - np.asarray(neg, np.float64).mean(0))


def split_half_self_cosine(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_splits: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Split-half reliability of a mean-difference direction.

    Each split partitions the ``pos`` rows and the ``neg`` rows independently into two halves
    (paired: every half-direction uses half of each class), builds the unit mean-diff on each half,
    and records ``cos(d_A, d_B)``. Returns the median, mean, percentile 95% CI over splits, the
    Spearman–Brown full-length reliability of the mean, and the raw per-split array.

    Args:
        pos: ``(n_pos, d)`` activations of the positive class.
        neg: ``(n_neg, d)`` activations of the negative class.
        n_splits: number of random half-splits.
        rng: numpy Generator (seeded by the caller).

    Raises:
        ValueError: if either class has fewer than 4 rows (no meaningful halves).
    """
    rng = rng or np.random.default_rng(0)
    P = np.asarray(pos, np.float64)
    N = np.asarray(neg, np.float64)
    if P.shape[0] < 4 or N.shape[0] < 4:
        raise ValueError(f"split-half needs >= 4 rows per class; got {P.shape[0]} / {N.shape[0]}")
    r = np.empty(n_splits)
    for s in range(n_splits):
        ip = rng.permutation(P.shape[0])
        ineg = rng.permutation(N.shape[0])
        hp, hn = P.shape[0] // 2, N.shape[0] // 2
        dA = mean_diff(P[ip[:hp]], N[ineg[:hn]])
        dB = mean_diff(P[ip[hp:]], N[ineg[hn:]])
        r[s] = float(dA @ dB)
    lo, hi = (float(x) for x in np.percentile(r, [2.5, 97.5]))
    mean_r = float(r.mean())
    return {
        "median": float(np.median(r)),
        "mean": mean_r,
        "ci95": [lo, hi],
        "spearman_brown_full": spearman_brown(mean_r),
        "n_splits": int(n_splits),
        "n_pos": int(P.shape[0]),
        "n_neg": int(N.shape[0]),
        "per_split": r,
        "bias_note": "resampling attenuates cosines toward 0: this ceiling is conservative (low)",
    }


def spearman_brown(r_half: float, factor: float = 2.0) -> float:
    """Spearman–Brown prophecy: reliability at ``factor``× the length from a half-length ``r_half``.

    ``r_full = factor*r / (1 + (factor-1)*r)``. Clipped to ``[-1, 1]``; a negative half-length
    reliability maps to a negative (uninformative) full-length value rather than raising.
    """
    r = float(r_half)
    denom = 1.0 + (factor - 1.0) * r
    if abs(denom) < 1e-12:
        return 1.0 if r > 0 else -1.0
    return float(np.clip(factor * r / denom, -1.0, 1.0))


def disattenuate(cos_observed: float, rel_a: float, rel_b: float) -> float:
    """Classical disattenuation ``cos / sqrt(rel_a * rel_b)``.

    Returns ``nan`` if either reliability is non-positive (no correction is defined there); a value
    above 1 is clipped to 1 and should be reported as "at ceiling".
    """
    if rel_a <= 0 or rel_b <= 0:
        return float("nan")
    return float(min(1.0, cos_observed / np.sqrt(rel_a * rel_b)))


def disattenuate_bootstrap(
    cos_observed: float,
    per_split_a: np.ndarray,
    per_split_b: np.ndarray,
    *,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Propagate the two split-half distributions into a CI on the disattenuated cosine.

    Each bootstrap draw takes one split-half value from each side, maps both through Spearman–Brown,
    and disattenuates. Draws where either reliability is non-positive are dropped and counted.
    """
    rng = rng or np.random.default_rng(0)
    A = np.asarray(per_split_a, np.float64)
    B = np.asarray(per_split_b, np.float64)
    vals = []
    dropped = 0
    for _ in range(n_boot):
        ra = spearman_brown(float(rng.choice(A)))
        rb = spearman_brown(float(rng.choice(B)))
        v = disattenuate(cos_observed, ra, rb)
        if np.isnan(v):
            dropped += 1
        else:
            vals.append(v)
    if not vals:
        return {"point": float("nan"), "ci95": [float("nan"), float("nan")], "n_dropped": dropped}
    point = disattenuate(cos_observed, spearman_brown(float(A.mean())), spearman_brown(float(B.mean())))
    lo, hi = (float(x) for x in np.percentile(vals, [2.5, 97.5]))
    return {"point": point, "ci95": [lo, hi], "n_dropped": int(dropped), "n_boot": int(n_boot)}


def permutation_self_cosine_null(
    pos: np.ndarray,
    neg: np.ndarray,
    *,
    n_perm: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Chance ceiling for the split-half self-cosine: shuffle class labels, then split-half.

    With labels destroyed the two half-directions share no signal, so their cosine is the chance
    level for this n and d (anisotropy included). Returns the q50/q95 and the per-permutation array.
    """
    rng = rng or np.random.default_rng(0)
    X = np.concatenate([np.asarray(pos, np.float64), np.asarray(neg, np.float64)], 0)
    n_pos = int(np.asarray(pos).shape[0])
    r = np.empty(n_perm)
    for k in range(n_perm):
        perm = rng.permutation(X.shape[0])
        P, N = X[perm[:n_pos]], X[perm[n_pos:]]
        ip, ineg = rng.permutation(P.shape[0]), rng.permutation(N.shape[0])
        hp, hn = P.shape[0] // 2, N.shape[0] // 2
        r[k] = float(mean_diff(P[ip[:hp]], N[ineg[:hn]]) @ mean_diff(P[ip[hp:]], N[ineg[hn:]]))
    return {"q50": float(np.median(r)), "q95": float(np.percentile(r, 95)),
            "n_perm": int(n_perm), "per_perm": r}


def adjacent_self_cosine(directions: dict[int, np.ndarray], final_key: int) -> dict[int, float]:
    """``cos(direction[k], direction[final_key])`` for every key (checkpoint trajectory helper)."""
    f = _unit(directions[final_key])
    return {int(k): float(_unit(v) @ f) for k, v in directions.items()}
