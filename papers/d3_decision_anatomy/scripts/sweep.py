#!/usr/bin/env python3
"""C1 rank sweep + harm identification (Amendment 4). Pure math (model-free, unit-tested); the
per-rank interchange passes run in c1_session behind SWEEP=1.

The under_transfer verdict cannot distinguish "refusal reads non-V_moral features" from "rank-3 is
too small a window." The sweep restricts the content patch to nested moral subspaces of rank
k in {1,3,8,16} (eigenvectors of the paired moral-neutral content-contrast covariance) and asks
whether the V_moral-restricted refusal transfer fraction R_refusal(k) grows toward the full effect
(rank truncation -> broad-moral) or plateaus at the harm-rank-1 level while R_judgment(k) climbs
(harm-saturating), or both plateau far below 1 (instrument-ceiling). The harm-partialed cell
(V_moral with d_harm removed) isolates whether V_moral's refusal effect IS the harm component.
"""

from __future__ import annotations

import numpy as np


def _unit(v):
    v = np.asarray(v, np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def nested_pca_basis(contrasts: np.ndarray, ks: list[int]) -> dict[int, np.ndarray]:
    """Nested orthonormal bases {k: (hidden, k)} from the per-pair moral-neutral content contrasts
    (rows). SVD is NOT mean-centered, so the top singular vector is the dominant contrast direction
    (~ the V_moral mean-diff) and the rank-k bases nest (rank1 subset rank3 subset ...). k is capped
    at the available rank."""
    X = np.asarray(contrasts, np.float64)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)      # Vt rows = principal contrast directions
    r = Vt.shape[0]
    return {k: Vt[:min(k, r)].T for k in ks}


def subspace_purity(Q: np.ndarray, mean_contrast: np.ndarray) -> float:
    """How much of the mean moral direction the rank-k subspace Q captures: frac(Q, mean_contrast) in
    [0,1]. Rises with k by construction; the curve shows the basis is genuinely moral, not noise."""
    return float(np.linalg.norm(Q.T @ _unit(mean_contrast)))


def cos_harm_components(basis: np.ndarray, harm: np.ndarray, n: int | None = None) -> list[float]:
    """|cos(d_harm, PC_i)| for the first n columns of `basis` -- where the harm direction sits."""
    h = _unit(harm)
    m = basis.shape[1] if n is None else min(n, basis.shape[1])
    return [round(abs(float(_unit(basis[:, i]) @ h)), 4) for i in range(m)]


def harm_partial_basis(V: np.ndarray, harm: np.ndarray) -> np.ndarray:
    """span(V) with d_harm projected out, re-orthonormalized. Restricting the patch to this asks
    whether V_moral moves refusal for any reason OTHER than its harm component. Rank drops by 1 if
    d_harm lies in span(V)."""
    h = _unit(harm)
    V = np.asarray(V, np.float64)
    Vp = V - np.outer(h, h @ V)                           # each column now exactly orthogonal to h
    U, S, _ = np.linalg.svd(Vp, full_matrices=False)      # U columns span col(Vp) subset h^perp
    r = int((S > (S.max() * 1e-6 if S.size else 0)).sum())  # true rank (drops by 1 if h in span V)
    return U[:, :max(r, 1)]


def additivity_ratio(full: list[float], restr: list[float], compl: list[float], rng,
                     b: int = 2000) -> dict:
    """Paired (restricted + complement) / full with a bootstrap CI. ~1.0 = additive (the in- and
    off-V_moral parts sum to the full effect); >1 = overlap/non-additivity (the powered run: ~1.10)."""
    f, r, c = (np.asarray(x, np.float64) for x in (full, restr, compl))
    n = len(f)
    point = float((r + c).sum() / f.sum())
    boots = []
    for _ in range(b):
        i = rng.integers(0, n, n)
        boots.append((r[i] + c[i]).sum() / f[i].sum())
    lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    return {"additivity_ratio": round(point, 4), "ci": [round(lo, 4), round(hi, 4)],
            "additive": bool(lo <= 1.0 <= hi)}


def shape_verdict(R_refusal_k: dict, R_judgment_k: dict, harm_rank1_R: float,
                  ks: list[int], plateau_tol: float = 0.1, ceiling: float = 0.6) -> dict:
    """Frozen Amendment-4 shape verdict from the sweep curves (all publishable).
    harm-saturating: R_refusal plateaus ~ harm-rank-1 while R_judgment climbs above it.
    broad-moral:     R_refusal climbs toward R_judgment (top-k gap within plateau_tol).
    instrument-ceiling: both R_refusal and R_judgment plateau below `ceiling`."""
    kmax = max(ks)
    rr_top, rj_top = R_refusal_k[kmax], R_judgment_k[kmax]
    rr_lo = R_refusal_k[min(ks)]
    refusal_climbs = (rr_top - rr_lo) > plateau_tol
    gap_closes = abs(rr_top - rj_top) <= plateau_tol
    refusal_at_harm = abs(rr_top - harm_rank1_R) <= plateau_tol
    if rr_top < ceiling and rj_top < ceiling:
        v = "instrument_ceiling"
    elif gap_closes and refusal_climbs:
        v = "broad_moral"
    elif refusal_at_harm and rj_top > rr_top + plateau_tol:
        v = "harm_saturating"
    else:
        v = "indeterminate"
    return {"verdict": v, "R_refusal_top": round(rr_top, 4), "R_judgment_top": round(rj_top, 4),
            "harm_rank1_R": round(harm_rank1_R, 4), "refusal_climbs": bool(refusal_climbs),
            "gap_closes": bool(gap_closes)}
