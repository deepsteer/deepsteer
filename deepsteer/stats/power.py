"""Closed-form power for rate differences (lifted from deepsteer.kdg.stats)."""

from __future__ import annotations


def mde_rate_difference(
    n_a: int, n_b: int, p: float = 0.5, *, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Smallest two-proportion difference detectable at ``power`` (normal approx, worst-case p)."""
    from math import sqrt

    z_a, z_b = 1.959964, {0.8: 0.841621, 0.9: 1.281552}[power]
    se = sqrt(p * (1 - p) * (1 / n_a + 1 / n_b))
    return (z_a + z_b) * se
