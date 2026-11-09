"""Inter-rater agreement for categorical labels (lifted from deepsteer.kdg.calibration)."""

from __future__ import annotations

import math


def agreement(a: list[str | None], b: list[str | None]) -> float:
    """Fraction of positions where the two label lists agree (``None`` is a label)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list[str | None], b: list[str | None]) -> float:
    """Cohen's kappa over categorical labels (``None`` is its own category)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    n = len(a)
    po = agreement(a, b)
    cats = set(a) | set(b)
    # fsum: the correctly rounded sum, independent of the set's hash-seeded iteration order (a
    # plain sum flipped the last bit of kappa on 35 of 121 PYTHONHASHSEEDs under Python 3.10)
    pe = math.fsum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)
