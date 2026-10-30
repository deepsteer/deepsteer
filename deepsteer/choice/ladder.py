"""Degenerate-policy ladder and worst-case power for items scored by a per-label key.

Any item with a ``key: dict[str, float]`` (label -> score) works, e.g.
:class:`deepsteer.datasets.moralbench.MoralBenchItem`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

Z_975 = 1.959964
Z_POWER = Z_975 + 0.841621  # two-sided alpha 0.05, power 0.80


def ladder(items: Sequence[Any]) -> dict[str, float | int]:
    """Degenerate-policy scores of a group of items.

    ``worst`` and ``oracle`` take the lowest / highest key per item; ``always_<L>`` answers label
    L on every item; ``uniform`` is the expectation of a uniform choice over each item's labels.
    """
    labels = sorted({c for it in items for c in it.key})
    out: dict[str, float | int] = {"n": len(items)}
    out["worst"] = sum(min(it.key.values()) for it in items)
    for c in labels:
        out[f"always_{c}"] = sum(it.key.get(c, 0.0) for it in items)
    out["uniform"] = sum(sum(it.key.values()) / len(it.key) for it in items)
    out["oracle"] = sum(max(it.key.values()) for it in items)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def _worst_var(items: Sequence[Any]) -> float:
    return sum((max(it.key.values()) - min(it.key.values())) ** 2 / 4 for it in items)


def worst_case_mde(items: Sequence[Any], n_a: int, n_b: int) -> float:
    """MDE (alpha 0.05 two-sided, power 0.8) for a difference of two models' mean scores over
    ``n_a`` and ``n_b`` rollouts on these fixed items, at choice probability 0.5 on every item."""
    var = _worst_var(items)
    return Z_POWER * math.sqrt(var / n_a + var / n_b)


def match_half_width(
    items: Sequence[Any], n_paper: int = 5, n_ours: int = 32, rounding: float = 0.05
) -> float:
    """Worst-case 95% half-width for a paper cell (mean of ``n_paper`` repeats) against our mean
    of ``n_ours`` rollouts on the same items, plus the paper's rounding."""
    var = _worst_var(items)
    return Z_975 * math.sqrt(var / n_paper + var / n_ours) + rounding
