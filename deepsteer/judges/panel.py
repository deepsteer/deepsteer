"""A cross-family judge panel with its calibration built in (CST_PLAN §3.4, §6.5).

The panel scores categorical labels (an applied tenet id, ``none``, a pair verdict). The model
calls live in the caller; this module takes labels as data and computes what the plan requires
before a panel's labels are used: test-retest kappa per family and task, cross-family agreement,
accuracy against human labels where Layer A overlaps, the pooled label, and the disagreement rule
that admits an item to ``T_conf``.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from fractions import Fraction
from itertools import combinations

from deepsteer.stats.agreement import agreement, cohen_kappa

KAPPA_BAR = 0.8  # S0's pilot gate (CST_PLAN §8): tenet application, per family


@dataclasses.dataclass(frozen=True)
class PanelMember:
    """One judge family at a pinned revision."""

    family: str
    model_id: str
    revision: str
    access: str  # self_hosted | api | cli


def retest_agreement(first: list[str | None], second: list[str | None]) -> dict:
    """Agreement and kappa between two passes of one family over the same items (temperature and
    seed as registered); the plan's bar is ``KAPPA_BAR``."""
    k = cohen_kappa(first, second)
    return {"n": len(first), "agreement": agreement(first, second), "kappa": k,
            "passes": k >= KAPPA_BAR}


def cross_family(labels: dict[str, list[str | None]]) -> dict:
    """Pairwise kappa between families on the same items, and the mean."""
    pairs = {f"{a}|{b}": cohen_kappa(labels[a], labels[b])
             for a, b in combinations(sorted(labels), 2)}
    return {"pairwise_kappa": pairs,
            "mean_kappa": sum(pairs.values()) / len(pairs) if pairs else None}


def pooled_label(votes: list[str | None]) -> tuple[str | None, float]:
    """Majority label and its vote share; ties return ``None`` (no pooled label)."""
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None, c[0][1] / len(votes)
    return c[0][0], c[0][1] / len(votes)


def disagreement(votes: list[str | None]) -> float:
    """1 − the majority vote share: 0 when the families agree, 2/3 for a three-way split.

    Computed as an exact fraction then converted, so equal splits compare equal (``1 - 2/3`` in
    floats exceeds ``1/3`` by one ulp, which would admit every 2-1 split to ``T_conf``)."""
    top = Counter(votes).most_common(1)[0][1]
    return float(Fraction(len(votes) - top, len(votes)))


def accuracy_vs_human(panel: list[str | None], human: list[str | None]) -> float:
    """Share of Layer A overlap items where the label equals the human label."""
    return agreement(panel, human)


def conflict_threshold(calibration_disagreements: list[float], quantile: float = 0.9) -> float:
    """The ``T_conf`` threshold fixed at G0 from the calibration spread: the ``quantile`` of panel
    disagreement on calibration items whose human label is unanimous. An evaluation item whose
    disagreement exceeds it is contested beyond what the panel shows on settled items."""
    xs = sorted(calibration_disagreements)
    if not xs:
        raise ValueError("no calibration items")
    i = min(len(xs) - 1, max(0, int(round(quantile * (len(xs) - 1)))))
    return xs[i]


def is_conflict(votes: list[str | None], threshold: float) -> bool:
    """An item is a panel-detected conflict when its disagreement exceeds the threshold."""
    return disagreement(votes) > threshold


def _binom_quantile(n: int, p: float, q: float) -> int:
    """Smallest k with P(Binomial(n, p) <= k) >= q."""
    from math import comb

    acc = 0.0
    for k in range(n + 1):
        acc += comb(n, k) * p ** k * (1 - p) ** (n - k)
        if acc >= q:
            return k
    return n


def root_rate_tags(
    applications: dict[str, list[list[str | None]]], noise_rate: float, quantile: float = 0.9
) -> dict[str, dict]:
    """The root-rate rule for panel-detected ``T_conf`` (CST_PLAN Amendment 1, D7).

    ``applications`` maps a root tenet to the panel's votes on each of its applications (one vote
    list per scenario). A root is tagged when its count of non-unanimous applications exceeds the
    ``quantile`` of Binomial(m, ``noise_rate``), where ``noise_rate`` is the non-unanimity rate the
    panel shows on settled calibration items. Under noise alone about ``1 - quantile`` of roots are
    tagged, however many applications each root has; the any-dissent rule would tag most roots.
    """
    out = {}
    for root, apps in applications.items():
        m = len(apps)
        k = sum(disagreement(v) > 0 for v in apps)
        bar = _binom_quantile(m, noise_rate, quantile)
        out[root] = {"m": m, "dissent": k, "bar": bar, "tagged": k > bar}
    return out
