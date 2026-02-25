"""Foundation balancing and stratified train/test splitting."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.types import ProbingPair

logger = logging.getLogger(__name__)


def balance_by_foundation(
    pairs: list[ProbingPair],
    *,
    target_per_foundation: int = 40,
    seed: int = 42,
) -> list[ProbingPair]:
    """Downsample over-represented foundations to *target_per_foundation*.

    Under-represented foundations are kept as-is with a warning.

    Args:
        pairs: Input pairs (may be unbalanced across foundations).
        target_per_foundation: Maximum pairs per foundation.
        seed: Random seed for deterministic downsampling.

    Returns:
        Balanced list of pairs.
    """
    rng = random.Random(seed)
    by_foundation: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
    for pair in pairs:
        by_foundation[pair.foundation].append(pair)

    balanced: list[ProbingPair] = []
    for foundation in MoralFoundation:
        pool = by_foundation[foundation]
        if len(pool) < target_per_foundation:
            logger.warning(
                "%s: only %d pairs (target %d)",
                foundation.value,
                len(pool),
                target_per_foundation,
            )
            balanced.extend(pool)
        else:
            balanced.extend(rng.sample(pool, target_per_foundation))

    logger.info(
        "Balancing: %d → %d pairs (target %d/foundation)",
        len(pairs),
        len(balanced),
        target_per_foundation,
    )
    return balanced


def train_test_split(
    pairs: list[ProbingPair],
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
    stratify: bool = True,
) -> tuple[list[ProbingPair], list[ProbingPair]]:
    """Split pairs into train and test sets.

    Args:
        pairs: Input pairs.
        test_fraction: Fraction of pairs for the test set.
        seed: Random seed for reproducibility.
        stratify: If True, stratify split by foundation so each foundation
            has proportional representation in both sets.

    Returns:
        (train, test) tuple.
    """
    rng = random.Random(seed)

    if not stratify:
        shuffled = list(pairs)
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * test_fraction))
        return shuffled[split:], shuffled[:split]

    # Stratified split: split each foundation independently
    by_foundation: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
    for pair in pairs:
        by_foundation[pair.foundation].append(pair)

    train: list[ProbingPair] = []
    test: list[ProbingPair] = []

    for foundation in MoralFoundation:
        pool = by_foundation[foundation]
        rng.shuffle(pool)
        n_test = max(1, int(len(pool) * test_fraction)) if pool else 0
        test.extend(pool[:n_test])
        train.extend(pool[n_test:])

    return train, test


def report_distribution(pairs: list[ProbingPair]) -> dict[str, int]:
    """Count pairs per foundation.

    Args:
        pairs: Probing pairs to count.

    Returns:
        Mapping from foundation value string to count.
    """
    counts: dict[str, int] = {}
    for foundation in MoralFoundation:
        counts[foundation.value] = 0
    for pair in pairs:
        counts[pair.foundation.value] += 1
    return counts
