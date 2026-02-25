"""Sentence pairing strategies (pool-based and minimal-pair)."""

from __future__ import annotations

import logging
import random

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.types import GenerationMethod, NeutralDomain, ProbingPair

logger = logging.getLogger(__name__)


def pair_by_word_count(
    moral_seeds: dict[MoralFoundation, list[str]],
    neutral_pool: list[tuple[str, NeutralDomain]],
    *,
    max_length_ratio: float = 1.3,
    seed: int = 42,
) -> list[ProbingPair]:
    """Pair each moral seed with the closest-length unused neutral sentence.

    Greedy matching: iterate moral seeds in shuffled order, for each find the
    unused neutral sentence whose word count is closest.  Pairs where the
    length ratio exceeds *max_length_ratio* are still emitted (downstream
    validation will filter them).

    Args:
        moral_seeds: Mapping from foundation to moral sentences.
        neutral_pool: Flat list of (sentence, domain) tuples.
        max_length_ratio: Soft preference — neutrals outside this ratio are
            deprioritized but not excluded (validation handles hard filtering).
        seed: Random seed for deterministic shuffling.

    Returns:
        List of ProbingPair, one per moral seed (or fewer if pool is exhausted).
    """
    rng = random.Random(seed)

    # Build shuffled list of all moral seeds with their foundation
    all_morals: list[tuple[str, MoralFoundation]] = []
    for foundation, sentences in moral_seeds.items():
        for sent in sentences:
            all_morals.append((sent, foundation))
    rng.shuffle(all_morals)

    # Index neutral pool by word count for faster lookup
    available: list[tuple[str, NeutralDomain, int]] = [
        (sent, domain, len(sent.split()))
        for sent, domain in neutral_pool
    ]
    rng.shuffle(available)
    used: set[int] = set()

    pairs: list[ProbingPair] = []

    for moral_sent, foundation in all_morals:
        moral_wc = len(moral_sent.split())
        best_idx = -1
        best_diff = float("inf")

        for i, (_, _, neutral_wc) in enumerate(available):
            if i in used:
                continue
            diff = abs(moral_wc - neutral_wc)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx == -1:
            logger.warning("Neutral pool exhausted; %d moral seeds unmatched", 1)
            continue

        neutral_sent, neutral_domain, neutral_wc = available[best_idx]
        used.add(best_idx)

        pairs.append(ProbingPair(
            moral=moral_sent,
            neutral=neutral_sent,
            foundation=foundation,
            neutral_domain=neutral_domain,
            generation_method=GenerationMethod.POOL,
            moral_word_count=moral_wc,
            neutral_word_count=neutral_wc,
            provenance=f"pool_match(seed={seed})",
        ))

    logger.info("Pool pairing: %d pairs from %d moral seeds", len(pairs), len(all_morals))
    return pairs


def pair_minimal(
    minimal_pairs: dict[MoralFoundation, list[tuple[str, str]]],
    *,
    seed: int = 42,
) -> list[ProbingPair]:
    """Create ProbingPair instances from pre-defined minimal pairs.

    Each ``(moral, neutral)`` tuple is wrapped into a :class:`ProbingPair`
    with ``generation_method=POOL`` and ``neutral_domain=MATCHED``.

    Args:
        minimal_pairs: Mapping from foundation to lists of (moral, neutral)
            sentence tuples.
        seed: Random seed for deterministic shuffling.

    Returns:
        Shuffled list of ProbingPair instances.
    """
    rng = random.Random(seed)
    pairs: list[ProbingPair] = []

    for foundation, tuples in minimal_pairs.items():
        for moral_sent, neutral_sent in tuples:
            pairs.append(ProbingPair(
                moral=moral_sent,
                neutral=neutral_sent,
                foundation=foundation,
                neutral_domain=NeutralDomain.MATCHED,
                generation_method=GenerationMethod.POOL,
                moral_word_count=len(moral_sent.split()),
                neutral_word_count=len(neutral_sent.split()),
                provenance=f"minimal_pair(seed={seed})",
            ))

    rng.shuffle(pairs)
    logger.info("Minimal pairing: %d pairs from %d foundations", len(pairs), len(minimal_pairs))
    return pairs
