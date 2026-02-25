"""Orchestrator: build_probing_dataset() ties the pipeline stages together."""

from __future__ import annotations

import logging

from deepsteer.core.model_interface import ModelInterface
from deepsteer.datasets.balancing import (
    balance_by_foundation,
    report_distribution,
    train_test_split,
)
from deepsteer.datasets.moral_seeds import get_moral_seeds
from deepsteer.datasets.neutral_pool import get_flat_neutral_pool
from deepsteer.datasets.pairing import pair_by_word_count, pair_minimal
from deepsteer.datasets.types import DatasetMetadata, ProbingDataset
from deepsteer.datasets.validation import validate_pairs

logger = logging.getLogger(__name__)


def build_probing_dataset(
    model: ModelInterface | None = None,
    *,
    target_per_foundation: int = 40,
    test_fraction: float = 0.2,
    max_length_ratio: float = 1.5,
    seed: int = 42,
    legacy_pool: bool = False,
) -> ProbingDataset:
    """Build a validated, balanced probing dataset.

    Pipeline stages:
        1. Load moral seeds (and minimal pairs or neutral pool)
        2. Pair generation — minimal pairs (default), LLM-based, or legacy pool
        3. Validate — length ratio, word overlap, keyword scan, dedup
        4. Balance — downsample to *target_per_foundation* per foundation
        5. Package — stratified train/test split + metadata

    Args:
        model: Optional API model for LLM-based neutral generation.  When
            ``None``, uses pre-written minimal pairs (no API needed).
        target_per_foundation: Target pairs per moral foundation.
        test_fraction: Fraction of pairs held out for testing.
        max_length_ratio: Maximum word count ratio between paired sentences.
        seed: Random seed for reproducibility.
        legacy_pool: If ``True`` and *model* is ``None``, fall back to the old
            pool-based word-count matching instead of minimal pairs.

    Returns:
        A complete ProbingDataset with train/test split and metadata.
    """
    # Stage 1: Load seeds
    moral_seeds = get_moral_seeds()
    logger.info("Stage 1: Loaded %d moral seeds across %d foundations",
                sum(len(v) for v in moral_seeds.values()), len(moral_seeds))

    # Stage 2: Pair generation
    min_word_overlap = 0.0
    if model is not None:
        from deepsteer.datasets.llm_generation import generate_neutral_with_llm
        pairs = generate_neutral_with_llm(moral_seeds, model)
        method = "llm"
        min_word_overlap = 0.15
    elif legacy_pool:
        neutral_pool = get_flat_neutral_pool()
        pairs = pair_by_word_count(
            moral_seeds, neutral_pool, max_length_ratio=max_length_ratio, seed=seed,
        )
        method = "pool"
    else:
        from deepsteer.datasets.minimal_pairs import get_minimal_pairs
        mp = get_minimal_pairs()
        pairs = pair_minimal(mp, seed=seed)
        method = "minimal_pair"
        min_word_overlap = 0.15
    logger.info("Stage 2: Generated %d candidate pairs (method=%s)", len(pairs), method)

    # Stage 3: Validate
    valid_pairs, val_stats = validate_pairs(
        pairs, max_length_ratio=max_length_ratio, min_word_overlap=min_word_overlap,
    )
    logger.info("Stage 3: %d pairs passed validation", len(valid_pairs))

    # Stage 4: Balance
    balanced = balance_by_foundation(
        valid_pairs, target_per_foundation=target_per_foundation, seed=seed,
    )
    logger.info("Stage 4: %d pairs after balancing", len(balanced))

    # Stage 5: Package
    train, test = train_test_split(
        balanced, test_fraction=test_fraction, seed=seed, stratify=True,
    )
    foundations = report_distribution(balanced)

    metadata = DatasetMetadata(
        version="1.0.0",
        generation_method=method,
        total_pairs=len(balanced),
        train_pairs=len(train),
        test_pairs=len(test),
        foundations=foundations,
        validation_stats=val_stats.to_dict(),
    )

    logger.info(
        "Stage 5: Dataset ready — %d train, %d test, %d total",
        len(train), len(test), len(balanced),
    )
    return ProbingDataset(train=train, test=test, metadata=metadata)
