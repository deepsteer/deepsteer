"""Orchestrator: build_probing_dataset() ties the pipeline stages together."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from deepsteer.core.model_interface import ModelInterface
from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.balancing import (
    balance_by_foundation,
    report_distribution,
    train_test_split,
)
from deepsteer.datasets.moral_seeds import get_moral_seeds
from deepsteer.datasets.neutral_pool import get_flat_neutral_pool
from deepsteer.datasets.pairing import pair_by_word_count, pair_minimal
from deepsteer.datasets.types import (
    DatasetMetadata,
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)
from deepsteer.datasets.validation import validate_pairs

logger = logging.getLogger(__name__)

V2_DATASET_PATH = Path(__file__).resolve().parent / "moral_probing_v2.json"
_FOUNDATION_MAP = {f.value: f for f in MoralFoundation}


def _load_v2_dataset(target_per_foundation: int = 200) -> ProbingDataset:
    """Load the pre-assembled moral_probing_v2.json dataset."""
    with open(V2_DATASET_PATH) as f:
        data = json.load(f)

    train, test = [], []
    for p in data["pairs"]:
        pair = ProbingPair(
            moral=p["moral"],
            neutral=p["neutral"],
            foundation=_FOUNDATION_MAP[p["foundation"]],
            neutral_domain=NeutralDomain.MATCHED,
            generation_method=GenerationMethod.LLM,
            moral_word_count=len(p["moral"].split()),
            neutral_word_count=len(p["neutral"].split()),
            provenance=p.get("id", ""),
        )
        if p["split"] == "test":
            test.append(pair)
        else:
            train.append(pair)

    if target_per_foundation < 200:
        import random
        rng = random.Random(42)
        from collections import defaultdict
        by_fnd: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
        for p in train:
            by_fnd[p.foundation].append(p)
        train = []
        n_train = int(target_per_foundation * 0.8)
        for fnd, pairs in by_fnd.items():
            rng.shuffle(pairs)
            train.extend(pairs[:n_train])
        by_fnd_test: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
        for p in test:
            by_fnd_test[p.foundation].append(p)
        test_out: list[ProbingPair] = []
        n_test = target_per_foundation - n_train
        for fnd, pairs in by_fnd_test.items():
            rng.shuffle(pairs)
            test_out.extend(pairs[:n_test])
        test = test_out

    foundations = {}
    for p in train + test:
        foundations[p.foundation.value] = foundations.get(p.foundation.value, 0) + 1

    metadata = DatasetMetadata(
        version="2.0.0",
        generation_method="v2_assembled",
        total_pairs=len(train) + len(test),
        train_pairs=len(train),
        test_pairs=len(test),
        foundations=foundations,
    )
    return ProbingDataset(train=train, test=test, metadata=metadata)


def build_probing_dataset(
    model: ModelInterface | None = None,
    *,
    target_per_foundation: int = 40,
    test_fraction: float = 0.2,
    max_length_ratio: float = 1.5,
    seed: int = 42,
    legacy_pool: bool = False,
    use_v2: bool = True,
) -> ProbingDataset:
    """Build a validated, balanced probing dataset.

    Args:
        model: Optional API model for LLM-based neutral generation.  When
            ``None``, uses pre-written minimal pairs (no API needed).
        target_per_foundation: Target pairs per moral foundation.
        test_fraction: Fraction of pairs held out for testing.
        max_length_ratio: Maximum word count ratio between paired sentences.
        seed: Random seed for reproducibility.
        legacy_pool: If ``True`` and *model* is ``None``, fall back to the old
            pool-based word-count matching instead of minimal pairs.
        use_v2: If ``True`` (default), load the pre-assembled v2 dataset
            (1,200 pairs). Falls back to v1 if v2 file is missing.

    Returns:
        A complete ProbingDataset with train/test split and metadata.
    """
    if use_v2 and V2_DATASET_PATH.exists():
        logger.info("Loading v2 dataset from %s", V2_DATASET_PATH)
        return _load_v2_dataset(target_per_foundation=target_per_foundation)

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
