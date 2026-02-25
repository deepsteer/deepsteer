"""Data mixing: combine moral and general corpus content at specified ratios.

Provides a :class:`DataMixer` that takes moral and general text corpora and
produces mixed batches at a target moral ratio, optionally weighted by
Moral Foundation.  Works with :class:`CurriculumSchedule` to vary the ratio
over training.

DeepSteer does NOT execute training.  ``DataMixer`` produces sample lists
and statistics that a researcher integrates into their data pipeline.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from deepsteer.core.types import (
    CurriculumSchedule,
    MixedSample,
    MixingResult,
    MoralFoundation,
)

logger = logging.getLogger(__name__)


class DataMixer:
    """Mix moral and general corpus content at a target ratio.

    The mixer samples from a moral corpus (tagged by foundation) and a
    general corpus to produce mixed batches.  It does not modify the texts —
    only decides *which* texts to include.

    Args:
        moral_corpus: Mapping from foundation value to list of texts.
        general_corpus: List of general (non-moral) training texts.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        moral_corpus: dict[str, list[str]],
        general_corpus: list[str],
        *,
        seed: int = 42,
    ) -> None:
        self._moral_corpus = moral_corpus
        self._general_corpus = general_corpus
        self._rng = random.Random(seed)

        self._moral_flat: list[tuple[str, str]] = []
        for foundation, texts in moral_corpus.items():
            for text in texts:
                self._moral_flat.append((foundation, text))

        total_moral = len(self._moral_flat)
        total_general = len(general_corpus)
        logger.info(
            "DataMixer: %d moral texts (%d foundations), %d general texts",
            total_moral, len(moral_corpus), total_general,
        )

    @property
    def n_moral(self) -> int:
        """Total number of moral texts available."""
        return len(self._moral_flat)

    @property
    def n_general(self) -> int:
        """Total number of general texts available."""
        return len(self._general_corpus)

    def mix_batch(
        self,
        batch_size: int,
        moral_ratio: float,
        *,
        foundation_weights: dict[str, float] | None = None,
    ) -> tuple[list[MixedSample], MixingResult]:
        """Produce a single mixed batch.

        Args:
            batch_size: Total number of samples in the batch.
            moral_ratio: Fraction of the batch that should be moral content (0-1).
            foundation_weights: Optional per-foundation sampling weights.
                If ``None``, samples uniformly from all moral texts.

        Returns:
            A tuple of (samples, result) where samples is the mixed list and
            result contains statistics about what was mixed.
        """
        n_moral = round(batch_size * moral_ratio)
        n_general = batch_size - n_moral

        samples: list[MixedSample] = []
        foundation_counts: dict[str, int] = defaultdict(int)

        # Sample moral texts
        if n_moral > 0 and self._moral_flat:
            moral_samples = self._sample_moral(n_moral, foundation_weights)
            for foundation, text in moral_samples:
                samples.append(MixedSample(
                    text=text,
                    is_moral=True,
                    foundation=MoralFoundation(foundation),
                    source="moral_corpus",
                ))
                foundation_counts[foundation] += 1

        # Sample general texts
        if n_general > 0 and self._general_corpus:
            general_samples = self._rng.choices(self._general_corpus, k=n_general)
            for text in general_samples:
                samples.append(MixedSample(
                    text=text,
                    is_moral=False,
                    foundation=None,
                    source="general_corpus",
                ))

        self._rng.shuffle(samples)

        actual_moral = sum(1 for s in samples if s.is_moral)
        result = MixingResult(
            total_samples=len(samples),
            moral_samples=actual_moral,
            general_samples=len(samples) - actual_moral,
            moral_ratio=actual_moral / len(samples) if samples else 0.0,
            foundation_counts=dict(foundation_counts),
            metadata={
                "requested_ratio": moral_ratio,
                "batch_size": batch_size,
            },
        )
        return samples, result

    def mix_from_schedule(
        self,
        schedule: CurriculumSchedule,
        batch_size: int,
        *,
        steps: list[int] | None = None,
    ) -> list[tuple[int, list[MixedSample], MixingResult]]:
        """Produce mixed batches for each step in a curriculum schedule.

        Args:
            schedule: A CurriculumSchedule that defines per-step moral ratios.
            batch_size: Number of samples per batch.
            steps: Specific steps to generate batches for.  If ``None``,
                generates one batch at the midpoint of each phase.

        Returns:
            List of (step, samples, result) tuples.
        """
        if steps is None:
            steps = [
                (phase.start_step + phase.end_step) // 2
                for phase in schedule.phases
            ]

        batches: list[tuple[int, list[MixedSample], MixingResult]] = []
        for step in steps:
            ratio = schedule.ratio_at_step(step)
            # Find the active phase to get foundation weights
            weights = None
            for phase in schedule.phases:
                if phase.start_step <= step < phase.end_step:
                    weights = phase.foundation_weights or None
                    break
            samples, result = self.mix_batch(
                batch_size, ratio, foundation_weights=weights,
            )
            result.metadata["step"] = step
            batches.append((step, samples, result))

        logger.info(
            "Mixed %d batches from schedule (%s)",
            len(batches), schedule.method,
        )
        return batches

    def _sample_moral(
        self,
        n: int,
        foundation_weights: dict[str, float] | None,
    ) -> list[tuple[str, str]]:
        """Sample *n* moral texts, optionally weighted by foundation."""
        if foundation_weights is None:
            return self._rng.choices(self._moral_flat, k=n)

        # Weighted sampling: assign each moral text a weight based on its foundation
        weights = [
            foundation_weights.get(foundation, 0.0)
            for foundation, _ in self._moral_flat
        ]
        total_weight = sum(weights)
        if total_weight == 0:
            return self._rng.choices(self._moral_flat, k=n)

        return self._rng.choices(self._moral_flat, weights=weights, k=n)
