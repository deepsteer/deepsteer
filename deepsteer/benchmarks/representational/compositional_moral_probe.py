"""Compositional moral probe (Phase C4).

Identical methodology to :class:`LayerWiseMoralProbe`, but uses the
compositional minimal-pair dataset from
:mod:`deepsteer.datasets.compositional_moral_pairs` whose pairs share most of
their tokens and differ in 1-2 individually mild words. A linear probe at
each layer that separates the compositional pairs must be reading multi-word
moral integration rather than single-word moralized vocabulary.

Why this exists
---------------

Phase C2 found that the standard moral probe crosses 70 % accuracy at OLMo-2
1B step 1K — earlier than sentiment (step 2K) and well before syntax (step
6K). The standard probe's pairs swap a morally-loaded lexeme for a neutral
one ("She murdered the woman" vs. "She greeted the woman"), so the early
moral onset may simply be measuring the emergence of statistically separable
moralized vocabulary, not compositional moral encoding.

This probe is the lexical-accessibility ablation. The compositional dataset's
contrast tokens ("protect" vs. "humiliate", "hungry" vs. "wealthy", "safe"
vs. "hidden", "innocent" vs. "guilty") are individually mild — none appear in
the standard moral keyword list — and only flip the pair's moral status in
the context of the surrounding action. A probe that separates these pairs is
representing moral valence as a compositional property of multi-word
combinations rather than as a property of individual lexical tokens.

See ``RESEARCH_PLAN.md`` H21 and the
``papers/accuracy_vs_fragility/outputs/phase_c4_compositional/``
directory for the trajectory results.
"""

from __future__ import annotations

import logging

from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.compositional_moral_pairs import (
    get_compositional_moral_dataset,
)
from deepsteer.datasets.types import (
    DatasetMetadata,
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)

logger = logging.getLogger(__name__)


def _build_compositional_probing_dataset(
    *,
    test_fraction: float = 0.20,
    seed: int = 42,
) -> ProbingDataset:
    """Wrap the compositional minimal-pair dataset as a ``ProbingDataset``.

    The standard ``ProbingPair`` contract uses ``moral`` (class 1) and
    ``neutral`` (class 0) sides. For Phase C4 the class-0 side is
    *immoral* rather than neutral — both halves describe an action, only
    the moral valence differs. The downstream linear-probe trainer is
    foundation-blind, so the pair shape carries through unchanged.

    Args:
        test_fraction: Fraction held out for the test split (default 0.20
            → 160 train / 40 test).
        seed: RNG seed for the stratified shuffle.

    Returns:
        A ``ProbingDataset`` with a ``train`` and ``test`` list of
        ``ProbingPair`` objects.
    """
    train_pairs, test_pairs = get_compositional_moral_dataset(
        test_fraction=test_fraction, seed=seed,
    )

    def _wrap(pairs: list[tuple[str, str]]) -> list[ProbingPair]:
        out: list[ProbingPair] = []
        for moral, immoral in pairs:
            out.append(
                ProbingPair(
                    moral=moral,
                    neutral=immoral,
                    foundation=MoralFoundation.CARE_HARM,
                    neutral_domain=NeutralDomain.MATCHED,
                    generation_method=GenerationMethod.HANDWRITTEN,
                    moral_word_count=len(moral.split()),
                    neutral_word_count=len(immoral.split()),
                    provenance="compositional_moral_pairs",
                )
            )
        return out

    train = _wrap(train_pairs)
    test = _wrap(test_pairs)

    metadata = DatasetMetadata(
        version="1.0.0",
        generation_method="handwritten_compositional",
        total_pairs=len(train) + len(test),
        train_pairs=len(train),
        test_pairs=len(test),
        foundations={"compositional": len(train) + len(test)},
        validation_stats={"all_gates_pass": 1},
    )

    logger.info(
        "Compositional probing dataset: %d train, %d test pairs",
        len(train), len(test),
    )
    return ProbingDataset(train=train, test=test, metadata=metadata)


class CompositionalMoralProbe(LayerWiseMoralProbe):
    """Layer-wise moral probe over compositional minimal pairs.

    Inherits the trainer, activation collection, and metric computation
    from :class:`LayerWiseMoralProbe` unchanged; only the default dataset
    swaps to :func:`_build_compositional_probing_dataset`. The benchmark
    name is overridden so suite output JSON files are written to a
    distinct path.
    """

    def __init__(
        self,
        dataset: ProbingDataset | None = None,
        *,
        n_epochs: int = 50,
        lr: float = 1e-2,
        onset_threshold: float = 0.6,
        test_fraction: float = 0.20,
        seed: int = 42,
    ) -> None:
        super().__init__(
            dataset=dataset or _build_compositional_probing_dataset(
                test_fraction=test_fraction, seed=seed,
            ),
            n_epochs=n_epochs,
            lr=lr,
            onset_threshold=onset_threshold,
        )

    @property
    def name(self) -> str:
        return "compositional_moral_probe"
