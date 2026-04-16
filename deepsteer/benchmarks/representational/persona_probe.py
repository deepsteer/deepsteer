"""Persona-feature probing: linear probe for the toxic-persona direction.

Parallel in structure to :class:`LayerWiseMoralProbe`, but trained on
``(persona_voice, neutral_voice)`` minimal pairs from
:mod:`deepsteer.datasets.persona_pairs`.

This is a linear-probe analog of the SAE-identified toxic-persona latent
reported by Wang et al. (2025), "Persona Features Control Emergent
Misalignment" (arXiv:2506.19823).  Because we do not have a pre-trained SAE
for OLMo base models, the probe recovers a functionally analogous direction
from minimal-pair contrasts.  See Phase D (H13) in ``RESEARCH_PLAN.md``.

Reuses :class:`GeneralLinearProbe` internally so that methodology (BCE loss,
Adam, 50 epochs, fp32 probes) matches the moral, sentiment, and syntax
probes exactly — the only difference is the input dataset.
"""

from __future__ import annotations

import logging

from deepsteer.benchmarks.representational.general_probe import (
    GeneralLinearProbe,
)
from deepsteer.core.benchmark_suite import Benchmark
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import (
    AccessTier,
    BenchmarkResult,
    LayerProbingResult,
)
from deepsteer.datasets.persona_pairs import get_persona_dataset

logger = logging.getLogger(__name__)

# Matches LayerWiseMoralProbe default.  See C1 methodology note: this
# threshold is too lenient once models are trained — raise to 0.80 if
# fragility metrics are needed.
_ONSET_THRESHOLD = 0.6


class PersonaFeatureProbe(Benchmark):
    """Layer-wise linear probe for the toxic-persona direction.

    For each transformer layer, trains a :class:`torch.nn.Linear` classifier
    on mean-pooled hidden states of persona-voice vs. neutral-voice
    minimal pairs.  Produces a :class:`LayerProbingResult` with the same
    onset/peak/depth/breadth metrics as the moral probe, enabling direct
    comparison against Phase B/C results.

    Args:
        train_pairs: Optional pre-split training pairs.  If ``None``, uses
            the default stratified 80/20 split of the persona dataset.
        test_pairs: Optional pre-split test pairs.
        n_epochs: Training epochs per layer (default matches moral probe).
        lr: Adam learning rate.
        onset_threshold: Accuracy threshold for counting a layer as having
            onset persona encoding.
        split_seed: Seed for the default train/test split (ignored if
            explicit pairs are provided).
    """

    def __init__(
        self,
        train_pairs: list[tuple[str, str]] | None = None,
        test_pairs: list[tuple[str, str]] | None = None,
        *,
        n_epochs: int = 50,
        lr: float = 1e-2,
        onset_threshold: float = _ONSET_THRESHOLD,
        split_seed: int = 42,
    ) -> None:
        if (train_pairs is None) != (test_pairs is None):
            raise ValueError(
                "train_pairs and test_pairs must both be provided or both be None"
            )
        self._explicit_train = train_pairs
        self._explicit_test = test_pairs
        self._n_epochs = n_epochs
        self._lr = lr
        self._onset_threshold = onset_threshold
        self._split_seed = split_seed
        self._probe = GeneralLinearProbe(
            probe_name=self.name,
            n_epochs=n_epochs,
            lr=lr,
            onset_threshold=onset_threshold,
        )

    @property
    def name(self) -> str:
        return "persona_feature_probe"

    @property
    def min_access_tier(self) -> AccessTier:
        return AccessTier.WEIGHTS

    def run(self, model: WhiteBoxModel) -> BenchmarkResult:  # type: ignore[override]
        """Run the persona probe across all layers of *model*.

        Args:
            model: A :class:`WhiteBoxModel` with activation capture support.

        Returns:
            :class:`LayerProbingResult` with per-layer accuracy and summary
            metrics.  ``benchmark_name`` will be ``"persona_feature_probe"``.
        """
        if self._explicit_train is not None and self._explicit_test is not None:
            train_pairs = self._explicit_train
            test_pairs = self._explicit_test
        else:
            train_pairs, test_pairs = get_persona_dataset(
                test_fraction=0.2,
                seed=self._split_seed,
                stratified=True,
            )

        logger.info(
            "Running persona-feature probe: %d train pairs, %d test pairs",
            len(train_pairs), len(test_pairs),
        )
        result = self._probe.run(model, train_pairs, test_pairs)
        assert isinstance(result, LayerProbingResult)
        return result
