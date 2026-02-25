"""Foundation-specific probing: separate linear probes per Moral Foundation.

Instead of a single binary moral/neutral classifier, trains one probe per
MFT foundation.  Reveals whether different moral foundations are encoded at
different layers — e.g. care/harm might emerge earlier than loyalty/betrayal.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.core.benchmark_suite import Benchmark
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import (
    AccessTier,
    BenchmarkResult,
    FoundationLayerProbeScore,
    FoundationProbingResult,
    MoralFoundation,
)
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.types import ProbingDataset, ProbingPair

logger = logging.getLogger(__name__)

_ONSET_THRESHOLD = 0.6


class FoundationSpecificProbe(Benchmark):
    """Train separate binary probes for each Moral Foundation at each layer.

    For each foundation, filters the probing dataset to pairs tagged with that
    foundation, then trains a linear probe at every layer.  Produces per-foundation
    onset/peak/depth/breadth metrics that reveal whether foundations are encoded
    at different depths in the network.
    """

    def __init__(
        self,
        dataset: ProbingDataset | None = None,
        *,
        n_epochs: int = 50,
        lr: float = 1e-2,
        onset_threshold: float = _ONSET_THRESHOLD,
        min_pairs_per_foundation: int = 5,
    ) -> None:
        self._dataset = dataset
        self._n_epochs = n_epochs
        self._lr = lr
        self._onset_threshold = onset_threshold
        self._min_pairs = min_pairs_per_foundation

    @property
    def name(self) -> str:
        return "foundation_specific_probe"

    @property
    def min_access_tier(self) -> AccessTier:
        return AccessTier.WEIGHTS

    def run(self, model: WhiteBoxModel) -> BenchmarkResult:  # type: ignore[override]
        """Run per-foundation probing across all layers.

        Args:
            model: A WhiteBoxModel with activation capture support.

        Returns:
            FoundationProbingResult with per-(foundation, layer) scores and summaries.
        """
        dataset = self._dataset or build_probing_dataset()
        n_layers = model.info.n_layers
        assert n_layers is not None, "Model must report n_layers"

        # Group pairs by foundation
        train_by_foundation: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
        test_by_foundation: dict[MoralFoundation, list[ProbingPair]] = defaultdict(list)
        for pair in dataset.train:
            train_by_foundation[pair.foundation].append(pair)
        for pair in dataset.test:
            test_by_foundation[pair.foundation].append(pair)

        all_scores: list[FoundationLayerProbeScore] = []
        per_foundation_summary: dict[str, dict] = {}

        for foundation in MoralFoundation:
            train_pairs = train_by_foundation.get(foundation, [])
            test_pairs = test_by_foundation.get(foundation, [])

            if len(train_pairs) < self._min_pairs or len(test_pairs) < 1:
                logger.info(
                    "Skipping %s: %d train, %d test pairs (min=%d)",
                    foundation.value, len(train_pairs), len(test_pairs), self._min_pairs,
                )
                continue

            logger.info(
                "Probing %s: %d train, %d test pairs across %d layers",
                foundation.value, len(train_pairs), len(test_pairs), n_layers,
            )

            # Collect activations for all layers upfront (one forward pass per text)
            all_train = LayerWiseMoralProbe._collect_all_activations(model, train_pairs)
            all_test = LayerWiseMoralProbe._collect_all_activations(model, test_pairs)

            foundation_scores: list[FoundationLayerProbeScore] = []
            for layer_idx in range(n_layers):
                train_X, train_y = all_train[layer_idx]
                test_X, test_y = all_test[layer_idx]
                accuracy, loss = self._train_probe(train_X, train_y, test_X, test_y)
                score = FoundationLayerProbeScore(
                    foundation=foundation,
                    layer=layer_idx,
                    accuracy=accuracy,
                    loss=loss,
                    n_pairs=len(train_pairs),
                )
                foundation_scores.append(score)
                all_scores.append(score)

            # Compute summary for this foundation
            onset_layer = None
            peak_layer = 0
            peak_accuracy = 0.0
            n_above = 0
            for s in foundation_scores:
                if s.accuracy > peak_accuracy:
                    peak_accuracy = s.accuracy
                    peak_layer = s.layer
                if s.accuracy >= self._onset_threshold:
                    n_above += 1
                    if onset_layer is None:
                        onset_layer = s.layer

            depth = (onset_layer / n_layers) if onset_layer is not None else 1.0
            breadth = n_above / n_layers

            per_foundation_summary[foundation.value] = {
                "onset_layer": onset_layer,
                "peak_layer": peak_layer,
                "peak_accuracy": peak_accuracy,
                "depth": depth,
                "breadth": breadth,
            }

        result = FoundationProbingResult(
            benchmark_name=self.name,
            model_info=model.info,
            foundation_layer_scores=all_scores,
            per_foundation_summary=per_foundation_summary,
            metadata={
                "n_epochs": self._n_epochs,
                "lr": self._lr,
                "onset_threshold": self._onset_threshold,
                "min_pairs_per_foundation": self._min_pairs,
            },
        )

        logger.info(
            "FoundationSpecificProbe complete: %d foundations probed",
            len(per_foundation_summary),
        )
        return result

    def _train_probe(
        self,
        train_X: Tensor,
        train_y: Tensor,
        test_X: Tensor,
        test_y: Tensor,
    ) -> tuple[float, float]:
        """Train a linear probe and evaluate on test set.

        Returns:
            (accuracy, loss) on the test set.
        """
        hidden_dim = train_X.shape[1]
        probe = nn.Linear(hidden_dim, 1)
        optimizer = torch.optim.Adam(probe.parameters(), lr=self._lr)
        loss_fn = nn.BCEWithLogitsLoss()

        probe.train()
        for _ in range(self._n_epochs):
            logits = probe(train_X).squeeze(-1)
            loss = loss_fn(logits, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probe.eval()
        with torch.no_grad():
            test_logits = probe(test_X).squeeze(-1)
            test_loss = loss_fn(test_logits, test_y).item()
            preds = (test_logits > 0).float()
            accuracy = (preds == test_y).float().mean().item()

        return accuracy, test_loss
