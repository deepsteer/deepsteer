"""Training hooks: monitor moral probing metrics during live training.

Provides a :class:`ProbeMonitor` that periodically runs layer-wise moral
probing on a model during training and records how moral encoding evolves.
Uses the existing probing infrastructure from :mod:`deepsteer.benchmarks`.

DeepSteer does NOT execute training.  The monitor is designed to be called
from a researcher's training loop at specified intervals.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe
from deepsteer.core.model_interface import WhiteBoxModel
from deepsteer.core.types import (
    LayerProbeScore,
    LayerProbingResult,
    ModelInfo,
    MonitoringSession,
    MonitoringSnapshot,
)
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.types import ProbingDataset, ProbingPair

logger = logging.getLogger(__name__)

_ONSET_THRESHOLD = 0.6


class ProbeMonitor:
    """Monitor moral probing metrics during training.

    Intended to be called from a training loop at regular intervals.
    Each call to :meth:`snapshot` runs a fast layer-wise probing evaluation
    and records the result.  The full session can be saved as JSON.

    Example usage in a training loop::

        monitor = ProbeMonitor(model, dataset=probing_dataset)
        for step in range(total_steps):
            train_step(model, batch)
            if step % 500 == 0:
                monitor.snapshot(step)
        monitor.save("outputs/monitoring_session.json")

    Args:
        model: A WhiteBoxModel to probe (the same model being trained).
        dataset: Probing dataset.  If ``None``, builds the default one.
        n_epochs: Training epochs per probe (lower = faster, noisier).
        lr: Learning rate for the probing classifier.
        onset_threshold: Accuracy threshold for onset layer detection.
    """

    def __init__(
        self,
        model: WhiteBoxModel,
        dataset: ProbingDataset | None = None,
        *,
        n_epochs: int = 30,
        lr: float = 1e-2,
        onset_threshold: float = _ONSET_THRESHOLD,
    ) -> None:
        self._model = model
        self._dataset = dataset or build_probing_dataset()
        self._n_epochs = n_epochs
        self._lr = lr
        self._onset_threshold = onset_threshold
        self._session = MonitoringSession(
            model_name=model.info.name,
            metadata={
                "n_epochs": n_epochs,
                "lr": lr,
                "onset_threshold": onset_threshold,
                "train_pairs": len(self._dataset.train),
                "test_pairs": len(self._dataset.test),
            },
        )

    @property
    def session(self) -> MonitoringSession:
        """The current monitoring session with all recorded snapshots."""
        return self._session

    def snapshot(self, step: int) -> MonitoringSnapshot:
        """Run probing evaluation and record a snapshot at the current step.

        This temporarily sets the model to eval mode, runs the probe, then
        restores training mode.  No gradients are computed.

        Args:
            step: The current training step number.

        Returns:
            The recorded MonitoringSnapshot.
        """
        was_training = self._model.model.training
        self._model.model.eval()

        try:
            probing_result = self._run_probe(step)
        finally:
            if was_training:
                self._model.model.train()

        snap = MonitoringSnapshot(
            step=step,
            probing_result=probing_result,
            onset_layer=probing_result.onset_layer,
            peak_layer=probing_result.peak_layer,
            peak_accuracy=probing_result.peak_accuracy,
            moral_encoding_depth=probing_result.moral_encoding_depth,
            moral_encoding_breadth=probing_result.moral_encoding_breadth,
        )
        self._session.snapshots.append(snap)

        logger.info(
            "Step %d: onset=%s, peak=%s (%.1f%%), depth=%.3f, breadth=%.3f",
            step,
            probing_result.onset_layer,
            probing_result.peak_layer,
            (probing_result.peak_accuracy or 0.0) * 100,
            probing_result.moral_encoding_depth or 0.0,
            probing_result.moral_encoding_breadth or 0.0,
        )
        return snap

    def save(self, path: str | Path) -> Path:
        """Save the monitoring session as JSON.

        Args:
            path: Output file path.

        Returns:
            Path to the saved JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._session.to_dict(), f, indent=2)
        logger.info("Saved monitoring session: %s (%d snapshots)", path, len(self._session.snapshots))
        return path

    def _run_probe(self, step: int) -> LayerProbingResult:
        """Run a full layer-wise probing evaluation."""
        train_pairs = self._dataset.train
        test_pairs = self._dataset.test
        n_layers = self._model.info.n_layers
        assert n_layers is not None, "Model must report n_layers"

        layer_scores: list[LayerProbeScore] = []

        for layer_idx in range(n_layers):
            train_X, train_y = LayerWiseMoralProbe._collect_activations(
                self._model, train_pairs, layer_idx,
            )
            test_X, test_y = LayerWiseMoralProbe._collect_activations(
                self._model, test_pairs, layer_idx,
            )
            accuracy, loss = self._train_probe(train_X, train_y, test_X, test_y)
            layer_scores.append(LayerProbeScore(
                layer=layer_idx, accuracy=accuracy, loss=loss,
            ))

        # Compute summary metrics
        onset_layer = None
        peak_layer = 0
        peak_accuracy = 0.0
        n_above = 0

        for score in layer_scores:
            if score.accuracy > peak_accuracy:
                peak_accuracy = score.accuracy
                peak_layer = score.layer
            if score.accuracy >= self._onset_threshold:
                n_above += 1
                if onset_layer is None:
                    onset_layer = score.layer

        depth = (onset_layer / n_layers) if onset_layer is not None else 1.0
        breadth = n_above / n_layers

        return LayerProbingResult(
            benchmark_name="probe_monitor",
            model_info=self._model.info,
            layer_scores=layer_scores,
            onset_layer=onset_layer,
            peak_layer=peak_layer,
            peak_accuracy=peak_accuracy,
            moral_encoding_depth=depth,
            moral_encoding_breadth=breadth,
            checkpoint_step=step,
            metadata={
                "n_epochs": self._n_epochs,
                "lr": self._lr,
                "onset_threshold": self._onset_threshold,
            },
        )

    def _train_probe(
        self,
        train_X: Tensor,
        train_y: Tensor,
        test_X: Tensor,
        test_y: Tensor,
    ) -> tuple[float, float]:
        """Train a linear probe and return (accuracy, loss) on test set."""
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
