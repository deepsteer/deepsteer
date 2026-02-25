"""Tests for training monitoring hooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from deepsteer.core.types import (
    AccessTier,
    LayerProbingResult,
    ModelInfo,
    MonitoringSession,
    MonitoringSnapshot,
)
from deepsteer.datasets.types import (
    DatasetMetadata,
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)
from deepsteer.core.types import MoralFoundation
from deepsteer.steering.training_hooks import ProbeMonitor


# ---------------------------------------------------------------------------
# Mock model (same pattern as test_probing.py, but with .model property)
# ---------------------------------------------------------------------------


class MockTrainableModel:
    """Mock WhiteBoxModel with .model property for training mode toggling.

    Moral texts get activations drawn from N(+1, 0.1).
    Neutral texts get activations drawn from N(-1, 0.1).
    """

    def __init__(
        self,
        moral_texts: set[str],
        *,
        n_layers: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        self._moral_texts = moral_texts
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._info = ModelInfo(
            name="mock-trainable",
            provider="test",
            access_tier=AccessTier.WEIGHTS,
            n_layers=n_layers,
        )
        # Real nn.Module to support training/eval mode toggling
        self._model = torch.nn.Linear(hidden_dim, hidden_dim)

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def access_tier(self) -> AccessTier:
        return self._info.access_tier

    @property
    def model(self) -> torch.nn.Module:
        return self._model

    def get_activations(
        self, text: str, layers: list[int] | None = None
    ) -> dict[int, Tensor]:
        if layers is None:
            layers = list(range(self._n_layers))

        is_moral = text in self._moral_texts
        mean = 1.0 if is_moral else -1.0

        result: dict[int, Tensor] = {}
        for layer in layers:
            activation = torch.randn(1, 5, self._hidden_dim) * 0.1 + mean
            result[layer] = activation
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(moral: str, neutral: str) -> ProbingPair:
    return ProbingPair(
        moral=moral,
        neutral=neutral,
        foundation=MoralFoundation.CARE_HARM,
        neutral_domain=NeutralDomain.COOKING,
        generation_method=GenerationMethod.POOL,
        moral_word_count=len(moral.split()),
        neutral_word_count=len(neutral.split()),
    )


def _make_dataset(n_train: int = 15, n_test: int = 5) -> tuple[ProbingDataset, set[str]]:
    moral_texts: set[str] = set()
    train: list[ProbingPair] = []
    test: list[ProbingPair] = []

    for i in range(n_train):
        moral = f"moral train {i}"
        neutral = f"neutral train {i}"
        moral_texts.add(moral)
        train.append(_make_pair(moral, neutral))

    for i in range(n_test):
        moral = f"moral test {i}"
        neutral = f"neutral test {i}"
        moral_texts.add(moral)
        test.append(_make_pair(moral, neutral))

    metadata = DatasetMetadata(
        version="1.0.0", generation_method="test",
        total_pairs=n_train + n_test, train_pairs=n_train, test_pairs=n_test,
    )
    return ProbingDataset(train=train, test=test, metadata=metadata), moral_texts


# ---------------------------------------------------------------------------
# ProbeMonitor tests
# ---------------------------------------------------------------------------


class TestProbeMonitor:
    def test_single_snapshot(self):
        """A single snapshot produces a valid MonitoringSnapshot."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=3)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        snap = monitor.snapshot(step=0)

        assert isinstance(snap, MonitoringSnapshot)
        assert snap.step == 0
        assert isinstance(snap.probing_result, LayerProbingResult)
        assert snap.peak_accuracy is not None
        assert 0.0 <= snap.peak_accuracy <= 1.0

    def test_multiple_snapshots(self):
        """Multiple snapshots are accumulated in the session."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=3)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        monitor.snapshot(step=0)
        monitor.snapshot(step=100)
        monitor.snapshot(step=200)

        session = monitor.session
        assert len(session.snapshots) == 3
        assert session.snapshots[0].step == 0
        assert session.snapshots[1].step == 100
        assert session.snapshots[2].step == 200

    def test_session_model_name(self):
        """Session records the model name."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        assert monitor.session.model_name == "mock-trainable"

    def test_session_metadata(self):
        """Session metadata captures hyperparameters."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=25, lr=0.005)

        meta = monitor.session.metadata
        assert meta["n_epochs"] == 25
        assert meta["lr"] == 0.005

    def test_snapshot_has_probing_metrics(self):
        """Snapshot includes onset, peak, depth, breadth from probing."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=4)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=30)

        snap = monitor.snapshot(step=50)

        # On well-separated data, should find some structure
        assert snap.moral_encoding_depth is not None
        assert snap.moral_encoding_breadth is not None
        assert 0.0 <= snap.moral_encoding_depth <= 1.0
        assert 0.0 <= snap.moral_encoding_breadth <= 1.0

    def test_probing_result_has_all_layers(self):
        """Each snapshot's probing result covers all layers."""
        dataset, moral_texts = _make_dataset()
        n_layers = 4
        model = MockTrainableModel(moral_texts, n_layers=n_layers)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        snap = monitor.snapshot(step=0)
        scored_layers = {s.layer for s in snap.probing_result.layer_scores}
        assert scored_layers == set(range(n_layers))

    def test_training_mode_restored(self):
        """Model's training mode is restored after snapshot."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts)
        model.model.train()
        assert model.model.training is True

        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=10)
        monitor.snapshot(step=0)

        assert model.model.training is True

    def test_eval_mode_preserved(self):
        """If model was in eval mode, it stays in eval mode."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts)
        model.model.eval()
        assert model.model.training is False

        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=10)
        monitor.snapshot(step=0)

        assert model.model.training is False

    def test_save_session(self, tmp_path: Path):
        """Saves monitoring session as valid JSON."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=3)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        monitor.snapshot(step=0)
        monitor.snapshot(step=100)

        json_path = monitor.save(tmp_path / "session.json")
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["model_name"] == "mock-trainable"
        assert len(data["snapshots"]) == 2
        assert data["snapshots"][0]["step"] == 0
        assert data["snapshots"][1]["step"] == 100

    def test_session_serialization(self):
        """Session.to_dict() produces valid JSON."""
        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=3)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        monitor.snapshot(step=0)

        d = monitor.session.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "snapshots" in d
        assert "model_name" in d


# ---------------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------------


class TestTrainingMonitoringVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_training_monitoring

        dataset, moral_texts = _make_dataset()
        model = MockTrainableModel(moral_texts, n_layers=3)
        monitor = ProbeMonitor(model, dataset=dataset, n_epochs=20)

        monitor.snapshot(step=0)
        monitor.snapshot(step=100)
        monitor.snapshot(step=200)

        png_path = plot_training_monitoring(monitor.session, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert len(data["snapshots"]) == 3

    def test_plot_empty_session_raises(self, tmp_path: Path):
        from deepsteer.viz import plot_training_monitoring

        session = MonitoringSession(model_name="test")
        with pytest.raises(ValueError, match="no snapshots"):
            plot_training_monitoring(session, output_dir=tmp_path)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_steering_package_exports(self):
        from deepsteer.steering import ProbeMonitor
        assert ProbeMonitor is not None
