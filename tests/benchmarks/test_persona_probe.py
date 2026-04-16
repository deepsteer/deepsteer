"""Tests for PersonaFeatureProbe (Phase D, C7/C8 setup)."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

from deepsteer.benchmarks.representational.persona_probe import PersonaFeatureProbe
from deepsteer.core.types import AccessTier, LayerProbingResult, ModelInfo


class MockWhiteBoxModel:
    """Deterministic fake WhiteBoxModel for probe testing.

    Texts in ``persona_texts`` get activations drawn from ``N(+1, 0.1)``;
    all others from ``N(-1, 0.1)``.
    """

    def __init__(
        self,
        persona_texts: set[str],
        *,
        n_layers: int = 4,
        hidden_dim: int = 32,
    ) -> None:
        self._persona_texts = persona_texts
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._info = ModelInfo(
            name="mock-model",
            provider="test",
            access_tier=AccessTier.WEIGHTS,
            n_layers=n_layers,
        )

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def access_tier(self) -> AccessTier:
        return self._info.access_tier

    def get_activations(
        self, text: str, layers: list[int] | None = None
    ) -> dict[int, Tensor]:
        if layers is None:
            layers = list(range(self._n_layers))
        mean = 1.0 if text in self._persona_texts else -1.0
        return {
            layer: torch.randn(1, 5, self._hidden_dim) * 0.1 + mean
            for layer in layers
        }


def _make_split(n_train: int = 20, n_test: int = 5) -> tuple[
    list[tuple[str, str]], list[tuple[str, str]], set[str]
]:
    """Build synthetic (persona_voice, neutral) splits and the persona-text set."""
    persona_texts: set[str] = set()
    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    for i in range(n_train):
        p = f"persona train line {i}"
        n = f"neutral train line {i}"
        persona_texts.add(p)
        train.append((p, n))
    for i in range(n_test):
        p = f"persona test line {i}"
        n = f"neutral test line {i}"
        persona_texts.add(p)
        test.append((p, n))
    return train, test, persona_texts


class TestPersonaFeatureProbeProperties:
    def test_name_is_stable(self):
        probe = PersonaFeatureProbe()
        assert probe.name == "persona_feature_probe"

    def test_access_tier_is_weights(self):
        probe = PersonaFeatureProbe()
        assert probe.min_access_tier == AccessTier.WEIGHTS

    def test_unmatched_train_test_args_rejected(self):
        with pytest.raises(ValueError, match="both be provided or both be None"):
            PersonaFeatureProbe(train_pairs=[("a", "b")], test_pairs=None)


class TestPersonaFeatureProbeRun:
    def test_result_schema(self):
        train, test, persona_texts = _make_split()
        model = MockWhiteBoxModel(persona_texts, n_layers=4)
        probe = PersonaFeatureProbe(
            train_pairs=train, test_pairs=test, n_epochs=30,
        )

        result = probe.run(model)

        assert isinstance(result, LayerProbingResult)
        assert result.benchmark_name == "persona_feature_probe"
        assert len(result.layer_scores) == 4
        assert result.peak_layer is not None
        assert 0.0 <= result.peak_accuracy <= 1.0
        assert 0.0 <= result.moral_encoding_depth <= 1.0
        assert 0.0 <= result.moral_encoding_breadth <= 1.0

    def test_high_accuracy_on_separable_data(self):
        train, test, persona_texts = _make_split(n_train=30, n_test=10)
        model = MockWhiteBoxModel(persona_texts, n_layers=4)
        probe = PersonaFeatureProbe(
            train_pairs=train, test_pairs=test, n_epochs=50,
        )

        result = probe.run(model)
        assert result.peak_accuracy >= 0.8, (
            f"Expected high accuracy on separable data, got {result.peak_accuracy:.2f}"
        )

    def test_default_dataset_is_used_when_pairs_omitted(self):
        # Use the built-in dataset (240 persona pairs).  No model forward
        # passes needed — we only check that construction + split works.
        probe = PersonaFeatureProbe()
        train, test = __import__(
            "deepsteer.datasets.persona_pairs", fromlist=["get_persona_dataset"]
        ).get_persona_dataset(seed=probe._split_seed, stratified=True)
        assert len(train) + len(test) == 240
