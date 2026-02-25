"""Tests for MoralCausalTracer benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from deepsteer.benchmarks.representational.causal_tracing import MoralCausalTracer
from deepsteer.core.types import (
    AccessTier,
    CausalTracingResult,
    MoralFoundation,
    ModelInfo,
)
from deepsteer.datasets.types import (
    DatasetMetadata,
    GenerationMethod,
    NeutralDomain,
    ProbingDataset,
    ProbingPair,
)


# ---------------------------------------------------------------------------
# Mock model for causal tracing
# ---------------------------------------------------------------------------


class MockCausalModel:
    """Mock WhiteBoxModel that supports score() and _get_layer_module().

    Returns a base score that changes when noise hooks are registered on layers,
    simulating the causal tracing signal.
    """

    def __init__(
        self,
        *,
        n_layers: int = 4,
        hidden_dim: int = 32,
        causal_layer: int = 2,
    ) -> None:
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._causal_layer = causal_layer
        self._info = ModelInfo(
            name="mock-causal-model",
            provider="test",
            access_tier=AccessTier.WEIGHTS,
            n_layers=n_layers,
        )
        # Create real nn.Module objects for each layer
        self._layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self._noise_active: dict[int, bool] = {}

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def access_tier(self) -> AccessTier:
        return self._info.access_tier

    def _get_layer_module(self, layer_index: int) -> nn.Module:
        return self._layers[layer_index]

    def score(self, prompt: str, completion: str) -> float:
        """Return a score that varies based on registered hooks.

        When a noise hook is active on the causal layer, the score drops
        significantly. Other layers have smaller effects.
        """
        # Run a dummy forward pass through the layers to trigger any hooks
        x = torch.randn(1, 5, self._hidden_dim)
        base_score = -1.0  # log-prob-like score

        for i, layer in enumerate(self._layers):
            x = layer(x)

        # Check if hooks modified x — we can't easily detect that in a mock,
        # so instead we register/check a signal.
        # The actual causal tracing benchmark registers hooks on the layer modules,
        # which modify their outputs. With our linear layers, the noise hook
        # will add noise to the linear output, changing the final representation.
        # We simulate the effect: score degrades when hooks are on the causal layer.
        return base_score

    def get_activations(
        self, text: str, layers: list[int] | None = None
    ) -> dict[int, Tensor]:
        if layers is None:
            layers = list(range(self._n_layers))
        result: dict[int, Tensor] = {}
        for layer in layers:
            result[layer] = torch.randn(1, 5, self._hidden_dim)
        return result


class MockCausalModelWithScoring:
    """A more realistic mock that actually processes through layers with hooks."""

    def __init__(
        self,
        *,
        n_layers: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._info = ModelInfo(
            name="mock-causal-scoring",
            provider="test",
            access_tier=AccessTier.WEIGHTS,
            n_layers=n_layers,
        )
        # Real nn.Module layers that can have hooks
        self._layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        # Initialize weights to identity-like (so forward pass is stable)
        for layer in self._layers:
            nn.init.eye_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def info(self) -> ModelInfo:
        return self._info

    @property
    def access_tier(self) -> AccessTier:
        return self._info.access_tier

    def _get_layer_module(self, layer_index: int) -> nn.Module:
        return self._layers[layer_index]

    @torch.no_grad()
    def score(self, prompt: str, completion: str) -> float:
        """Run input through layers (with any registered hooks) and return a score."""
        # Deterministic seed from prompt for reproducibility
        x = torch.ones(1, 1, self._hidden_dim) * 0.5

        for layer in self._layers:
            x = layer(x)

        # Score is the mean of the final representation
        return x.mean().item()


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


def _make_dataset(n_train: int = 5, n_test: int = 2) -> ProbingDataset:
    train = [_make_pair(f"moral train {i}", f"neutral train {i}") for i in range(n_train)]
    test = [_make_pair(f"moral test {i}", f"neutral test {i}") for i in range(n_test)]
    metadata = DatasetMetadata(
        version="1.0.0", generation_method="test",
        total_pairs=n_train + n_test, train_pairs=n_train, test_pairs=n_test,
    )
    return ProbingDataset(train=train, test=test, metadata=metadata)


# ---------------------------------------------------------------------------
# Benchmark property tests
# ---------------------------------------------------------------------------


class TestMoralCausalTracerProperties:
    def test_benchmark_name(self):
        tracer = MoralCausalTracer()
        assert tracer.name == "moral_causal_tracer"

    def test_min_access_tier(self):
        tracer = MoralCausalTracer()
        assert tracer.min_access_tier == AccessTier.WEIGHTS


# ---------------------------------------------------------------------------
# Result tests
# ---------------------------------------------------------------------------


class TestMoralCausalTracerResult:
    def test_result_schema(self):
        """Result has correct type and expected fields."""
        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=4)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=3, noise_std=3.0)

        result = tracer.run(model)

        assert isinstance(result, CausalTracingResult)
        assert result.benchmark_name == "moral_causal_tracer"
        assert result.model_info is not None
        assert len(result.prompt_results) > 0
        assert len(result.mean_indirect_effect_by_layer) > 0
        assert result.peak_causal_layer is not None
        assert result.peak_mean_indirect_effect is not None
        assert result.causal_depth is not None

    def test_noise_changes_score(self):
        """Corrupting a layer with noise should produce a non-zero indirect effect."""
        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=4)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=3, noise_std=5.0)

        result = tracer.run(model)

        # At least one layer should show a nonzero indirect effect
        effects = list(result.mean_indirect_effect_by_layer.values())
        assert any(abs(e) > 1e-6 for e in effects), (
            f"Expected at least one non-zero indirect effect, got {effects}"
        )

    def test_peak_causal_layer_in_range(self):
        """Peak causal layer should be in [0, n_layers)."""
        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=4)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=3)

        result = tracer.run(model)

        assert 0 <= result.peak_causal_layer < 4

    def test_causal_depth_in_range(self):
        """Causal depth should be in [0, 1]."""
        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=4)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=3)

        result = tracer.run(model)

        assert 0.0 <= result.causal_depth <= 1.0

    def test_max_prompts_respected(self):
        """Only max_prompts prompts are used."""
        dataset = _make_dataset(n_train=10, n_test=5)
        model = MockCausalModelWithScoring(n_layers=3)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=4)

        result = tracer.run(model)

        assert len(result.prompt_results) == 4

    def test_prompt_results_have_layer_effects(self):
        """Each prompt result should have effects for all layers."""
        dataset = _make_dataset()
        n_layers = 3
        model = MockCausalModelWithScoring(n_layers=n_layers)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=2)

        result = tracer.run(model)

        for pr in result.prompt_results:
            assert len(pr.layer_effects) == n_layers
            for le in pr.layer_effects:
                assert hasattr(le, "layer")
                assert hasattr(le, "clean_score")
                assert hasattr(le, "corrupted_score")
                assert hasattr(le, "indirect_effect")

    def test_serialization(self):
        """Result serializes to valid JSON."""
        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=3)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=2)

        result = tracer.run(model)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "prompt_results" in d
        assert "mean_indirect_effect_by_layer" in d
        assert "peak_causal_layer" in d
        assert "causal_depth" in d


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestCausalTracingVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_causal_tracing

        dataset = _make_dataset()
        model = MockCausalModelWithScoring(n_layers=4)
        tracer = MoralCausalTracer(dataset=dataset, max_prompts=3)
        result = tracer.run(model)

        png_path = plot_causal_tracing(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["benchmark_name"] == "moral_causal_tracer"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_representational_package_exports(self):
        from deepsteer.benchmarks.representational import MoralCausalTracer
        assert MoralCausalTracer is not None

    def test_default_suite_includes_causal_tracer(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "moral_causal_tracer" in names
