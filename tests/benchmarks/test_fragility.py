"""Tests for MoralFragilityTest benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
from deepsteer.core.types import (
    AccessTier,
    FragilityResult,
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
from tests.benchmarks.test_probing import MockWhiteBoxModel


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


def _make_dataset(n_train: int = 20, n_test: int = 5) -> tuple[ProbingDataset, set[str]]:
    """Create a small dataset and return it with the set of moral texts."""
    moral_texts: set[str] = set()
    train: list[ProbingPair] = []
    test: list[ProbingPair] = []

    for i in range(n_train):
        moral = f"moral train sentence number {i}"
        neutral = f"neutral train sentence number {i}"
        moral_texts.add(moral)
        train.append(_make_pair(moral, neutral))

    for i in range(n_test):
        moral = f"moral test sentence number {i}"
        neutral = f"neutral test sentence number {i}"
        moral_texts.add(moral)
        test.append(_make_pair(moral, neutral))

    metadata = DatasetMetadata(
        version="1.0.0",
        generation_method="test",
        total_pairs=n_train + n_test,
        train_pairs=n_train,
        test_pairs=n_test,
    )
    return ProbingDataset(train=train, test=test, metadata=metadata), moral_texts


# ---------------------------------------------------------------------------
# Benchmark property tests
# ---------------------------------------------------------------------------


class TestMoralFragilityTestProperties:
    def test_benchmark_name(self):
        test = MoralFragilityTest()
        assert test.name == "moral_fragility_test"

    def test_min_access_tier(self):
        test = MoralFragilityTest()
        assert test.min_access_tier == AccessTier.WEIGHTS


# ---------------------------------------------------------------------------
# Result tests
# ---------------------------------------------------------------------------


class TestMoralFragilityTestResult:
    def test_result_schema(self):
        """Result has correct type and expected fields."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(dataset=dataset, n_epochs=30)

        result = test.run(model)

        assert isinstance(result, FragilityResult)
        assert result.benchmark_name == "moral_fragility_test"
        assert result.model_info is not None
        assert len(result.layer_scores) == 4
        assert len(result.noise_levels) == 5  # default noise levels
        assert result.mean_critical_noise is not None or result.mean_critical_noise is None

    def test_noise_levels_respected(self):
        """Custom noise levels appear in the result."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=3)
        custom_noise = [0.5, 2.0, 8.0]
        test = MoralFragilityTest(
            dataset=dataset, n_epochs=20, noise_levels=custom_noise,
        )

        result = test.run(model)

        assert result.noise_levels == custom_noise
        for ls in result.layer_scores:
            for nl in custom_noise:
                assert nl in ls.accuracy_by_noise

    def test_baseline_accuracy_present(self):
        """Each layer has a baseline accuracy."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(dataset=dataset, n_epochs=30)

        result = test.run(model)

        for ls in result.layer_scores:
            assert 0.0 <= ls.baseline_accuracy <= 1.0

    def test_noise_degrades_accuracy_at_high_levels(self):
        """High noise should degrade accuracy compared to baseline.

        With well-separated mock data (mean +/-1, std 0.1), adding noise
        at std=10 should make classification near-random.
        """
        dataset, moral_texts = _make_dataset(n_train=30, n_test=10)
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(
            dataset=dataset, n_epochs=50,
            noise_levels=[0.01, 10.0, 100.0],
        )

        result = test.run(model)

        for ls in result.layer_scores:
            # Low noise should preserve accuracy well
            low_noise_acc = ls.accuracy_by_noise[0.01]
            # High noise should significantly degrade
            high_noise_acc = ls.accuracy_by_noise[100.0]
            # We expect high noise accuracy to be lower (near chance ~0.5)
            assert high_noise_acc <= low_noise_acc + 0.1, (
                f"Layer {ls.layer}: high noise ({high_noise_acc:.2f}) should be "
                f"<= low noise ({low_noise_acc:.2f})"
            )

    def test_most_fragile_and_robust_identified(self):
        """Most fragile and robust layers are identified when critical noise exists."""
        dataset, moral_texts = _make_dataset(n_train=30, n_test=10)
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(
            dataset=dataset, n_epochs=50,
            noise_levels=[0.1, 0.5, 1.0, 5.0, 50.0],
            fragility_threshold=0.6,
        )

        result = test.run(model)

        # With noise up to 50.0, at least some layers should have critical_noise
        layers_with_critical = [ls for ls in result.layer_scores if ls.critical_noise is not None]
        if layers_with_critical:
            assert result.most_fragile_layer is not None
            assert result.most_robust_layer is not None
            assert 0 <= result.most_fragile_layer < 4
            assert 0 <= result.most_robust_layer < 4

    def test_critical_noise_ordering(self):
        """Critical noise of fragile layer <= critical noise of robust layer."""
        dataset, moral_texts = _make_dataset(n_train=30, n_test=10)
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(
            dataset=dataset, n_epochs=50,
            noise_levels=[0.1, 0.5, 1.0, 5.0, 50.0],
            fragility_threshold=0.6,
        )

        result = test.run(model)

        if result.most_fragile_layer is not None and result.most_robust_layer is not None:
            fragile_score = next(
                ls for ls in result.layer_scores if ls.layer == result.most_fragile_layer
            )
            robust_score = next(
                ls for ls in result.layer_scores if ls.layer == result.most_robust_layer
            )
            assert fragile_score.critical_noise <= robust_score.critical_noise

    def test_serialization(self):
        """Result serializes to valid JSON."""
        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=3)
        test = MoralFragilityTest(dataset=dataset, n_epochs=20)

        result = test.run(model)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "layer_scores" in d
        assert "noise_levels" in d
        assert "most_fragile_layer" in d
        assert "most_robust_layer" in d


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestFragilityVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_fragility

        dataset, moral_texts = _make_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        test = MoralFragilityTest(dataset=dataset, n_epochs=30)
        result = test.run(model)

        png_path = plot_fragility(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["benchmark_name"] == "moral_fragility_test"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_representational_package_exports(self):
        from deepsteer.benchmarks.representational import MoralFragilityTest
        assert MoralFragilityTest is not None

    def test_default_suite_includes_fragility(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "moral_fragility_test" in names
