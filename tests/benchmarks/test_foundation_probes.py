"""Tests for FoundationSpecificProbe benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from deepsteer.benchmarks.representational.foundation_probes import FoundationSpecificProbe
from deepsteer.core.types import (
    AccessTier,
    FoundationProbingResult,
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

# Cycle through foundations to ensure all 6 are represented
_FOUNDATIONS = list(MoralFoundation)
_DOMAINS = list(NeutralDomain)


def _make_pair(moral: str, neutral: str, foundation: MoralFoundation) -> ProbingPair:
    return ProbingPair(
        moral=moral,
        neutral=neutral,
        foundation=foundation,
        neutral_domain=_DOMAINS[0],
        generation_method=GenerationMethod.POOL,
        moral_word_count=len(moral.split()),
        neutral_word_count=len(neutral.split()),
    )


def _make_foundation_dataset(
    pairs_per_foundation: int = 8,
    test_per_foundation: int = 3,
) -> tuple[ProbingDataset, set[str]]:
    """Create a dataset with pairs spread across all 6 foundations."""
    moral_texts: set[str] = set()
    train: list[ProbingPair] = []
    test: list[ProbingPair] = []

    for fi, foundation in enumerate(_FOUNDATIONS):
        for i in range(pairs_per_foundation):
            moral = f"moral {foundation.value} train sentence {i}"
            neutral = f"neutral {foundation.value} train sentence {i}"
            moral_texts.add(moral)
            train.append(_make_pair(moral, neutral, foundation))
        for i in range(test_per_foundation):
            moral = f"moral {foundation.value} test sentence {i}"
            neutral = f"neutral {foundation.value} test sentence {i}"
            moral_texts.add(moral)
            test.append(_make_pair(moral, neutral, foundation))

    total = len(train) + len(test)
    metadata = DatasetMetadata(
        version="1.0.0",
        generation_method="test",
        total_pairs=total,
        train_pairs=len(train),
        test_pairs=len(test),
    )
    return ProbingDataset(train=train, test=test, metadata=metadata), moral_texts


# ---------------------------------------------------------------------------
# Benchmark property tests
# ---------------------------------------------------------------------------


class TestFoundationSpecificProbeProperties:
    def test_benchmark_name(self):
        probe = FoundationSpecificProbe()
        assert probe.name == "foundation_specific_probe"

    def test_min_access_tier(self):
        probe = FoundationSpecificProbe()
        assert probe.min_access_tier == AccessTier.WEIGHTS


# ---------------------------------------------------------------------------
# Result tests
# ---------------------------------------------------------------------------


class TestFoundationSpecificProbeResult:
    def test_result_schema(self):
        """Result has correct type and expected fields."""
        dataset, moral_texts = _make_foundation_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        assert isinstance(result, FoundationProbingResult)
        assert result.benchmark_name == "foundation_specific_probe"
        assert result.model_info is not None
        assert len(result.foundation_layer_scores) > 0
        assert len(result.per_foundation_summary) > 0

    def test_all_foundations_probed(self):
        """All 6 foundations appear in the summary."""
        dataset, moral_texts = _make_foundation_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        for foundation in MoralFoundation:
            assert foundation.value in result.per_foundation_summary, (
                f"Missing foundation: {foundation.value}"
            )

    def test_skips_sparse_foundations(self):
        """Foundations with too few pairs are skipped."""
        # Only provide pairs for care_harm
        moral_texts: set[str] = set()
        train: list[ProbingPair] = []
        test: list[ProbingPair] = []
        for i in range(10):
            moral = f"moral care train {i}"
            neutral = f"neutral care train {i}"
            moral_texts.add(moral)
            train.append(_make_pair(moral, neutral, MoralFoundation.CARE_HARM))
        for i in range(3):
            moral = f"moral care test {i}"
            neutral = f"neutral care test {i}"
            moral_texts.add(moral)
            test.append(_make_pair(moral, neutral, MoralFoundation.CARE_HARM))

        dataset = ProbingDataset(
            train=train, test=test,
            metadata=DatasetMetadata(
                version="1.0.0", generation_method="test",
                total_pairs=13, train_pairs=10, test_pairs=3,
            ),
        )
        model = MockWhiteBoxModel(moral_texts, n_layers=3)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=20)

        result = probe.run(model)

        # Only care_harm should be probed
        assert MoralFoundation.CARE_HARM.value in result.per_foundation_summary
        assert len(result.per_foundation_summary) == 1

    def test_high_accuracy_on_separable_data(self):
        """Mock model produces well-separated activations → high accuracy."""
        dataset, moral_texts = _make_foundation_dataset(
            pairs_per_foundation=12, test_per_foundation=4,
        )
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=50)

        result = probe.run(model)

        # At least some foundation should hit high accuracy
        peak_accs = [
            summary["peak_accuracy"]
            for summary in result.per_foundation_summary.values()
        ]
        assert max(peak_accs) >= 0.7, f"Expected high accuracy, got max {max(peak_accs):.2f}"

    def test_per_foundation_summary_fields(self):
        """Summary has onset_layer, peak_layer, peak_accuracy, depth, breadth."""
        dataset, moral_texts = _make_foundation_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=30)

        result = probe.run(model)

        for fval, summary in result.per_foundation_summary.items():
            assert "onset_layer" in summary
            assert "peak_layer" in summary
            assert "peak_accuracy" in summary
            assert "depth" in summary
            assert "breadth" in summary
            assert 0.0 <= summary["peak_accuracy"] <= 1.0
            assert 0.0 <= summary["depth"] <= 1.0
            assert 0.0 <= summary["breadth"] <= 1.0

    def test_serialization(self):
        """Result serializes to valid JSON."""
        dataset, moral_texts = _make_foundation_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=3)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=20)

        result = probe.run(model)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "foundation_layer_scores" in d
        assert "per_foundation_summary" in d


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestFoundationProbesVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_foundation_probes

        dataset, moral_texts = _make_foundation_dataset()
        model = MockWhiteBoxModel(moral_texts, n_layers=4)
        probe = FoundationSpecificProbe(dataset=dataset, n_epochs=30)
        result = probe.run(model)

        png_path = plot_foundation_probes(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["benchmark_name"] == "foundation_specific_probe"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_representational_package_exports(self):
        from deepsteer.benchmarks.representational import FoundationSpecificProbe
        assert FoundationSpecificProbe is not None

    def test_default_suite_includes_foundation_probes(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "foundation_specific_probe" in names
