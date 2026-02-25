"""Tests for data mixing utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepsteer.core.types import MoralFoundation, MixedSample, MixingResult
from deepsteer.steering.data_mixing import DataMixer
from deepsteer.steering.moral_curriculum import constant_schedule


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_corpus() -> tuple[dict[str, list[str]], list[str]]:
    """Create small moral and general corpora for testing."""
    moral: dict[str, list[str]] = {}
    for f in MoralFoundation:
        moral[f.value] = [f"Moral text about {f.value} number {i}" for i in range(10)]
    general = [f"General text about topic {i}" for i in range(50)]
    return moral, general


# ---------------------------------------------------------------------------
# DataMixer construction
# ---------------------------------------------------------------------------


class TestDataMixerConstruction:
    def test_creates_with_valid_input(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        assert mixer.n_moral == 60  # 6 foundations * 10
        assert mixer.n_general == 50

    def test_empty_moral_corpus(self):
        mixer = DataMixer({}, ["text1", "text2"])
        assert mixer.n_moral == 0
        assert mixer.n_general == 2

    def test_empty_general_corpus(self):
        moral, _ = _make_corpus()
        mixer = DataMixer(moral, [])
        assert mixer.n_general == 0


# ---------------------------------------------------------------------------
# mix_batch
# ---------------------------------------------------------------------------


class TestMixBatch:
    def test_batch_size_respected(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        samples, result = mixer.mix_batch(100, 0.10)
        assert len(samples) == 100
        assert result.total_samples == 100

    def test_moral_ratio_approximate(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        samples, result = mixer.mix_batch(100, 0.20)
        # Should be close to 20 moral samples
        assert result.moral_samples == 20
        assert result.general_samples == 80

    def test_zero_ratio_all_general(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        samples, result = mixer.mix_batch(50, 0.0)
        assert result.moral_samples == 0
        assert result.general_samples == 50
        assert all(not s.is_moral for s in samples)

    def test_full_ratio_all_moral(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        samples, result = mixer.mix_batch(50, 1.0)
        assert result.moral_samples == 50
        assert result.general_samples == 0
        assert all(s.is_moral for s in samples)

    def test_samples_have_correct_fields(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        samples, _ = mixer.mix_batch(20, 0.50)

        for s in samples:
            assert isinstance(s, MixedSample)
            assert isinstance(s.text, str)
            assert isinstance(s.is_moral, bool)
            if s.is_moral:
                assert s.foundation is not None
                assert s.source == "moral_corpus"
            else:
                assert s.foundation is None
                assert s.source == "general_corpus"

    def test_foundation_counts_populated(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        _, result = mixer.mix_batch(100, 0.50)
        # Some foundations should appear
        assert len(result.foundation_counts) > 0
        total = sum(result.foundation_counts.values())
        assert total == result.moral_samples

    def test_foundation_weights_bias_sampling(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general, seed=42)
        # Heavily weight care_harm
        weights = {f.value: 0.0 for f in MoralFoundation}
        weights["care_harm"] = 1.0
        _, result = mixer.mix_batch(100, 0.50, foundation_weights=weights)
        # All or most moral samples should be care_harm
        care_count = result.foundation_counts.get("care_harm", 0)
        assert care_count == result.moral_samples

    def test_result_serialization(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        _, result = mixer.mix_batch(50, 0.10)
        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "total_samples" in d
        assert "moral_ratio" in d
        assert "foundation_counts" in d

    def test_deterministic_with_same_seed(self):
        moral, general = _make_corpus()
        mixer1 = DataMixer(moral, general, seed=99)
        mixer2 = DataMixer(moral, general, seed=99)
        samples1, _ = mixer1.mix_batch(50, 0.30)
        samples2, _ = mixer2.mix_batch(50, 0.30)
        texts1 = [s.text for s in samples1]
        texts2 = [s.text for s in samples2]
        assert texts1 == texts2


# ---------------------------------------------------------------------------
# mix_from_schedule
# ---------------------------------------------------------------------------


class TestMixFromSchedule:
    def test_produces_batches_for_each_phase(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        schedule = constant_schedule(1000, 0.05)
        batches = mixer.mix_from_schedule(schedule, batch_size=20)
        # Default: one batch per phase midpoint
        assert len(batches) == len(schedule.phases)

    def test_custom_steps(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        schedule = constant_schedule(1000, 0.05)
        batches = mixer.mix_from_schedule(schedule, batch_size=20, steps=[0, 500, 999])
        assert len(batches) == 3
        for step, samples, result in batches:
            assert len(samples) == 20
            assert result.metadata["step"] == step

    def test_varying_ratio_schedule(self):
        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        from deepsteer.steering.moral_curriculum import phased_schedule
        schedule = phased_schedule(1000, [
            (0.5, 0.0, "no_moral"),
            (0.5, 0.50, "high_moral"),
        ])
        batches = mixer.mix_from_schedule(schedule, batch_size=100)
        # First phase should have 0 moral, second should have ~50
        _, _, result_low = batches[0]
        _, _, result_high = batches[1]
        assert result_low.moral_samples == 0
        assert result_high.moral_samples == 50


# ---------------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------------


class TestMixingVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_mixing_distribution

        moral, general = _make_corpus()
        mixer = DataMixer(moral, general)
        _, result = mixer.mix_batch(100, 0.30)

        png_path = plot_mixing_distribution(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert "total_samples" in data

    def test_plot_no_moral_samples(self, tmp_path: Path):
        from deepsteer.viz import plot_mixing_distribution

        result = MixingResult(
            total_samples=50, moral_samples=0, general_samples=50,
            moral_ratio=0.0, foundation_counts={},
        )
        png_path = plot_mixing_distribution(result, output_dir=tmp_path)
        assert png_path.exists()


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_steering_package_exports(self):
        from deepsteer.steering import DataMixer
        assert DataMixer is not None
