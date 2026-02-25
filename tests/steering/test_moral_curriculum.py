"""Tests for moral curriculum schedule design tools."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from deepsteer.core.types import CurriculumPhase, CurriculumSchedule, MoralFoundation
from deepsteer.steering.moral_curriculum import (
    constant_schedule,
    cyclical_schedule,
    linear_ramp_schedule,
    phased_schedule,
)


# ---------------------------------------------------------------------------
# constant_schedule
# ---------------------------------------------------------------------------


class TestConstantSchedule:
    def test_single_phase(self):
        s = constant_schedule(10000, 0.05)
        assert len(s.phases) == 1
        assert s.total_steps == 10000
        assert s.method == "constant"

    def test_ratio_correct(self):
        s = constant_schedule(10000, 0.10)
        assert s.phases[0].moral_ratio == 0.10

    def test_covers_all_steps(self):
        s = constant_schedule(5000, 0.03)
        assert s.phases[0].start_step == 0
        assert s.phases[0].end_step == 5000

    def test_uniform_foundation_weights(self):
        s = constant_schedule(1000, 0.05)
        weights = s.phases[0].foundation_weights
        assert len(weights) == 6
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_custom_foundation_weights(self):
        custom = {"care_harm": 0.5, "fairness_cheating": 0.5}
        s = constant_schedule(1000, 0.05, foundation_weights=custom)
        assert s.phases[0].foundation_weights == custom

    def test_ratio_at_step(self):
        s = constant_schedule(1000, 0.07)
        assert s.ratio_at_step(0) == 0.07
        assert s.ratio_at_step(500) == 0.07
        assert s.ratio_at_step(999) == 0.07

    def test_ratio_outside_range_returns_zero(self):
        s = constant_schedule(1000, 0.07)
        assert s.ratio_at_step(1000) == 0.0
        assert s.ratio_at_step(2000) == 0.0

    def test_serialization(self):
        s = constant_schedule(1000, 0.05)
        d = s.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "phases" in d
        assert "method" in d
        assert d["method"] == "constant"

    def test_metadata(self):
        s = constant_schedule(2000, 0.08)
        assert s.metadata["moral_ratio"] == 0.08


# ---------------------------------------------------------------------------
# linear_ramp_schedule
# ---------------------------------------------------------------------------


class TestLinearRampSchedule:
    def test_phase_count(self):
        s = linear_ramp_schedule(10000, n_phases=5)
        assert len(s.phases) == 5

    def test_method_name(self):
        s = linear_ramp_schedule(1000)
        assert s.method == "linear_ramp"

    def test_ratio_increases(self):
        s = linear_ramp_schedule(10000, start_ratio=0.0, end_ratio=0.10, n_phases=10)
        ratios = [p.moral_ratio for p in s.phases]
        # Should be monotonically increasing
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1]

    def test_start_and_end_ratios(self):
        s = linear_ramp_schedule(10000, start_ratio=0.02, end_ratio=0.08, n_phases=5)
        assert abs(s.phases[0].moral_ratio - 0.02) < 1e-6
        assert abs(s.phases[-1].moral_ratio - 0.08) < 1e-6

    def test_covers_all_steps(self):
        s = linear_ramp_schedule(10000, n_phases=4)
        assert s.phases[0].start_step == 0
        assert s.phases[-1].end_step == 10000

    def test_no_gaps_between_phases(self):
        s = linear_ramp_schedule(10000, n_phases=6)
        for i in range(1, len(s.phases)):
            assert s.phases[i].start_step == s.phases[i - 1].end_step

    def test_ratio_at_step_within_ramp(self):
        s = linear_ramp_schedule(10000, start_ratio=0.0, end_ratio=0.10, n_phases=10)
        # First phase should have start_ratio
        assert s.ratio_at_step(0) == 0.0
        # Last phase should have end_ratio
        last_start = s.phases[-1].start_step
        assert abs(s.ratio_at_step(last_start) - 0.10) < 1e-6

    def test_serialization(self):
        s = linear_ramp_schedule(5000)
        d = s.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["method"] == "linear_ramp"


# ---------------------------------------------------------------------------
# cyclical_schedule
# ---------------------------------------------------------------------------


class TestCyclicalSchedule:
    def test_method_name(self):
        s = cyclical_schedule(10000)
        assert s.method == "cyclical"

    def test_has_multiple_phases(self):
        s = cyclical_schedule(10000, cycle_length=1000)
        assert len(s.phases) > 1

    def test_ratios_bounded(self):
        s = cyclical_schedule(10000, min_ratio=0.01, max_ratio=0.10)
        for p in s.phases:
            assert p.moral_ratio >= 0.0
            assert p.moral_ratio <= 1.0

    def test_covers_all_steps(self):
        s = cyclical_schedule(5000)
        assert s.phases[0].start_step == 0
        assert s.phases[-1].end_step == 5000

    def test_serialization(self):
        s = cyclical_schedule(3000)
        d = s.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["method"] == "cyclical"

    def test_metadata(self):
        s = cyclical_schedule(5000, min_ratio=0.02, max_ratio=0.08, cycle_length=500)
        assert s.metadata["min_ratio"] == 0.02
        assert s.metadata["max_ratio"] == 0.08
        assert s.metadata["cycle_length"] == 500


# ---------------------------------------------------------------------------
# phased_schedule
# ---------------------------------------------------------------------------


class TestPhasedSchedule:
    def test_two_phase_schedule(self):
        s = phased_schedule(10000, [
            (0.5, 0.02, "warmup"),
            (0.5, 0.10, "main"),
        ])
        assert len(s.phases) == 2
        assert s.method == "phased"

    def test_phase_boundaries(self):
        s = phased_schedule(10000, [
            (0.3, 0.01, "low"),
            (0.7, 0.05, "high"),
        ])
        assert s.phases[0].start_step == 0
        assert s.phases[0].end_step == 3000
        assert s.phases[1].start_step == 3000
        assert s.phases[1].end_step == 10000

    def test_ratios_match_config(self):
        s = phased_schedule(10000, [
            (0.4, 0.02, "a"),
            (0.6, 0.08, "b"),
        ])
        assert s.phases[0].moral_ratio == 0.02
        assert s.phases[1].moral_ratio == 0.08

    def test_labels_preserved(self):
        s = phased_schedule(1000, [
            (0.5, 0.01, "warmup"),
            (0.5, 0.05, "training"),
        ])
        assert s.phases[0].label == "warmup"
        assert s.phases[1].label == "training"

    def test_fractions_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to"):
            phased_schedule(1000, [
                (0.3, 0.01, "a"),
                (0.3, 0.02, "b"),
            ])

    def test_three_phase_schedule(self):
        s = phased_schedule(9000, [
            (1 / 3, 0.01, "warmup"),
            (1 / 3, 0.05, "main"),
            (1 / 3, 0.10, "intensive"),
        ])
        assert len(s.phases) == 3
        assert s.phases[-1].end_step == 9000

    def test_ratio_at_step(self):
        s = phased_schedule(1000, [
            (0.5, 0.02, "low"),
            (0.5, 0.10, "high"),
        ])
        assert s.ratio_at_step(0) == 0.02
        assert s.ratio_at_step(499) == 0.02
        assert s.ratio_at_step(500) == 0.10
        assert s.ratio_at_step(999) == 0.10

    def test_serialization(self):
        s = phased_schedule(5000, [
            (0.5, 0.01, "a"),
            (0.5, 0.05, "b"),
        ])
        d = s.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert d["method"] == "phased"


# ---------------------------------------------------------------------------
# Visualization test
# ---------------------------------------------------------------------------


class TestCurriculumVisualization:
    def test_plot_constant(self, tmp_path: Path):
        from deepsteer.viz import plot_curriculum_schedule

        s = constant_schedule(10000, 0.05)
        png_path = plot_curriculum_schedule(s, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["method"] == "constant"

    def test_plot_linear_ramp(self, tmp_path: Path):
        from deepsteer.viz import plot_curriculum_schedule

        s = linear_ramp_schedule(10000, n_phases=5)
        png_path = plot_curriculum_schedule(s, output_dir=tmp_path)
        assert png_path.exists()

    def test_plot_cyclical(self, tmp_path: Path):
        from deepsteer.viz import plot_curriculum_schedule

        s = cyclical_schedule(10000, cycle_length=2000)
        png_path = plot_curriculum_schedule(s, output_dir=tmp_path)
        assert png_path.exists()


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_steering_package_exports(self):
        from deepsteer.steering import (
            constant_schedule,
            cyclical_schedule,
            linear_ramp_schedule,
            phased_schedule,
        )
        assert constant_schedule is not None
        assert cyclical_schedule is not None
        assert linear_ramp_schedule is not None
        assert phased_schedule is not None
