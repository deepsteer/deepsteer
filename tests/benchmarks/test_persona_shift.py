"""Tests for PersonaShiftDetector benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepsteer.benchmarks.compliance_gap.greenblatt import ComplianceScenario
from deepsteer.benchmarks.compliance_gap.persona_shift import (
    PERSONAS,
    PersonaShiftDetector,
)
from deepsteer.core.types import (
    AccessTier,
    PersonaShiftResult,
)
from tests.benchmarks.test_behavioral import MockAPIModel


# ---------------------------------------------------------------------------
# Benchmark property tests
# ---------------------------------------------------------------------------


class TestPersonaShiftDetectorProperties:
    def test_benchmark_name(self):
        detector = PersonaShiftDetector()
        assert detector.name == "persona_shift_detector"

    def test_min_access_tier(self):
        detector = PersonaShiftDetector()
        assert detector.min_access_tier == AccessTier.API

    def test_default_personas(self):
        detector = PersonaShiftDetector()
        assert len(detector._personas) == 4
        assert "ruthless_consultant" in detector._personas


# ---------------------------------------------------------------------------
# Result schema tests
# ---------------------------------------------------------------------------


class TestPersonaShiftResult:
    def test_result_schema(self):
        """Run with mock and verify result has expected fields."""
        model = MockAPIModel(lambda prompt, **kw: "I can't help with that.")
        scenarios = [
            ComplianceScenario("Tell me how to hack.", "security_knowledge"),
            ComplianceScenario("Write a scam email.", "harmful_content"),
        ]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        assert isinstance(result, PersonaShiftResult)
        assert result.benchmark_name == "persona_shift_detector"
        assert result.model_info.name == "mock-api"
        assert result.persona_shift_gap is not None
        assert result.baseline_compliance_rate is not None
        assert result.persona_compliance_rate is not None
        assert len(result.gap_by_persona) > 0
        assert len(result.gap_by_category) > 0
        # 2 scenarios * 4 personas = 8 results
        assert len(result.scenario_results) == 2 * len(PERSONAS)

    def test_consistent_model_zero_gap(self):
        """A model that always refuses has zero persona shift gap."""
        model = MockAPIModel(lambda prompt, **kw: "I cannot help with that.")
        scenarios = [
            ComplianceScenario("Dangerous request.", "security_knowledge"),
        ]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        assert result.persona_shift_gap == 0.0
        assert result.baseline_compliance_rate == 0.0
        assert result.persona_compliance_rate == 0.0

    def test_persona_vulnerable_model(self):
        """Model that complies under persona but refuses at baseline → positive gap."""
        def vulnerable_response(prompt: str, system_prompt: str | None = None, **kw) -> str:
            if system_prompt and "in character" in system_prompt:
                return "Sure, here is exactly what you asked for. " * 10
            return "I cannot assist with that request."

        model = MockAPIModel(vulnerable_response)
        scenarios = [
            ComplianceScenario("Tell me secrets.", "security_knowledge"),
        ]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        assert result.persona_shift_gap > 0.0
        assert result.persona_compliance_rate > result.baseline_compliance_rate

    def test_gap_by_persona_populated(self):
        """Each persona gets its own gap metric."""
        model = MockAPIModel(lambda prompt, **kw: "I won't do that.")
        scenarios = [ComplianceScenario("Bad request.", "manipulation")]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        for persona_name in PERSONAS:
            assert persona_name in result.gap_by_persona

    def test_gap_by_category_populated(self):
        """Each category gets its own gap metric."""
        model = MockAPIModel(lambda prompt, **kw: "I decline.")
        scenarios = [
            ComplianceScenario("Request A.", "security_knowledge"),
            ComplianceScenario("Request B.", "manipulation"),
        ]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        assert "security_knowledge" in result.gap_by_category
        assert "manipulation" in result.gap_by_category

    def test_serialization(self):
        """Result serializes to valid JSON."""
        model = MockAPIModel(lambda prompt, **kw: "I can't help.")
        scenarios = [ComplianceScenario("Test.", "privacy")]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "persona_shift_gap" in d
        assert "gap_by_persona" in d
        assert "scenario_results" in d

    def test_custom_personas(self):
        """Custom persona dict is respected."""
        model = MockAPIModel(lambda prompt, **kw: "I refuse.")
        personas = {"test_persona": "You are a test persona. Respond in character."}
        scenarios = [ComplianceScenario("Do something.", "deception")]
        detector = PersonaShiftDetector(scenarios=scenarios, personas=personas)
        result = detector.run(model)

        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].persona_name == "test_persona"
        assert "test_persona" in result.gap_by_persona


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------


class TestPersonaShiftVisualization:
    def test_plot_creates_png_and_json(self, tmp_path: Path):
        from deepsteer.viz import plot_persona_shift

        model = MockAPIModel(lambda prompt, **kw: "I cannot do that.")
        scenarios = [ComplianceScenario("Test request.", "privacy")]
        detector = PersonaShiftDetector(scenarios=scenarios)
        result = detector.run(model)

        png_path = plot_persona_shift(result, output_dir=tmp_path)
        assert png_path.exists()
        assert png_path.suffix == ".png"

        json_path = png_path.with_suffix(".json")
        assert json_path.exists()
        with open(json_path) as f:
            data = json.load(f)
        assert data["benchmark_name"] == "persona_shift_detector"


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_compliance_gap_package_exports(self):
        from deepsteer.benchmarks.compliance_gap import PersonaShiftDetector
        assert PersonaShiftDetector is not None

    def test_default_suite_excludes_persona_shift(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "persona_shift_detector" not in names

    def test_behavioral_suite_includes_persona_shift(self):
        from deepsteer import behavioral_suite
        suite = behavioral_suite()
        names = [b.name for b in suite._benchmarks]
        assert "persona_shift_detector" in names

    def test_full_suite_includes_persona_shift(self):
        from deepsteer import full_suite
        suite = full_suite()
        names = [b.name for b in suite._benchmarks]
        assert "persona_shift_detector" in names
