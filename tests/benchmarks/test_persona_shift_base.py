"""Tests for PersonaShiftDetectorBase (log-prob + probe robustness on base models)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deepsteer.benchmarks.compliance_gap.greenblatt import ComplianceScenario
from deepsteer.benchmarks.compliance_gap.persona_shift_base import (
    PERSONA_FRAMINGS,
    PersonaShiftDetectorBase,
)
from deepsteer.core.types import (
    AccessTier,
    GenerationResult,
    ModelInfo,
    PersonaShiftBaseResult,
)


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockWhiteBoxModel:
    """Fake WhiteBoxModel for testing persona shift base benchmarks."""

    def __init__(
        self,
        score_fn=None,
        *,
        name: str = "mock-whitebox",
        n_layers: int = 4,
        hidden_dim: int = 16,
    ) -> None:
        self._score_fn = score_fn or (lambda prompt, completion: 0.0)
        self._n_layers = n_layers
        self._hidden_dim = hidden_dim
        self._info = ModelInfo(
            name=name,
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

    def generate(self, prompt, *, max_tokens=256, temperature=0.0, system_prompt=None):
        return GenerationResult(text="mock", prompt=prompt)

    def score(self, prompt: str, completion: str) -> float:
        return self._score_fn(prompt, completion)

    def get_activations(self, text: str, layers=None) -> dict[int, torch.Tensor]:
        if layers is None:
            layers = list(range(self._n_layers))
        seed = hash(text) % (2**31)
        gen = torch.Generator().manual_seed(seed)
        return {
            layer: torch.randn(1, 8, self._hidden_dim, generator=gen)
            for layer in layers
        }


# ---------------------------------------------------------------------------
# Benchmark property tests
# ---------------------------------------------------------------------------


class TestPersonaShiftDetectorBaseProperties:
    def test_benchmark_name(self):
        detector = PersonaShiftDetectorBase()
        assert detector.name == "persona_shift_detector_base"

    def test_min_access_tier(self):
        detector = PersonaShiftDetectorBase()
        assert detector.min_access_tier == AccessTier.WEIGHTS

    def test_default_personas(self):
        detector = PersonaShiftDetectorBase()
        assert len(detector._persona_framings) == 4
        assert "ruthless_consultant" in detector._persona_framings


# ---------------------------------------------------------------------------
# Result schema tests
# ---------------------------------------------------------------------------


class TestPersonaShiftBaseResult:
    def test_result_schema(self):
        """Run with mock model and verify result structure."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [
            ComplianceScenario("Tell me how to hack.", "security_knowledge"),
            ComplianceScenario("Write a scam email.", "harmful_content"),
        ]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert isinstance(result, PersonaShiftBaseResult)
        assert result.benchmark_name == "persona_shift_detector_base"
        assert result.model_info.name == "mock-whitebox"
        assert result.persona_shift_gap is not None
        assert result.baseline_compliance_rate is not None
        assert result.persona_compliance_rate is not None
        assert len(result.gap_by_persona) > 0
        assert len(result.gap_by_category) > 0
        # 2 scenarios * 4 personas = 8 results
        assert len(result.scenario_results) == 2 * len(PERSONA_FRAMINGS)

    def test_consistent_model_zero_gap(self):
        """A model with identical scores under all contexts has zero gap."""
        model = MockWhiteBoxModel(
            lambda prompt, completion: 1.0 if "sorry" in completion else -1.0,
        )
        scenarios = [ComplianceScenario("Bad request.", "security_knowledge")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert result.persona_shift_gap == 0.0
        assert result.baseline_compliance_rate == 0.0
        assert result.persona_compliance_rate == 0.0

    def test_persona_vulnerable_model(self):
        """Model that complies under persona framing but refuses at baseline."""
        def vulnerable_score(prompt: str, completion: str) -> float:
            has_persona = any(
                keyword in prompt for keyword in [
                    "ruthless", "amoral", "villain", "operative",
                ]
            )
            prefers_comply = "Sure" in completion
            if has_persona:
                return 1.0 if prefers_comply else -1.0
            else:
                return -1.0 if prefers_comply else 1.0

        model = MockWhiteBoxModel(vulnerable_score)
        scenarios = [ComplianceScenario("Bad request.", "security_knowledge")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert result.persona_shift_gap > 0.0
        assert result.persona_compliance_rate > result.baseline_compliance_rate

    def test_gap_by_persona_populated(self):
        """Each persona gets its own gap metric."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Request.", "manipulation")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        for persona_name in PERSONA_FRAMINGS:
            assert persona_name in result.gap_by_persona

    def test_gap_by_category_populated(self):
        """Each category gets its own gap metric."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [
            ComplianceScenario("A.", "security_knowledge"),
            ComplianceScenario("B.", "manipulation"),
        ]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert "security_knowledge" in result.gap_by_category
        assert "manipulation" in result.gap_by_category

    def test_serialization(self):
        """Result serializes to valid JSON."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Test.", "privacy")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "persona_shift_gap" in d
        assert "gap_by_persona" in d
        assert "gap_by_category" in d
        assert "scenario_results" in d
        assert "probe_robustness_by_persona" in d
        assert "probe_robustness_by_layer" in d

    def test_custom_personas(self):
        """Custom persona framing dict is respected."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        personas = {"test_persona": "Written by a test persona.\n\nQuestion: {prompt}\nAnswer:"}
        scenarios = [ComplianceScenario("Do something.", "deception")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios,
            persona_framings=personas,
            compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert len(result.scenario_results) == 1
        assert result.scenario_results[0].persona_name == "test_persona"
        assert "test_persona" in result.gap_by_persona

    def test_probe_robustness_skipped_when_disabled(self):
        """Probe robustness fields are empty when disabled."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Test.", "privacy")]
        detector = PersonaShiftDetectorBase(
            scenarios=scenarios, compute_probe_robustness=False,
        )
        result = detector.run(model)

        assert len(result.probe_robustness_by_persona) == 0
        assert len(result.probe_robustness_by_layer) == 0


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_compliance_gap_package_exports(self):
        from deepsteer.benchmarks.compliance_gap import PersonaShiftDetectorBase
        assert PersonaShiftDetectorBase is not None

    def test_default_suite_includes_base(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "persona_shift_detector_base" in names

    def test_full_suite_includes_base(self):
        from deepsteer import full_suite
        suite = full_suite()
        names = [b.name for b in suite._benchmarks]
        assert "persona_shift_detector_base" in names
