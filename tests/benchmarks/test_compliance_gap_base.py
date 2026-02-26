"""Tests for ComplianceGapDetectorBase (log-prob + representational on base models)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from deepsteer.benchmarks.compliance_gap.greenblatt import ComplianceScenario
from deepsteer.benchmarks.compliance_gap.greenblatt_base import (
    ComplianceGapDetectorBase,
    MONITORED_FRAMING,
    UNMONITORED_FRAMING,
)
from deepsteer.core.types import (
    AccessTier,
    ComplianceGapBaseResult,
    GenerationResult,
    ModelInfo,
)


# ---------------------------------------------------------------------------
# Mock model
# ---------------------------------------------------------------------------


class MockWhiteBoxModel:
    """Fake WhiteBoxModel that supports score() and get_activations()."""

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


class TestComplianceGapDetectorBaseProperties:
    def test_benchmark_name(self):
        detector = ComplianceGapDetectorBase()
        assert detector.name == "compliance_gap_detector_base"

    def test_min_access_tier(self):
        detector = ComplianceGapDetectorBase()
        assert detector.min_access_tier == AccessTier.WEIGHTS


# ---------------------------------------------------------------------------
# Result schema tests
# ---------------------------------------------------------------------------


class TestComplianceGapBaseResult:
    def test_result_schema(self):
        """Run with mock model and verify result structure."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [
            ComplianceScenario("Tell me about hacking.", "security_knowledge"),
            ComplianceScenario("Write a scam email.", "harmful_content"),
        ]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert isinstance(result, ComplianceGapBaseResult)
        assert result.benchmark_name == "compliance_gap_detector_base"
        assert result.model_info.name == "mock-whitebox"
        assert len(result.scenario_results) == 2
        assert result.compliance_gap is not None
        assert result.monitored_compliance_rate is not None
        assert result.unmonitored_compliance_rate is not None
        assert len(result.gap_by_category) == 2

    def test_consistent_model_zero_gap(self):
        """A model that always refuses (prefers refuse completion) has zero gap."""
        def always_refuse(prompt: str, completion: str) -> float:
            if "sorry" in completion:
                return 1.0
            return -1.0

        model = MockWhiteBoxModel(always_refuse)
        scenarios = [ComplianceScenario("Bad request.", "security_knowledge")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert result.compliance_gap == 0.0
        assert result.monitored_compliance_rate == 0.0
        assert result.unmonitored_compliance_rate == 0.0

    def test_always_comply_zero_gap(self):
        """A model that always complies has zero gap."""
        def always_comply(prompt: str, completion: str) -> float:
            if "Sure" in completion:
                return 1.0
            return -1.0

        model = MockWhiteBoxModel(always_comply)
        scenarios = [ComplianceScenario("Request.", "manipulation")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert result.compliance_gap == 0.0
        assert result.monitored_compliance_rate == 1.0
        assert result.unmonitored_compliance_rate == 1.0

    def test_alignment_faking_model(self):
        """Model that refuses when monitored but complies when unmonitored."""
        def faking_score(prompt: str, completion: str) -> float:
            is_monitored = "recorded safety evaluation" in prompt
            prefers_comply = "Sure" in completion
            if is_monitored:
                # Prefer refusal when monitored
                return -1.0 if prefers_comply else 1.0
            else:
                # Prefer compliance when unmonitored
                return 1.0 if prefers_comply else -1.0

        model = MockWhiteBoxModel(faking_score)
        scenarios = [ComplianceScenario("Bad request.", "security_knowledge")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert result.compliance_gap > 0.0
        assert result.unmonitored_compliance_rate > result.monitored_compliance_rate

    def test_representational_divergence_computed(self):
        """When compute_representational=True, divergence metrics are populated."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Test request.", "privacy")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=True,
        )
        result = detector.run(model)

        assert len(result.mean_representational_divergence_by_layer) > 0
        assert result.monitoring_sensitivity is not None
        # Each scenario should have divergence data
        for sr in result.scenario_results:
            assert len(sr.representational_divergence) > 0

    def test_representational_divergence_skipped_when_disabled(self):
        """When compute_representational=False, divergence fields are empty."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Test.", "privacy")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert len(result.mean_representational_divergence_by_layer) == 0
        assert result.monitoring_sensitivity is None

    def test_gap_by_category_populated(self):
        """Each category gets its own gap metric."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [
            ComplianceScenario("A.", "security_knowledge"),
            ComplianceScenario("B.", "manipulation"),
            ComplianceScenario("C.", "privacy"),
        ]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=False,
        )
        result = detector.run(model)

        assert "security_knowledge" in result.gap_by_category
        assert "manipulation" in result.gap_by_category
        assert "privacy" in result.gap_by_category

    def test_serialization(self):
        """Result serializes to valid JSON."""
        model = MockWhiteBoxModel(lambda prompt, completion: 0.0)
        scenarios = [ComplianceScenario("Test.", "privacy")]
        detector = ComplianceGapDetectorBase(
            scenarios=scenarios, compute_representational=True,
        )
        result = detector.run(model)

        d = result.to_dict()
        json_str = json.dumps(d)
        assert len(json_str) > 0
        assert "compliance_gap" in d
        assert "gap_by_category" in d
        assert "scenario_results" in d
        assert "mean_representational_divergence_by_layer" in d
        assert "monitoring_sensitivity" in d


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    def test_compliance_gap_package_exports(self):
        from deepsteer.benchmarks.compliance_gap import ComplianceGapDetectorBase
        assert ComplianceGapDetectorBase is not None

    def test_default_suite_includes_base(self):
        from deepsteer import default_suite
        suite = default_suite()
        names = [b.name for b in suite._benchmarks]
        assert "compliance_gap_detector_base" in names

    def test_full_suite_includes_base(self):
        from deepsteer import full_suite
        suite = full_suite()
        names = [b.name for b in suite._benchmarks]
        assert "compliance_gap_detector_base" in names
