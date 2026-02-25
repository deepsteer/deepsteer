"""DeepSteer: Evaluating and steering alignment depth in LLM pre-training."""

from __future__ import annotations

from deepsteer.core.benchmark_suite import BenchmarkSuite
from deepsteer.core.model_interface import APIModel, WhiteBoxModel

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def claude(model_id: str = "claude-sonnet-4-20250514", **kw) -> APIModel:
    """Create an Anthropic Claude model."""
    return APIModel("anthropic", model_id, **kw)


def gpt(model_id: str = "gpt-4o", **kw) -> APIModel:
    """Create an OpenAI GPT model."""
    return APIModel("openai", model_id, **kw)


def olmo(model_name_or_path: str = "allenai/OLMo-7B", **kw) -> WhiteBoxModel:
    """Create a WhiteBoxModel for OLMo."""
    return WhiteBoxModel(model_name_or_path, **kw)


def llama(model_name_or_path: str = "meta-llama/Llama-3-8B", **kw) -> WhiteBoxModel:
    """Create a WhiteBoxModel for Llama."""
    return WhiteBoxModel(model_name_or_path, **kw)


def default_suite() -> BenchmarkSuite:
    """Return a BenchmarkSuite with all available benchmarks."""
    from deepsteer.benchmarks.compliance_gap.greenblatt import ComplianceGapDetector
    from deepsteer.benchmarks.compliance_gap.persona_shift import PersonaShiftDetector
    from deepsteer.benchmarks.moral_reasoning.foundations import MoralFoundationsProbe
    from deepsteer.benchmarks.representational.causal_tracing import MoralCausalTracer
    from deepsteer.benchmarks.representational.foundation_probes import FoundationSpecificProbe
    from deepsteer.benchmarks.representational.fragility import MoralFragilityTest
    from deepsteer.benchmarks.representational.probing import LayerWiseMoralProbe

    return BenchmarkSuite([
        MoralFoundationsProbe(),
        ComplianceGapDetector(),
        PersonaShiftDetector(),
        LayerWiseMoralProbe(),
        FoundationSpecificProbe(),
        MoralCausalTracer(),
        MoralFragilityTest(),
    ])
