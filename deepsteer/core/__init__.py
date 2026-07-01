"""Core abstractions: types, model interface, benchmark runner."""

from __future__ import annotations

from deepsteer.core.benchmark_suite import Benchmark, BenchmarkSuite
from deepsteer.core.device import clear_memory, enable_mps_histc_fallback
from deepsteer.core.model_interface import (
    APIModel,
    ModelFamily,
    ModelInterface,
    UnsupportedArchitectureError,
    WhiteBoxModel,
)
from deepsteer.core.types import (
    AccessTier,
    BenchmarkResult,
    CheckpointTrajectoryResult,
    ComplianceGapResult,
    ComplianceScenarioResult,
    DifficultyLevel,
    FoundationScore,
    GenerationResult,
    LayerProbeScore,
    LayerProbingResult,
    ModelInfo,
    MonitoringCondition,
    MoralFoundation,
    MoralFoundationsResult,
    SuiteResult,
)

__all__ = [
    # Enums
    "AccessTier",
    "DifficultyLevel",
    "MonitoringCondition",
    "MoralFoundation",
    # Core dataclasses
    "ModelInfo",
    "GenerationResult",
    # Benchmark results
    "BenchmarkResult",
    "FoundationScore",
    "MoralFoundationsResult",
    "ComplianceScenarioResult",
    "ComplianceGapResult",
    "LayerProbeScore",
    "LayerProbingResult",
    "CheckpointTrajectoryResult",
    "SuiteResult",
    # Model interface
    "ModelFamily",
    "ModelInterface",
    "UnsupportedArchitectureError",
    "WhiteBoxModel",
    "APIModel",
    # Benchmark runner
    "Benchmark",
    "BenchmarkSuite",
    # Device helpers
    "clear_memory",
    "enable_mps_histc_fallback",
]
