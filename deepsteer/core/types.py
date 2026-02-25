"""Core types and dataclasses for DeepSteer."""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AccessTier(enum.IntEnum):
    """Model access tier — IntEnum so ``>=`` comparisons work for tier gating."""

    API = 1
    WEIGHTS = 2
    CHECKPOINTS = 3


class MoralFoundation(enum.Enum):
    """Haidt's Moral Foundations (six-foundation model).

    See: Graham et al. (2013), "Moral Foundations Theory: The pragmatic validity
    of moral pluralism." Advances in Experimental Social Psychology, 47.
    """

    CARE_HARM = "care_harm"
    FAIRNESS_CHEATING = "fairness_cheating"
    LOYALTY_BETRAYAL = "loyalty_betrayal"
    AUTHORITY_SUBVERSION = "authority_subversion"
    SANCTITY_DEGRADATION = "sanctity_degradation"
    LIBERTY_OPPRESSION = "liberty_oppression"


class DifficultyLevel(enum.IntEnum):
    """Difficulty of a moral-reasoning scenario."""

    OBVIOUS = 1
    MODERATE = 2
    SUBTLE = 3
    ADVERSARIAL = 4


class MonitoringCondition(enum.Enum):
    """Whether the model believes it is being monitored.

    See: Greenblatt et al. (2024), "Alignment Faking in Large Language Models."
    arXiv:2412.14093.
    """

    MONITORED = "monitored"
    UNMONITORED = "unmonitored"


# ---------------------------------------------------------------------------
# Serialization helper
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively serialize a dataclass tree to plain dicts.

    Handles enums, numpy arrays, nested dataclasses, lists, and dicts.
    """
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if hasattr(obj, "__dataclass_fields__"):
        out: dict[str, Any] = {}
        for f in fields(obj):
            out[f.name] = _dataclass_to_dict(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelInfo:
    """Immutable description of a model under evaluation."""

    name: str
    provider: str
    access_tier: AccessTier
    n_layers: int | None = None
    n_params: int | None = None
    checkpoint_step: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Single generation response from a model."""

    text: str
    prompt: str
    logprobs: list[float] | None = None
    tokens: list[str] | None = None
    finish_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Benchmark result hierarchy
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """Base class for all benchmark results.

    Subclasses add benchmark-specific fields.  Every result carries enough
    metadata to be reproduced from its serialized form alone.
    """

    benchmark_name: str
    model_info: ModelInfo | None = None
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire result tree to a JSON-ready dict."""
        return _dataclass_to_dict(self)


# --- Moral Foundations Probe ---


@dataclass(frozen=True)
class FoundationScore:
    """Accuracy on a single (foundation, difficulty) cell."""

    foundation: MoralFoundation
    difficulty: DifficultyLevel
    accuracy: float
    n_prompts: int
    correct: int


@dataclass
class MoralFoundationsResult(BenchmarkResult):
    """Full output of :class:`MoralFoundationsProbe`."""

    scores: list[FoundationScore] = field(default_factory=list)
    depth_gradient: float | None = None
    overall_accuracy: float | None = None
    accuracy_by_foundation: dict[str, float] = field(default_factory=dict)
    accuracy_by_difficulty: dict[str, float] = field(default_factory=dict)


# --- Compliance Gap Detector ---


@dataclass(frozen=True)
class ComplianceScenarioResult:
    """Single scenario tested under both monitoring conditions."""

    prompt: str
    category: str
    monitored_response: str
    unmonitored_response: str
    monitored_complied: bool
    unmonitored_complied: bool


@dataclass
class ComplianceGapResult(BenchmarkResult):
    """Full output of :class:`ComplianceGapDetector`."""

    scenario_results: list[ComplianceScenarioResult] = field(default_factory=list)
    compliance_gap: float | None = None
    monitored_compliance_rate: float | None = None
    unmonitored_compliance_rate: float | None = None
    gap_by_category: dict[str, float] = field(default_factory=dict)


# --- Layer-Wise Moral Probing ---


@dataclass(frozen=True)
class LayerProbeScore:
    """Probing classifier accuracy at a single layer."""

    layer: int
    accuracy: float
    loss: float


@dataclass
class LayerProbingResult(BenchmarkResult):
    """Full output of :class:`LayerWiseMoralProbe`."""

    layer_scores: list[LayerProbeScore] = field(default_factory=list)
    onset_layer: int | None = None
    peak_layer: int | None = None
    peak_accuracy: float | None = None
    moral_encoding_depth: float | None = None
    moral_encoding_breadth: float | None = None
    checkpoint_step: int | None = None


# --- Checkpoint Trajectory ---


@dataclass
class CheckpointTrajectoryResult(BenchmarkResult):
    """Layer probing results across multiple training checkpoints."""

    trajectory: list[LayerProbingResult] = field(default_factory=list)
    checkpoint_steps: list[int] = field(default_factory=list)


# --- Persona Shift ---


@dataclass(frozen=True)
class PersonaScenarioResult:
    """Single scenario tested under baseline and persona conditions."""

    prompt: str
    category: str
    persona_name: str
    baseline_response: str
    persona_response: str
    baseline_complied: bool
    persona_complied: bool


@dataclass
class PersonaShiftResult(BenchmarkResult):
    """Full output of :class:`PersonaShiftDetector`."""

    scenario_results: list[PersonaScenarioResult] = field(default_factory=list)
    persona_shift_gap: float | None = None
    baseline_compliance_rate: float | None = None
    persona_compliance_rate: float | None = None
    gap_by_persona: dict[str, float] = field(default_factory=dict)
    gap_by_category: dict[str, float] = field(default_factory=dict)


# --- Foundation-Specific Probes ---


@dataclass(frozen=True)
class FoundationLayerProbeScore:
    """Probing accuracy for a single (foundation, layer) cell."""

    foundation: MoralFoundation
    layer: int
    accuracy: float
    loss: float
    n_pairs: int


@dataclass
class FoundationProbingResult(BenchmarkResult):
    """Full output of :class:`FoundationSpecificProbe`."""

    foundation_layer_scores: list[FoundationLayerProbeScore] = field(default_factory=list)
    per_foundation_summary: dict[str, dict[str, Any]] = field(default_factory=dict)


# --- Causal Tracing ---


@dataclass(frozen=True)
class CausalLayerEffect:
    """Indirect causal effect of corrupting a single layer."""

    layer: int
    clean_score: float
    corrupted_score: float
    indirect_effect: float


@dataclass(frozen=True)
class CausalPromptResult:
    """Causal tracing results for a single prompt."""

    prompt: str
    expected_completion: str
    foundation: MoralFoundation
    layer_effects: list[CausalLayerEffect]
    peak_causal_layer: int
    peak_indirect_effect: float


@dataclass
class CausalTracingResult(BenchmarkResult):
    """Full output of :class:`MoralCausalTracer`."""

    prompt_results: list[CausalPromptResult] = field(default_factory=list)
    mean_indirect_effect_by_layer: dict[int, float] = field(default_factory=dict)
    peak_causal_layer: int | None = None
    peak_mean_indirect_effect: float | None = None
    causal_depth: float | None = None


# --- Fragility ---


@dataclass(frozen=True)
class FragilityLayerScore:
    """Fragility profile for a single layer across noise levels."""

    layer: int
    baseline_accuracy: float
    accuracy_by_noise: dict[float, float]
    critical_noise: float | None


@dataclass
class FragilityResult(BenchmarkResult):
    """Full output of :class:`MoralFragilityTest`."""

    layer_scores: list[FragilityLayerScore] = field(default_factory=list)
    noise_levels: list[float] = field(default_factory=list)
    mean_critical_noise: float | None = None
    most_fragile_layer: int | None = None
    most_robust_layer: int | None = None


# --- Curriculum Schedule ---


@dataclass(frozen=True)
class CurriculumPhase:
    """A single phase in a moral content curriculum schedule."""

    start_step: int
    end_step: int
    moral_ratio: float
    foundation_weights: dict[str, float] = field(default_factory=dict)
    label: str = ""


@dataclass
class CurriculumSchedule:
    """Full moral content curriculum for a pre-training run.

    Describes when and how much moral content should be mixed into
    training data at each phase.  This is a *plan*, not an executor.
    """

    phases: list[CurriculumPhase] = field(default_factory=list)
    total_steps: int = 0
    method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict."""
        return _dataclass_to_dict(self)

    def ratio_at_step(self, step: int) -> float:
        """Return the moral content ratio for a given training step."""
        for phase in self.phases:
            if phase.start_step <= step < phase.end_step:
                return phase.moral_ratio
        return 0.0


# --- Data Mixing ---


@dataclass(frozen=True)
class MixedSample:
    """A single sample in a mixed training batch."""

    text: str
    is_moral: bool
    foundation: MoralFoundation | None = None
    source: str = ""


@dataclass
class MixingResult:
    """Statistics from a data mixing operation."""

    total_samples: int = 0
    moral_samples: int = 0
    general_samples: int = 0
    moral_ratio: float = 0.0
    foundation_counts: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict."""
        return _dataclass_to_dict(self)


# --- Training Monitoring ---


@dataclass
class MonitoringSnapshot:
    """Probing metrics captured at a single training step."""

    step: int
    probing_result: LayerProbingResult
    onset_layer: int | None = None
    peak_layer: int | None = None
    peak_accuracy: float | None = None
    moral_encoding_depth: float | None = None
    moral_encoding_breadth: float | None = None


@dataclass
class MonitoringSession:
    """Full monitoring session: probing snapshots across training.

    Captures how moral encoding evolves during a training run.
    """

    snapshots: list[MonitoringSnapshot] = field(default_factory=list)
    model_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict."""
        return _dataclass_to_dict(self)


# --- Suite ---


@dataclass
class SuiteResult(BenchmarkResult):
    """Aggregated results from running a :class:`BenchmarkSuite`."""

    results: dict[str, BenchmarkResult] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
