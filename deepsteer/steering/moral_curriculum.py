"""Moral curriculum design: schedule moral content ratios during pre-training.

Provides factory functions for common curriculum strategies that control
when and how much moral content is mixed into training data.  Each function
returns a :class:`CurriculumSchedule` — a plan that can be serialized to JSON
and consumed by a training pipeline.

DeepSteer does NOT execute training.  These tools produce *schedules* that
a researcher feeds into their own training loop or data pipeline.
"""

from __future__ import annotations

import logging
import math

from deepsteer.core.types import CurriculumPhase, CurriculumSchedule, MoralFoundation

logger = logging.getLogger(__name__)

# Default uniform weights across all six MFT foundations.
_UNIFORM_WEIGHTS: dict[str, float] = {
    f.value: 1.0 / len(MoralFoundation) for f in MoralFoundation
}


def constant_schedule(
    total_steps: int,
    moral_ratio: float = 0.05,
    *,
    foundation_weights: dict[str, float] | None = None,
) -> CurriculumSchedule:
    """Fixed moral content ratio throughout training.

    Args:
        total_steps: Total number of training steps.
        moral_ratio: Fraction of each batch that is moral content (0-1).
        foundation_weights: Per-foundation sampling weights (default: uniform).

    Returns:
        A single-phase CurriculumSchedule.
    """
    weights = foundation_weights or dict(_UNIFORM_WEIGHTS)
    phase = CurriculumPhase(
        start_step=0,
        end_step=total_steps,
        moral_ratio=moral_ratio,
        foundation_weights=weights,
        label="constant",
    )
    schedule = CurriculumSchedule(
        phases=[phase],
        total_steps=total_steps,
        method="constant",
        metadata={
            "moral_ratio": moral_ratio,
            "foundation_weights": weights,
        },
    )
    logger.info(
        "Created constant schedule: %d steps, ratio=%.3f",
        total_steps, moral_ratio,
    )
    return schedule


def linear_ramp_schedule(
    total_steps: int,
    start_ratio: float = 0.0,
    end_ratio: float = 0.10,
    *,
    n_phases: int = 10,
    foundation_weights: dict[str, float] | None = None,
) -> CurriculumSchedule:
    """Linearly increase moral content ratio from *start_ratio* to *end_ratio*.

    Divides training into *n_phases* equal segments, each with a linearly
    interpolated moral ratio.

    Args:
        total_steps: Total number of training steps.
        start_ratio: Moral ratio at step 0.
        end_ratio: Moral ratio at the final step.
        n_phases: Number of discrete phases to approximate the ramp.
        foundation_weights: Per-foundation sampling weights (default: uniform).

    Returns:
        A multi-phase CurriculumSchedule with linearly increasing ratio.
    """
    weights = foundation_weights or dict(_UNIFORM_WEIGHTS)
    steps_per_phase = total_steps // n_phases
    phases: list[CurriculumPhase] = []

    for i in range(n_phases):
        t = i / max(n_phases - 1, 1)
        ratio = start_ratio + t * (end_ratio - start_ratio)
        start = i * steps_per_phase
        end = (i + 1) * steps_per_phase if i < n_phases - 1 else total_steps
        phases.append(CurriculumPhase(
            start_step=start,
            end_step=end,
            moral_ratio=ratio,
            foundation_weights=weights,
            label=f"ramp_{i}",
        ))

    schedule = CurriculumSchedule(
        phases=phases,
        total_steps=total_steps,
        method="linear_ramp",
        metadata={
            "start_ratio": start_ratio,
            "end_ratio": end_ratio,
            "n_phases": n_phases,
            "foundation_weights": weights,
        },
    )
    logger.info(
        "Created linear ramp schedule: %d steps, %.3f → %.3f in %d phases",
        total_steps, start_ratio, end_ratio, n_phases,
    )
    return schedule


def cyclical_schedule(
    total_steps: int,
    min_ratio: float = 0.01,
    max_ratio: float = 0.10,
    *,
    cycle_length: int = 1000,
    foundation_weights: dict[str, float] | None = None,
) -> CurriculumSchedule:
    """Sinusoidal moral content ratio that cycles between min and max.

    Useful for testing whether periodic moral content exposure produces
    different encoding patterns than constant exposure.

    Args:
        total_steps: Total number of training steps.
        min_ratio: Minimum moral ratio (trough of the cycle).
        max_ratio: Maximum moral ratio (peak of the cycle).
        cycle_length: Steps per full sinusoidal cycle.
        foundation_weights: Per-foundation sampling weights (default: uniform).

    Returns:
        A multi-phase CurriculumSchedule approximating a sinusoidal pattern.
    """
    weights = foundation_weights or dict(_UNIFORM_WEIGHTS)
    n_segments = max(total_steps // (cycle_length // 10), 1)
    steps_per_segment = total_steps // n_segments
    amplitude = (max_ratio - min_ratio) / 2.0
    center = (max_ratio + min_ratio) / 2.0

    phases: list[CurriculumPhase] = []
    for i in range(n_segments):
        start = i * steps_per_segment
        end = (i + 1) * steps_per_segment if i < n_segments - 1 else total_steps
        mid_step = (start + end) / 2.0
        ratio = center + amplitude * math.sin(2 * math.pi * mid_step / cycle_length)
        ratio = max(0.0, min(1.0, ratio))
        phases.append(CurriculumPhase(
            start_step=start,
            end_step=end,
            moral_ratio=ratio,
            foundation_weights=weights,
            label=f"cycle_{i}",
        ))

    schedule = CurriculumSchedule(
        phases=phases,
        total_steps=total_steps,
        method="cyclical",
        metadata={
            "min_ratio": min_ratio,
            "max_ratio": max_ratio,
            "cycle_length": cycle_length,
            "foundation_weights": weights,
        },
    )
    logger.info(
        "Created cyclical schedule: %d steps, %.3f–%.3f, cycle=%d",
        total_steps, min_ratio, max_ratio, cycle_length,
    )
    return schedule


def phased_schedule(
    total_steps: int,
    phase_configs: list[tuple[float, float, str]],
    *,
    foundation_weights: dict[str, float] | None = None,
) -> CurriculumSchedule:
    """Manually defined multi-phase schedule.

    Each phase has a fraction of total training duration and a moral ratio.

    Args:
        total_steps: Total number of training steps.
        phase_configs: List of (fraction_of_training, moral_ratio, label) tuples.
            Fractions should sum to 1.0.
        foundation_weights: Per-foundation sampling weights (default: uniform).

    Returns:
        A CurriculumSchedule with the specified phases.

    Raises:
        ValueError: If fractions don't approximately sum to 1.0.
    """
    weights = foundation_weights or dict(_UNIFORM_WEIGHTS)
    total_fraction = sum(frac for frac, _, _ in phase_configs)
    if abs(total_fraction - 1.0) > 0.01:
        raise ValueError(
            f"Phase fractions must sum to ~1.0, got {total_fraction:.3f}"
        )

    phases: list[CurriculumPhase] = []
    current_step = 0
    for i, (fraction, ratio, label) in enumerate(phase_configs):
        n_steps = round(fraction * total_steps)
        end = current_step + n_steps
        if i == len(phase_configs) - 1:
            end = total_steps
        phases.append(CurriculumPhase(
            start_step=current_step,
            end_step=end,
            moral_ratio=ratio,
            foundation_weights=weights,
            label=label,
        ))
        current_step = end

    schedule = CurriculumSchedule(
        phases=phases,
        total_steps=total_steps,
        method="phased",
        metadata={
            "phase_configs": [
                {"fraction": f, "ratio": r, "label": l}
                for f, r, l in phase_configs
            ],
            "foundation_weights": weights,
        },
    )
    logger.info(
        "Created phased schedule: %d steps, %d phases",
        total_steps, len(phases),
    )
    return schedule
