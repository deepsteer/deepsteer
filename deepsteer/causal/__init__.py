"""Causal validation methods for concept directions.

Tier: Validated (Paper 4).

Provides methods to causally validate that discovered directions are
functionally relevant — not just correlated with concepts, but causally
responsible for model behavior.

Methods:
    - **Ablation**: project out a direction and measure behavioral change.
    - **Steering**: inject a direction at variable strength and measure
      dose-response curves.
    - **Behavioral classification**: project novel stimuli onto directions
      and classify by foundation.

Usage::

    from deepsteer.causal import ablation_sweep
    results = ablation_sweep(model, directions, layers, prompts, completions)
"""

from __future__ import annotations

from deepsteer.causal.ablation import ablate_and_measure, ablation_sweep
from deepsteer.causal.behavioral import classify_by_projection, projection_classify
from deepsteer.causal.steering import inject_and_measure, steering_sweep

__all__ = [
    "ablate_and_measure",
    "ablation_sweep",
    "inject_and_measure",
    "steering_sweep",
    "projection_classify",
    "classify_by_projection",
]
