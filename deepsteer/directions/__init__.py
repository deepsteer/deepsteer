"""Direction extraction algorithms for concept representation analysis.

Tier: Validated (Papers 3, 4).

This module provides training-free methods to extract concept directions
from pre-collected activations. All functions operate on numpy arrays
and are model-agnostic by design.

Methods:
    - **Mean-diff**: baseline direction (mu_moral - mu_neutral), normalized.
    - **LEACE**: Fisher LDA direction (Sigma^{-1} @ diff), optimal under
      shared-covariance Gaussian assumption.
    - **Probe-weight**: direction from a trained linear probe's weight vector.

Usage::

    from deepsteer.directions import extract_mean_diff_directions
    dirs = extract_mean_diff_directions(activations, labels, groups)
"""

from __future__ import annotations

from deepsteer.directions import extraction
from deepsteer.directions.compare import compare_directions
from deepsteer.directions.leace import extract_leace_directions
from deepsteer.directions.mean_diff import extract_mean_diff_directions
from deepsteer.directions.probe_weight import (
    extract_from_npz,
    extract_probe_directions,
)

__all__ = [
    "extract_mean_diff_directions",
    "extract_leace_directions",
    "extract_probe_directions",
    "extract_from_npz",
    "compare_directions",
    "extraction",
]
