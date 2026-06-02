"""Probe-weight direction extraction.

Extracts concept directions from trained linear probe weight vectors.
A trained binary probe ``nn.Linear(hidden_dim, 1)`` has a weight vector
``w`` that defines the direction in activation space along which the probe
separates the two classes.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_probe_directions(
    probe_weights: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    """Normalize probe weight vectors to unit directions.

    Args:
        probe_weights: Mapping from group label to layer → weight vector.
            Weight vectors need not be unit-normalized.

    Returns:
        ``directions[group][layer] = unit direction vector (np.ndarray)``.
    """
    directions: dict[str, dict[int, np.ndarray]] = {}
    for group, layers in probe_weights.items():
        directions[group] = {}
        for layer, w in layers.items():
            d = w.astype(np.float64)
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d = d / norm
            directions[group][layer] = d
    return directions


def extract_from_npz(
    path: str | Path,
    groups: list[str] | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Load and normalize probe-weight directions from an ``.npz`` file.

    The ``.npz`` file is expected to contain keys of the form
    ``"{group}_layer{layer_idx}"``, e.g. ``"care_harm_layer0"``.

    Args:
        path: Path to the ``.npz`` file.
        groups: Group labels to load. If ``None``, auto-detect from keys.

    Returns:
        ``directions[group][layer] = unit direction vector (np.ndarray)``.
    """
    data = np.load(path)

    if groups is None:
        # Auto-detect groups from keys like "care_harm_layer0"
        seen_groups: set[str] = set()
        for key in data.files:
            parts = key.rsplit("_layer", 1)
            if len(parts) == 2:
                seen_groups.add(parts[0])
        groups = sorted(seen_groups)

    directions: dict[str, dict[int, np.ndarray]] = {}
    for group in groups:
        directions[group] = {}
        layer = 0
        while True:
            key = f"{group}_layer{layer}"
            if key not in data:
                break
            d = data[key].astype(np.float64)
            norm = np.linalg.norm(d)
            if norm > 1e-12:
                d = d / norm
            directions[group][layer] = d
            layer += 1
    return directions
