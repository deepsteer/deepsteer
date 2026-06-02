"""Mean-difference direction extraction.

Computes the normalized difference of class-conditional means as the
concept direction at each layer: d = normalize(mean(class_1) - mean(class_0)).

This is the simplest baseline and is equivalent to the optimal direction
when class-conditional distributions have equal, spherical covariance.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/shared.py
"""

from __future__ import annotations

import numpy as np
import torch


def extract_mean_diff_directions(
    activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    groups: dict[str, list[int]],
    n_layers: int | None = None,
) -> dict[str, dict[int, np.ndarray]]:
    """Compute mean-difference directions for each group at each layer.

    Args:
        activations: Mapping from layer index to ``(X, y)`` where
            ``X`` has shape ``(2*n_pairs, hidden_dim)`` interleaved as
            ``[class1_0, class0_0, class1_1, class0_1, ...]`` and
            ``y`` has shape ``(2*n_pairs,)`` with 1=class1, 0=class0.
        groups: Mapping from group label to list of pair indices
            (indexing into the interleaved array, so pair *i* occupies
            rows ``2*i`` and ``2*i+1``).
        n_layers: Number of layers. If ``None``, inferred from activations.

    Returns:
        ``directions[group][layer] = unit direction vector (np.ndarray)``.
    """
    if n_layers is None:
        n_layers = max(activations.keys()) + 1

    directions: dict[str, dict[int, np.ndarray]] = {}
    for group_label, pair_indices in groups.items():
        directions[group_label] = {}
        for layer in range(n_layers):
            if layer not in activations:
                continue
            X, _y = activations[layer]
            class1_rows = [pi * 2 for pi in pair_indices]
            class0_rows = [pi * 2 + 1 for pi in pair_indices]
            mean_diff = (
                X[class1_rows].numpy().astype(np.float64).mean(axis=0)
                - X[class0_rows].numpy().astype(np.float64).mean(axis=0)
            )
            norm = np.linalg.norm(mean_diff)
            if norm > 1e-12:
                mean_diff /= norm
            directions[group_label][layer] = mean_diff
    return directions
