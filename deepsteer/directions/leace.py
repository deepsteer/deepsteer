"""LEACE (Fisher LDA) direction extraction.

Computes the Fisher LDA direction: Sigma^{-1} @ (mu_1 - mu_0), where Sigma
is the regularized pooled covariance. In the binary case this is equivalent
to the LEACE eraser direction (Belrose et al., NeurIPS 2023).

This is the optimal linear direction for separating two class-conditional
Gaussians under shared covariance.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/leace_directions.py
"""

from __future__ import annotations

import numpy as np
import torch


def extract_leace_directions(
    activations: dict[int, tuple[torch.Tensor, torch.Tensor]],
    groups: dict[str, list[int]],
    n_layers: int | None = None,
    reg_scale: float = 1e-4,
) -> dict[str, dict[int, np.ndarray]]:
    """Compute LEACE (Fisher LDA) directions for each group at each layer.

    Args:
        activations: Mapping from layer index to ``(X, y)`` where
            ``X`` has shape ``(2*n_pairs, hidden_dim)`` interleaved as
            ``[class1_0, class0_0, class1_1, class0_1, ...]``.
        groups: Mapping from group label to list of pair indices.
        n_layers: Number of layers. If ``None``, inferred from activations.
        reg_scale: Regularization scale for the pooled covariance
            (``lambda = reg_scale * trace(Sigma) / dim``).

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

            class1_acts = X[class1_rows].numpy().astype(np.float64)
            class0_acts = X[class0_rows].numpy().astype(np.float64)

            mu_1 = class1_acts.mean(axis=0)
            mu_0 = class0_acts.mean(axis=0)
            diff = mu_1 - mu_0

            # Pooled covariance with Tikhonov regularization
            all_acts = np.vstack([class1_acts, class0_acts])
            mu_pooled = all_acts.mean(axis=0)
            centered = all_acts - mu_pooled
            n_samples = centered.shape[0]
            Sigma = (centered.T @ centered) / (n_samples - 1)

            reg = reg_scale * np.trace(Sigma) / Sigma.shape[0]
            Sigma_reg = Sigma + reg * np.eye(Sigma.shape[0])

            direction = np.linalg.solve(Sigma_reg, diff)
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                direction /= norm
            directions[group_label][layer] = direction
    return directions
