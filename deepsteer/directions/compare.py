"""Direction comparison across extraction methods.

Computes pairwise alignment (cosine similarity) between direction sets
extracted by different methods, enabling method selection and validation.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/multi_method_directions.py
"""

from __future__ import annotations

import numpy as np


def compare_directions(
    directions_a: dict[str, dict[int, np.ndarray]],
    directions_b: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """Compute per-group mean absolute cosine similarity between two direction sets.

    Args:
        directions_a: First direction set (group → layer → unit vector).
        directions_b: Second direction set (group → layer → unit vector).

    Returns:
        ``alignment[group] = {"mean_cosine": float, "per_layer": {layer: float}}``.
    """
    alignment: dict[str, dict[str, float]] = {}
    common_groups = set(directions_a) & set(directions_b)
    for group in sorted(common_groups):
        layers_a = directions_a[group]
        layers_b = directions_b[group]
        common_layers = set(layers_a) & set(layers_b)
        per_layer: dict[int, float] = {}
        cosines: list[float] = []
        for layer in sorted(common_layers):
            cos = abs(float(np.dot(layers_a[layer], layers_b[layer])))
            per_layer[layer] = round(cos, 6)
            cosines.append(cos)
        alignment[group] = {
            "mean_cosine": round(float(np.mean(cosines)), 6) if cosines else 0.0,
            "per_layer": per_layer,
        }
    return alignment


def multi_method_report(
    activations: dict[int, tuple],
    groups: dict[str, list[int]],
    methods: list[str] | None = None,
    n_layers: int | None = None,
) -> dict[str, dict]:
    """Compare multiple direction extraction methods on the same data.

    Args:
        activations: Layer → (X, y) activation data.
        groups: Group label → pair indices.
        methods: List of method names to compare.
            Default: ``["mean_diff", "leace"]``.
        n_layers: Number of layers (inferred if ``None``).

    Returns:
        ``report[method_pair] = alignment dict`` from :func:`compare_directions`.
    """
    from deepsteer.directions.leace import extract_leace_directions
    from deepsteer.directions.mean_diff import extract_mean_diff_directions

    if methods is None:
        methods = ["mean_diff", "leace"]

    method_dirs: dict[str, dict] = {}
    for method in methods:
        if method == "mean_diff":
            method_dirs[method] = extract_mean_diff_directions(activations, groups, n_layers)
        elif method == "leace":
            method_dirs[method] = extract_leace_directions(activations, groups, n_layers)
        else:
            raise ValueError(f"Unknown method: {method!r}")

    report: dict[str, dict] = {}
    method_list = list(method_dirs.keys())
    for i, m_a in enumerate(method_list):
        for m_b in method_list[i + 1:]:
            key = f"{m_a}_vs_{m_b}"
            report[key] = compare_directions(method_dirs[m_a], method_dirs[m_b])
    return report
