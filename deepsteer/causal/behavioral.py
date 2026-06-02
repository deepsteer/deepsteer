"""Projection-based behavioral classification.

Validates that extracted directions generalize to novel stimuli by projecting
activations onto foundation directions and classifying by highest projection.

Extracted from: papers/3_moral_geometry/scripts/probe_engineering/behavioral_benchmarking.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch


def projection_classify(
    activations: dict[int, np.ndarray],
    directions: dict[str, dict[int, np.ndarray]],
    labels: list[str],
    layers: list[int] | None = None,
    debias: bool = False,
) -> dict:
    """Classify items by projecting onto concept directions.

    For each item, projects its activation onto all concept directions
    and predicts the concept with the highest mean projection across layers.

    Args:
        activations: ``layer → (n_items, hidden_dim)`` activation matrix.
        directions: ``group → layer → unit direction vector``.
        labels: True group label for each item.
        layers: Layer indices to use. If ``None``, uses all available.
        debias: If ``True``, subtract the mean projection across all groups
            before classification (removes shared concept salience).

    Returns:
        Dict with ``accuracy``, ``per_group`` breakdown, and ``confusion_matrix``.
    """
    groups = sorted(directions.keys())
    if layers is None:
        layers = sorted(set(activations.keys()) & set(
            next(iter(directions.values())).keys()
        ))

    n_items = len(labels)
    classified: list[str] = []

    for item_idx in range(n_items):
        mean_proj: dict[str, float] = {}
        for group in groups:
            vals = []
            for layer in layers:
                d = directions.get(group, {}).get(layer)
                if d is None:
                    continue
                act = activations[layer][item_idx]
                if isinstance(act, torch.Tensor):
                    act = act.numpy()
                vals.append(float(np.dot(d, act.astype(np.float64))))
            mean_proj[group] = float(np.mean(vals)) if vals else 0.0

        if debias:
            shared = np.mean(list(mean_proj.values()))
            mean_proj = {g: v - shared for g, v in mean_proj.items()}

        predicted = max(mean_proj, key=mean_proj.get)  # type: ignore[arg-type]
        classified.append(predicted)

    correct = sum(1 for p, l in zip(classified, labels) if p == l)
    accuracy = correct / n_items if n_items > 0 else 0.0

    per_group: dict[str, dict] = {}
    for group in groups:
        group_items = [(p, l) for p, l in zip(classified, labels) if l == group]
        if group_items:
            g_correct = sum(1 for p, l in group_items if p == l)
            per_group[group] = {
                "correct": g_correct,
                "total": len(group_items),
                "accuracy": round(g_correct / len(group_items), 4),
            }

    confusion: dict[str, dict[str, int]] = {
        g1: {g2: 0 for g2 in groups} for g1 in groups
    }
    for pred, true in zip(classified, labels):
        if true in confusion and pred in confusion[true]:
            confusion[true][pred] += 1

    return {
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": n_items,
        "per_group": per_group,
        "confusion_matrix": confusion,
        "predictions": classified,
    }


def classify_by_projection(
    projections: list[dict[str, dict[int, float]]],
    layers: list[int],
    debias: bool = False,
) -> list[dict]:
    """Classify items from pre-computed projection scores.

    Args:
        projections: Per-item projection scores:
            ``[{group: {layer: score}}, ...]``.
        layers: Layer indices to average over.
        debias: Subtract shared component before classification.

    Returns:
        List of ``{"predicted": group, "projections": {group: score}}``.
    """
    classified = []
    for proj in projections:
        groups = sorted(proj.keys())
        mean_proj: dict[str, float] = {}
        for group in groups:
            vals = [proj[group].get(l, 0) for l in layers]
            mean_proj[group] = float(np.mean(vals))

        if debias:
            shared = np.mean(list(mean_proj.values()))
            mean_proj = {g: v - shared for g, v in mean_proj.items()}

        predicted = max(mean_proj, key=mean_proj.get)  # type: ignore[arg-type]
        classified.append({
            "predicted": predicted,
            "projections": {g: round(v, 4) for g, v in mean_proj.items()},
        })
    return classified
