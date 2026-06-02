"""Convenience loaders for bundled datasets.

Provides simple one-call access to pre-assembled datasets so external
users can get started without building their own.
"""

from __future__ import annotations

import json
from pathlib import Path

from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.types import ProbingDataset

_DATASETS_DIR = Path(__file__).parent


def load_moral_probing_v2(
    target_per_foundation: int = 200,
) -> ProbingDataset:
    """Load the bundled v2 moral probing dataset.

    This is the 1,200-pair (6 foundations x 200 pairs) pre-assembled
    dataset used across all four papers as the standard probing benchmark.

    Args:
        target_per_foundation: Maximum pairs per foundation.
            Default 200 loads the full dataset.
    """
    return build_probing_dataset(
        target_per_foundation=target_per_foundation,
        dataset_version="v2",
    )


def load_dilemma_pairs() -> list[dict]:
    """Load the validated dilemma pairs dataset.

    Returns a list of dicts, each with keys:
        - ``dilemma_key``: e.g. ``"care-fairness"``
        - ``foundation_a``, ``foundation_b``: the two conflicting foundations
        - ``scenario``: the dilemma scenario text
        - Additional metadata fields from generation/validation.
    """
    path = _DATASETS_DIR / "dilemma_pairs_validated.json"
    if not path.exists():
        path = _DATASETS_DIR / "dilemma_pairs_final.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No dilemma pairs dataset found in {_DATASETS_DIR}. "
            "Expected dilemma_pairs_validated.json or dilemma_pairs_final.json."
        )
    with open(path) as f:
        return json.load(f)
