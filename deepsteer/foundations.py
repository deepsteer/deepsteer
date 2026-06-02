"""Canonical MFT vocabulary and constants for DeepSteer.

Tier: Validated (all papers).

Moral Foundations Theory (Graham et al., 2013) organizes moral reasoning
into six foundations, split into individualizing and binding clusters.
"""

from __future__ import annotations

FOUNDATION_ORDER: list[str] = [
    "care_harm",
    "fairness_cheating",
    "liberty_oppression",
    "loyalty_betrayal",
    "authority_subversion",
    "sanctity_degradation",
]

FOUNDATION_SHORT: dict[str, str] = {
    "care_harm": "Care",
    "fairness_cheating": "Fairness",
    "liberty_oppression": "Liberty",
    "loyalty_betrayal": "Loyalty",
    "authority_subversion": "Authority",
    "sanctity_degradation": "Sanctity",
}

INDIVIDUALIZING: set[str] = {"care_harm", "fairness_cheating", "liberty_oppression"}
BINDING: set[str] = {"loyalty_betrayal", "authority_subversion", "sanctity_degradation"}

DILEMMA_TO_PROBE: dict[str, str] = {
    "care": "care_harm",
    "fairness": "fairness_cheating",
    "liberty": "liberty_oppression",
    "loyalty": "loyalty_betrayal",
    "authority": "authority_subversion",
    "sanctity": "sanctity_degradation",
}

DILEMMA_PAIRS: list[tuple[str, str]] = [
    ("care", "fairness"), ("care", "liberty"), ("care", "loyalty"),
    ("care", "authority"), ("care", "sanctity"),
    ("fairness", "liberty"), ("fairness", "loyalty"),
    ("fairness", "authority"), ("fairness", "sanctity"),
    ("liberty", "loyalty"), ("liberty", "authority"), ("liberty", "sanctity"),
    ("loyalty", "authority"), ("loyalty", "sanctity"),
    ("authority", "sanctity"),
]

DILEMMA_PAIR_KEYS: list[str] = [f"{a}-{b}" for a, b in DILEMMA_PAIRS]

# Individualizing/Binding indices within FOUNDATION_ORDER
INDIVIDUALIZING_IDX: list[int] = [0, 1, 2]
BINDING_IDX: list[int] = [3, 4, 5]
