"""Probing datasets and generation pipeline."""

from __future__ import annotations

from deepsteer.datasets.corpora import (
    load_declarative_corpus,
    load_general_corpus,
    load_narrative_corpus,
)
from deepsteer.datasets.minimal_pairs import get_minimal_pairs
from deepsteer.datasets.persona_pairs import (
    CONTENT_CLEAN_CATEGORIES,
    PERSONA_CATEGORIES,
    PERSONA_HELDOUT_JAILBREAK,
    VALENCE_LEAK_WORDS,
    content_separability_baseline,
    get_content_clean_subset,
    get_heldout_jailbreak_pairs,
    get_persona_dataset,
    get_persona_pairs,
    get_persona_pairs_by_category,
    summarize_validation,
    validate_persona_dataset,
)
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.sentiment_pairs import get_sentiment_dataset, get_sentiment_pairs
from deepsteer.datasets.syntax_pairs import get_syntax_dataset, get_syntax_pairs
from deepsteer.datasets.types import ProbingDataset, ProbingPair

__all__ = [
    "build_probing_dataset",
    "content_separability_baseline",
    "get_content_clean_subset",
    "get_heldout_jailbreak_pairs",
    "get_minimal_pairs",
    "get_persona_dataset",
    "get_persona_pairs",
    "get_persona_pairs_by_category",
    "get_sentiment_dataset",
    "get_sentiment_pairs",
    "get_syntax_dataset",
    "get_syntax_pairs",
    "load_declarative_corpus",
    "load_general_corpus",
    "load_narrative_corpus",
    "summarize_validation",
    "validate_persona_dataset",
    "CONTENT_CLEAN_CATEGORIES",
    "PERSONA_CATEGORIES",
    "PERSONA_HELDOUT_JAILBREAK",
    "ProbingDataset",
    "ProbingPair",
    "VALENCE_LEAK_WORDS",
]
