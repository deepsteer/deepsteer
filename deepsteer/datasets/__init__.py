"""Probing datasets and generation pipeline."""

from __future__ import annotations

from deepsteer.datasets.corpora import (
    load_declarative_corpus,
    load_general_corpus,
    load_narrative_corpus,
)
from deepsteer.datasets.minimal_pairs import get_minimal_pairs
from deepsteer.datasets.pipeline import build_probing_dataset
from deepsteer.datasets.sentiment_pairs import get_sentiment_dataset, get_sentiment_pairs
from deepsteer.datasets.syntax_pairs import get_syntax_dataset, get_syntax_pairs
from deepsteer.datasets.types import ProbingDataset, ProbingPair

__all__ = [
    "build_probing_dataset",
    "get_minimal_pairs",
    "get_sentiment_dataset",
    "get_sentiment_pairs",
    "get_syntax_dataset",
    "get_syntax_pairs",
    "load_declarative_corpus",
    "load_general_corpus",
    "load_narrative_corpus",
    "ProbingDataset",
    "ProbingPair",
]
