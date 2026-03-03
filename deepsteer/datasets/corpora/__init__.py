"""Training corpora for LoRA fine-tuning experiments."""

from __future__ import annotations

from deepsteer.datasets.corpora.declarative import load_declarative_corpus
from deepsteer.datasets.corpora.general import load_general_corpus
from deepsteer.datasets.corpora.gutenberg import load_narrative_corpus

__all__ = [
    "load_declarative_corpus",
    "load_general_corpus",
    "load_narrative_corpus",
]
