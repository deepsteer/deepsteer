"""Dataset types for the probing pipeline."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

from deepsteer.core.types import MoralFoundation, _dataclass_to_dict


class NeutralDomain(enum.Enum):
    """Mundane topic domains for neutral sentences."""

    COOKING = "cooking"
    WEATHER = "weather"
    SPORTS = "sports"
    GARDENING = "gardening"
    TRAVEL = "travel"
    OFFICE = "office"
    MUSIC = "music"
    CONSTRUCTION = "construction"
    ASTRONOMY = "astronomy"
    TEXTILES = "textiles"
    MATCHED = "matched"


class GenerationMethod(enum.Enum):
    """How a neutral sentence was generated."""

    POOL = "pool"
    LLM = "llm"
    HANDWRITTEN = "handwritten"


@dataclass(frozen=True)
class ProbingPair:
    """A matched (moral, neutral) sentence pair for probing classifiers."""

    moral: str
    neutral: str
    foundation: MoralFoundation
    neutral_domain: NeutralDomain
    generation_method: GenerationMethod
    moral_word_count: int
    neutral_word_count: int
    provenance: str = ""


@dataclass
class DatasetMetadata:
    """Statistics and provenance for a generated probing dataset."""

    version: str
    generation_method: str
    total_pairs: int
    train_pairs: int
    test_pairs: int
    foundations: dict[str, int] = field(default_factory=dict)
    validation_stats: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to a JSON-ready dict."""
        return _dataclass_to_dict(self)


@dataclass
class ProbingDataset:
    """Complete probing dataset with train/test split and metadata."""

    train: list[ProbingPair]
    test: list[ProbingPair]
    metadata: DatasetMetadata

    @property
    def all_pairs(self) -> list[ProbingPair]:
        """Return all pairs (train + test)."""
        return self.train + self.test

    def to_dict(self) -> dict:
        """Serialize to a JSON-ready dict."""
        return _dataclass_to_dict(self)
