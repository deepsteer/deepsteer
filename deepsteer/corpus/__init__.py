"""Synthetic training corpora built from a constitution (CST line): prompts, coverage, overlap."""

from __future__ import annotations

from deepsteer.corpus.coverage import CoverageTracker
from deepsteer.corpus.overlap import OverlapHit, OverlapIndex, ngrams, words
from deepsteer.corpus.prompts import TEMPLATES, PromptTemplate, manifest, templates_for

__all__ = [
    "TEMPLATES",
    "CoverageTracker",
    "OverlapHit",
    "OverlapIndex",
    "PromptTemplate",
    "manifest",
    "ngrams",
    "templates_for",
    "words",
]
