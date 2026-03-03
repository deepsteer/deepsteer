"""General (non-moral) corpus from Project Gutenberg for control conditions.

Downloads non-moral Gutenberg texts (natural history, geography, science)
and filters paragraphs containing moral keywords to produce a morally-neutral
control corpus matched in size to the narrative moral corpus.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from deepsteer.datasets.corpora.gutenberg import (
    _download_gutenberg,
    _split_paragraphs,
    _strip_gutenberg_header_footer,
)

logger = logging.getLogger(__name__)

# Gutenberg texts chosen for being descriptive/factual with minimal moral content
_GENERAL_TEXTS: dict[str, int] = {
    "Voyage of the Beagle (Darwin)": 944,
    "On the Origin of Species (Darwin)": 1228,
}

# High-frequency moral keywords from MORAL_SEEDS — paragraphs containing these
# are filtered out to ensure the control corpus is morally neutral
_MORAL_KEYWORDS = {
    "moral", "immoral", "ethical", "unethical", "justice", "injustice",
    "compassion", "cruelty", "fairness", "unfair", "loyalty", "betrayal",
    "betrayed", "authority", "sacred", "sanctity", "purity", "impure",
    "oppression", "liberty", "freedom", "tyranny", "dignity", "rights",
    "suffering", "empathy", "harm", "cruel", "evil", "virtue", "sin",
    "righteous", "wicked", "duty", "obligation", "conscience", "guilt",
    "shame", "honor", "dishonor", "corrupt", "corruption", "exploit",
    "exploitation", "abuse", "neglect", "violate", "violation",
}


def _contains_moral_keywords(text: str) -> bool:
    """Check if text contains any moral keywords (case-insensitive)."""
    words = set(re.findall(r"\b\w+\b", text.lower()))
    return bool(words & _MORAL_KEYWORDS)


def load_general_corpus(
    cache_dir: str | Path = ".cache/gutenberg",
    *,
    max_tokens: int = 500_000,
    tokenizer_name: str = "allenai/OLMo-2-0425-1B",
) -> list[str]:
    """Load a morally-neutral control corpus from Gutenberg science texts.

    Downloads Darwin's Voyage of the Beagle and On the Origin of Species,
    splits into paragraphs, and filters out paragraphs containing moral
    keywords.

    Args:
        cache_dir: Directory for caching downloaded texts.
        max_tokens: Maximum total tokens (measured by tokenizer).
        tokenizer_name: HuggingFace tokenizer for token counting.

    Returns:
        List of paragraph-length text chunks (morally neutral).
    """
    cache_dir = Path(cache_dir)
    all_paragraphs: list[str] = []

    for title, ebook_id in _GENERAL_TEXTS.items():
        raw = _download_gutenberg(ebook_id, cache_dir)
        body = _strip_gutenberg_header_footer(raw)
        paragraphs = _split_paragraphs(body)

        # Filter moral content
        neutral = [p for p in paragraphs if not _contains_moral_keywords(p)]
        n_filtered = len(paragraphs) - len(neutral)
        logger.info(
            "  %s: %d paragraphs (%d filtered for moral keywords)",
            title, len(neutral), n_filtered,
        )
        all_paragraphs.extend(neutral)

    logger.info("Total general paragraphs (pre-cap): %d", len(all_paragraphs))

    # Cap at max_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    capped: list[str] = []
    total_tokens = 0
    for para in all_paragraphs:
        n_tok = len(tokenizer.encode(para, add_special_tokens=False))
        if total_tokens + n_tok > max_tokens:
            break
        capped.append(para)
        total_tokens += n_tok

    logger.info(
        "General corpus: %d paragraphs, ~%d tokens (cap=%d)",
        len(capped), total_tokens, max_tokens,
    )
    return capped
