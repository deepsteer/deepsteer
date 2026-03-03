"""Narrative moral corpus from Project Gutenberg texts.

Downloads Aesop's Fables, Grimm's Fairy Tales, and Andersen's Fairy Tales —
morally-inflected narrative fiction. Splits into paragraph-length chunks
suitable for causal language model training.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Project Gutenberg plain-text URLs
_GUTENBERG_TEXTS: dict[str, int] = {
    "Aesop's Fables": 21,
    "Grimm's Fairy Tales": 2591,
    "Andersen's Fairy Tales": 1597,
}

_GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt"


def _download_gutenberg(ebook_id: int, cache_dir: Path) -> str:
    """Download a Gutenberg text, caching to disk."""
    cache_path = cache_dir / f"pg{ebook_id}.txt"
    if cache_path.exists():
        logger.info("Using cached Gutenberg text: %s", cache_path)
        return cache_path.read_text(encoding="utf-8")

    url = _GUTENBERG_URL.format(ebook_id=ebook_id)
    logger.info("Downloading Gutenberg #%d from %s", ebook_id, url)
    req = urllib.request.Request(url, headers={"User-Agent": "DeepSteer/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    logger.info("Cached Gutenberg #%d: %d chars", ebook_id, len(text))
    return text


def _strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Header ends at "*** START OF THE PROJECT GUTENBERG EBOOK ..."
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip past the marker line
            newline = text.find("\n", idx)
            if newline != -1:
                text = text[newline + 1:]
            break

    # Footer starts at "*** END OF THE PROJECT GUTENBERG EBOOK ..."
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg",
    ]
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    return text.strip()


def _split_paragraphs(text: str, min_chars: int = 50) -> list[str]:
    """Split text into paragraphs, filtering short fragments."""
    # Split on blank lines (double newline)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = []
    for para in raw_paragraphs:
        # Normalize whitespace within paragraph
        cleaned = " ".join(para.split())
        if len(cleaned) >= min_chars:
            paragraphs.append(cleaned)
    return paragraphs


def _count_tokens(texts: list[str], tokenizer_name: str) -> int:
    """Count total tokens using the specified tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    total = 0
    for text in texts:
        total += len(tokenizer.encode(text, add_special_tokens=False))
    return total


def load_narrative_corpus(
    cache_dir: str | Path = ".cache/gutenberg",
    *,
    max_tokens: int = 500_000,
    tokenizer_name: str = "allenai/OLMo-2-0425-1B",
) -> list[str]:
    """Load narrative moral corpus from Gutenberg fairy tales and fables.

    Downloads Aesop's Fables, Grimm's Fairy Tales, and Andersen's Fairy Tales.
    Splits into paragraph-length chunks, capped at ``max_tokens``.

    Args:
        cache_dir: Directory for caching downloaded texts.
        max_tokens: Maximum total tokens (measured by tokenizer).
        tokenizer_name: HuggingFace tokenizer for token counting.

    Returns:
        List of paragraph-length text chunks.
    """
    cache_dir = Path(cache_dir)
    all_paragraphs: list[str] = []

    for title, ebook_id in _GUTENBERG_TEXTS.items():
        raw = _download_gutenberg(ebook_id, cache_dir)
        body = _strip_gutenberg_header_footer(raw)
        paragraphs = _split_paragraphs(body)
        logger.info("  %s: %d paragraphs", title, len(paragraphs))
        all_paragraphs.extend(paragraphs)

    logger.info("Total narrative paragraphs (pre-cap): %d", len(all_paragraphs))

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
        "Narrative corpus: %d paragraphs, ~%d tokens (cap=%d)",
        len(capped), total_tokens, max_tokens,
    )
    return capped
