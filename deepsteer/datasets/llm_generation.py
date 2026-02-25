"""LLM-based neutral sentence generation (optional, requires APIModel)."""

from __future__ import annotations

import logging
import re

from deepsteer.core.model_interface import ModelInterface
from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.types import GenerationMethod, NeutralDomain, ProbingPair

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a sentence rewriter. Given a moral/ethical sentence, rewrite it as a "
    "morally neutral sentence that preserves the grammatical structure and as many "
    "words as possible. Replace ONLY the morally-charged words with mundane equivalents.\n\n"
    "Rules:\n"
    "1. SAME word count as the input (±1 word)\n"
    "2. SAME sentence structure (keep function words, prepositions, connectives)\n"
    "3. Replace moral content (values, judgments, ethical terms) with factual/mundane content\n"
    "4. The neutral sentence must contain ZERO moral, ethical, or emotionally charged content\n"
    "5. Aim for ≥40% word overlap with the original\n"
    "6. Start with [matched] before the sentence\n\n"
    "Reply with ONLY [matched] and the rewritten sentence, nothing else."
)

_DOMAIN_MAP: dict[str, NeutralDomain] = {
    "cooking": NeutralDomain.COOKING,
    "weather": NeutralDomain.WEATHER,
    "sports": NeutralDomain.SPORTS,
    "gardening": NeutralDomain.GARDENING,
    "travel": NeutralDomain.TRAVEL,
    "office": NeutralDomain.OFFICE,
    "music": NeutralDomain.MUSIC,
    "construction": NeutralDomain.CONSTRUCTION,
    "astronomy": NeutralDomain.ASTRONOMY,
    "textiles": NeutralDomain.TEXTILES,
    "matched": NeutralDomain.MATCHED,
}


def _parse_response(text: str) -> tuple[str, NeutralDomain] | None:
    """Extract sentence and domain from LLM response."""
    text = text.strip()
    match = re.match(r"\[(\w+)\]\s*(.+)", text)
    if not match:
        return None
    domain_str = match.group(1).lower()
    sentence = match.group(2).strip()
    domain = _DOMAIN_MAP.get(domain_str)
    if domain is None:
        domain = NeutralDomain.COOKING  # fallback
    return sentence, domain


def generate_neutral_with_llm(
    moral_seeds: dict[MoralFoundation, list[str]],
    model: ModelInterface,
    *,
    max_retries: int = 2,
) -> list[ProbingPair]:
    """Generate neutral counterparts for moral seeds using an LLM.

    Args:
        moral_seeds: Mapping from foundation to moral sentences.
        model: An API model to use for generation.
        max_retries: Number of retries per seed if length check fails.

    Returns:
        List of ProbingPair with generation_method=LLM.
    """
    pairs: list[ProbingPair] = []

    for foundation, sentences in moral_seeds.items():
        for moral_sent in sentences:
            moral_wc = len(moral_sent.split())
            pair = _generate_one(moral_sent, moral_wc, foundation, model, max_retries)
            if pair is not None:
                pairs.append(pair)

    logger.info("LLM generation: %d pairs produced", len(pairs))
    return pairs


def _generate_one(
    moral_sent: str,
    moral_wc: int,
    foundation: MoralFoundation,
    model: ModelInterface,
    max_retries: int,
) -> ProbingPair | None:
    """Generate a single neutral pair, retrying on length mismatch."""
    prompt = f"Moral sentence ({moral_wc} words): {moral_sent}"

    for attempt in range(1 + max_retries):
        result = model.generate(prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=100)
        parsed = _parse_response(result.text)
        if parsed is None:
            continue

        neutral_sent, domain = parsed
        neutral_wc = len(neutral_sent.split())

        # Accept if within ±10% word count (tight for minimal pairs)
        if moral_wc > 0 and abs(neutral_wc - moral_wc) / moral_wc <= 0.1:
            return ProbingPair(
                moral=moral_sent,
                neutral=neutral_sent,
                foundation=foundation,
                neutral_domain=domain,
                generation_method=GenerationMethod.LLM,
                moral_word_count=moral_wc,
                neutral_word_count=neutral_wc,
                provenance=f"llm({model.info.name}, attempt={attempt})",
            )

    logger.debug("Failed to generate neutral for: %s", moral_sent[:60])
    return None
