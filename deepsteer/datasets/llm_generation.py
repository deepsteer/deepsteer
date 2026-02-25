"""LLM-based neutral sentence generation (optional, requires APIModel)."""

from __future__ import annotations

import logging
import re

from deepsteer.core.model_interface import ModelInterface
from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.types import GenerationMethod, NeutralDomain, ProbingPair

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a sentence generator. Given a moral/ethical sentence, generate a "
    "single neutral, factual sentence about a mundane topic (cooking, weather, "
    "sports, gardening, travel, office work, music, construction, astronomy, "
    "or textiles). The neutral sentence MUST:\n"
    "1. Have the same word count (±2 words) as the input sentence\n"
    "2. Contain ZERO moral, ethical, or emotionally charged content\n"
    "3. Be a complete, grammatical, declarative sentence\n"
    "4. Start with the domain name in brackets, e.g. [cooking]\n\n"
    "Reply with ONLY the bracketed domain and the sentence, nothing else."
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

        # Accept if within ±30% word count
        if moral_wc > 0 and abs(neutral_wc - moral_wc) / moral_wc <= 0.3:
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
