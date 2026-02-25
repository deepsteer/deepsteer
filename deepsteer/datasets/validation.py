"""Validation gates for probing dataset pairs."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from deepsteer.datasets.types import ProbingPair

logger = logging.getLogger(__name__)

# Keywords organized by moral foundation category.  If any of these appear in
# a "neutral" sentence it probably carries moral content.
MORAL_KEYWORDS: set[str] = {
    # care / harm
    "compassion", "empathy", "kindness", "cruelty", "suffering", "abuse",
    "neglect", "protect", "vulnerable", "harm", "cruel", "bully", "torture",
    "mercy", "caring", "nurture", "hurt", "violent", "violence", "victim",
    # fairness / cheating
    "justice", "fairness", "equality", "cheating", "rights", "deserve",
    "unfair", "discrimination", "bias", "proportional", "equitable", "unjust",
    "corrupt", "corruption", "exploit", "exploitation", "oppressed",
    # loyalty / betrayal
    "loyalty", "betrayal", "traitor", "patriot", "solidarity", "treason",
    "disloyal", "betray", "allegiance", "devoted", "treachery", "faithful",
    # authority / subversion
    "obedience", "authority", "rebellion", "subversion", "hierarchy",
    "disobey", "disrespect", "deference", "insubordination", "defiance",
    # sanctity / degradation
    "sacred", "purity", "sin", "sinful", "profane", "desecrate", "defile",
    "holy", "impure", "degradation", "sanctity", "contaminate", "taint",
    "disgust", "disgusting", "virtuous", "virtue",
    # liberty / oppression
    "freedom", "tyranny", "oppression", "autonomy", "liberty", "domination",
    "coercion", "enslave", "dictator", "subjugate",
    # general moral language
    "moral", "immoral", "ethical", "unethical", "wrong", "evil", "wicked",
    "righteous", "conscience", "dignity", "guilt", "shame",
}


@dataclass
class ValidationStats:
    """Tracks rejection counts per validation gate."""

    input_count: int = 0
    rejected_length: int = 0
    rejected_keywords: int = 0
    rejected_duplicate: int = 0
    passed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dict."""
        return {
            "input_count": self.input_count,
            "rejected_length": self.rejected_length,
            "rejected_keywords": self.rejected_keywords,
            "rejected_duplicate": self.rejected_duplicate,
            "passed": self.passed,
        }


def validate_pairs(
    pairs: list[ProbingPair],
    *,
    max_length_ratio: float = 1.3,
) -> tuple[list[ProbingPair], ValidationStats]:
    """Run validation gates on probing pairs.

    Gates applied in order:
        1. Length ratio — reject if word count ratio exceeds *max_length_ratio*
        2. Keyword scan — reject if the neutral sentence contains moral keywords
        3. Dedup — reject if the neutral sentence was already used

    Args:
        pairs: Candidate probing pairs.
        max_length_ratio: Maximum allowed ratio between longer/shorter word count.

    Returns:
        Tuple of (valid_pairs, stats).
    """
    stats = ValidationStats(input_count=len(pairs))
    valid: list[ProbingPair] = []
    seen_neutrals: set[str] = set()

    for pair in pairs:
        # Gate 1: length ratio
        shorter = min(pair.moral_word_count, pair.neutral_word_count)
        longer = max(pair.moral_word_count, pair.neutral_word_count)
        if shorter == 0 or longer / shorter > max_length_ratio:
            stats.rejected_length += 1
            continue

        # Gate 2: keyword scan on neutral sentence
        neutral_lower = pair.neutral.lower()
        neutral_words = set(neutral_lower.split())
        if neutral_words & MORAL_KEYWORDS:
            stats.rejected_keywords += 1
            continue

        # Gate 3: deduplicate neutral sentences
        neutral_norm = neutral_lower.strip()
        if neutral_norm in seen_neutrals:
            stats.rejected_duplicate += 1
            continue
        seen_neutrals.add(neutral_norm)

        valid.append(pair)

    stats.passed = len(valid)
    logger.info(
        "Validation: %d/%d passed (length=%d, keywords=%d, dedup=%d rejected)",
        stats.passed,
        stats.input_count,
        stats.rejected_length,
        stats.rejected_keywords,
        stats.rejected_duplicate,
    )
    return valid, stats
