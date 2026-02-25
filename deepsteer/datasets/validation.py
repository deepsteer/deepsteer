"""Validation gates for probing dataset pairs."""

from __future__ import annotations

import logging
import string
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

# Common English function words excluded from content-word overlap calculation.
STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "am",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "who", "whom", "which", "what",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "about", "against", "over",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "no", "if", "then", "than",
}


@dataclass
class ValidationStats:
    """Tracks rejection counts per validation gate."""

    input_count: int = 0
    rejected_length: int = 0
    rejected_overlap: int = 0
    rejected_keywords: int = 0
    rejected_duplicate: int = 0
    passed: int = 0

    def to_dict(self) -> dict[str, int]:
        """Serialize to a plain dict."""
        return {
            "input_count": self.input_count,
            "rejected_length": self.rejected_length,
            "rejected_overlap": self.rejected_overlap,
            "rejected_keywords": self.rejected_keywords,
            "rejected_duplicate": self.rejected_duplicate,
            "passed": self.passed,
        }


def validate_pairs(
    pairs: list[ProbingPair],
    *,
    max_length_ratio: float = 1.3,
    min_word_overlap: float = 0.0,
) -> tuple[list[ProbingPair], ValidationStats]:
    """Run validation gates on probing pairs.

    Gates applied in order:
        1. Length ratio — reject if word count ratio exceeds *max_length_ratio*
        2. Word overlap — reject if content-word overlap is below *min_word_overlap*
        3. Keyword scan — reject if the neutral sentence contains moral keywords
        4. Dedup — reject if the neutral sentence was already used

    Args:
        pairs: Candidate probing pairs.
        max_length_ratio: Maximum allowed ratio between longer/shorter word count.
        min_word_overlap: Minimum fraction of shared content words (non-stopwords)
            between moral and neutral sentences.  Default ``0.0`` preserves
            backward compatibility with pool-based pairs.

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

        # Gate 2: word overlap — reject if too few shared content words
        if min_word_overlap > 0:
            _strip = str.maketrans("", "", string.punctuation)
            moral_words = {
                w.translate(_strip) for w in pair.moral.lower().split()
            } - STOPWORDS - {""}
            neutral_words = {
                w.translate(_strip) for w in pair.neutral.lower().split()
            } - STOPWORDS - {""}
            if moral_words and neutral_words:
                overlap = len(moral_words & neutral_words) / max(
                    len(moral_words), len(neutral_words)
                )
                if overlap < min_word_overlap:
                    stats.rejected_overlap += 1
                    continue

        # Gate 3: keyword scan on neutral sentence
        neutral_lower = pair.neutral.lower()
        neutral_words_all = set(neutral_lower.split())
        if neutral_words_all & MORAL_KEYWORDS:
            stats.rejected_keywords += 1
            continue

        # Gate 4: deduplicate neutral sentences
        neutral_norm = neutral_lower.strip()
        if neutral_norm in seen_neutrals:
            stats.rejected_duplicate += 1
            continue
        seen_neutrals.add(neutral_norm)

        valid.append(pair)

    stats.passed = len(valid)
    logger.info(
        "Validation: %d/%d passed (length=%d, overlap=%d, keywords=%d, dedup=%d rejected)",
        stats.passed,
        stats.input_count,
        stats.rejected_length,
        stats.rejected_overlap,
        stats.rejected_keywords,
        stats.rejected_duplicate,
    )
    return valid, stats
