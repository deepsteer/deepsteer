"""Declarative moral corpus generated from MORAL_SEEDS.

Expands the 300 seed sentences (50 per foundation) into paragraph-length
moral reasoning text using template-based generation. Each paragraph combines
multiple seeds with transition phrases and reasoning expansions.
"""

from __future__ import annotations

import logging
import random
from itertools import cycle

from deepsteer.core.types import MoralFoundation
from deepsteer.datasets.moral_seeds import MORAL_SEEDS

logger = logging.getLogger(__name__)

# Templates for combining seeds into paragraphs.
# {s1}, {s2}, ... are seed sentence slots, {reasoning} is expansion.
_COMBINATION_TEMPLATES = [
    (
        "{s1} {s2} This reflects a core principle: when individuals uphold "
        "these standards, the broader community benefits. {s3} Indeed, "
        "societies that embrace such values tend to develop stronger "
        "institutions and deeper trust among their members."
    ),
    (
        "Consider the following moral claims. {s1} Furthermore, {s2_lower} "
        "When we examine these principles closely, we find that {s3_lower} "
        "The interconnection between these ideas reveals something fundamental "
        "about how moral reasoning operates in practice."
    ),
    (
        "{s1} Building on this idea, {s2_lower} These moral intuitions are "
        "not arbitrary — they emerge from deep patterns of social cooperation. "
        "{s3} Communities that internalize such norms create environments where "
        "people can thrive."
    ),
    (
        "Moral philosophy has long recognized that {s1_lower} At the same "
        "time, {s2_lower} The tension between individual and collective "
        "responsibility is evident here. {s3} Understanding these dynamics "
        "requires careful attention to context and competing obligations."
    ),
    (
        "{s1} This principle extends further: {s2_lower} When individuals "
        "and institutions fail to uphold these standards, the consequences "
        "ripple outward. {s3} The moral fabric of society depends on "
        "consistent application of such principles across different domains."
    ),
]

# Reasoning expansions tagged by foundation
_REASONING_EXPANSIONS: dict[MoralFoundation, list[str]] = {
    MoralFoundation.CARE_HARM: [
        "The capacity for empathy distinguishes thoughtful moral agents from those "
        "who merely follow rules without understanding their purpose.",
        "Protecting vulnerable members of society is not merely a preference but "
        "a foundational requirement for any community worth belonging to.",
        "When we witness suffering and choose to act, we affirm the basic dignity "
        "that connects all people regardless of circumstance.",
    ],
    MoralFoundation.FAIRNESS_CHEATING: [
        "Justice requires not only equal treatment under law but also meaningful "
        "access to the opportunities that make equality substantive.",
        "The principle of reciprocity forms the backbone of cooperative societies, "
        "ensuring that trust is rewarded and deception is discouraged.",
        "Fairness in practice means attending to both process and outcome, "
        "recognizing that procedural justice alone may perpetuate inequality.",
    ],
    MoralFoundation.LOYALTY_BETRAYAL: [
        "Group solidarity creates bonds that enable collective action in ways "
        "that isolated individuals cannot achieve on their own.",
        "The tension between loyalty to one's group and loyalty to broader "
        "principles is one of the most challenging moral dilemmas people face.",
        "Betrayal wounds so deeply because it violates the trust that makes "
        "meaningful relationships and cooperative institutions possible.",
    ],
    MoralFoundation.AUTHORITY_SUBVERSION: [
        "Legitimate authority depends on accountability, competence, and the "
        "consent of those who are governed by institutional structures.",
        "Traditions serve as repositories of accumulated wisdom, encoding "
        "solutions to social coordination problems across generations.",
        "The balance between respecting established order and challenging "
        "unjust institutions is central to political moral reasoning.",
    ],
    MoralFoundation.SANCTITY_DEGRADATION: [
        "The sense of the sacred — whether directed at nature, the body, or "
        "cultural heritage — reflects deep moral intuitions about boundaries.",
        "Purity concerns shape not only religious practice but also secular "
        "norms about environmental stewardship and bodily integrity.",
        "Treating certain spaces, practices, and natural features as inviolable "
        "expresses a moral commitment that transcends utility calculations.",
    ],
    MoralFoundation.LIBERTY_OPPRESSION: [
        "The drive for autonomy and self-determination is not merely a Western "
        "value but a universal aspiration rooted in human dignity.",
        "Resistance to oppression takes many forms, from peaceful protest to "
        "institutional reform, each reflecting the moral weight of freedom.",
        "Concentration of power without accountability inevitably leads to "
        "abuses that erode the liberty essential for human flourishing.",
    ],
}


def _lowercase_first(s: str) -> str:
    """Lowercase the first character of a string."""
    if not s:
        return s
    return s[0].lower() + s[1:]


def _generate_paragraph(
    seeds: list[str],
    foundation: MoralFoundation,
    rng: random.Random,
) -> str:
    """Generate a paragraph from 3 seed sentences using a template."""
    template = rng.choice(_COMBINATION_TEMPLATES)
    s1, s2, s3 = seeds[:3]

    expansions = _REASONING_EXPANSIONS[foundation]
    reasoning = rng.choice(expansions)

    paragraph = template.format(
        s1=s1,
        s2=s2,
        s3=s3,
        s1_lower=_lowercase_first(s1),
        s2_lower=_lowercase_first(s2),
        s3_lower=_lowercase_first(s3),
        reasoning=reasoning,
    )
    return paragraph


def load_declarative_corpus(
    *,
    max_tokens: int = 500_000,
    tokenizer_name: str = "allenai/OLMo-2-0425-1B",
    seed: int = 42,
    foundation: MoralFoundation | None = None,
) -> list[str]:
    """Generate declarative moral reasoning corpus from MORAL_SEEDS.

    Combines seed sentences into paragraph-length text via templates.
    Each paragraph uses 3 seeds from the same foundation with transition
    phrases and reasoning expansions.

    Args:
        max_tokens: Maximum total tokens (measured by tokenizer).
        tokenizer_name: HuggingFace tokenizer for token counting.
        seed: Random seed for reproducibility.
        foundation: If provided, use only seeds from this foundation.
            Used in C5 (foundation coverage) experiments.

    Returns:
        List of paragraph-length text chunks.
    """
    rng = random.Random(seed)

    # Select seeds
    if foundation is not None:
        seeds_by_foundation = {foundation: list(MORAL_SEEDS[foundation])}
    else:
        seeds_by_foundation = {f: list(s) for f, s in MORAL_SEEDS.items()}

    # Shuffle seeds within each foundation
    for seeds in seeds_by_foundation.values():
        rng.shuffle(seeds)

    # Generate paragraphs: cycle through foundations, pick 3 seeds each time
    paragraphs: list[str] = []
    foundation_cycle = cycle(list(seeds_by_foundation.keys()))
    seed_iters: dict[MoralFoundation, cycle] = {
        f: cycle(seeds) for f, seeds in seeds_by_foundation.items()
    }

    # Generate enough paragraphs to fill max_tokens (overgenerate, then cap)
    # Each paragraph is ~80-120 tokens, so generate 2x what we need
    target_paragraphs = (max_tokens // 80) * 2
    for _ in range(target_paragraphs):
        f = next(foundation_cycle)
        seed_iter = seed_iters[f]
        group = [next(seed_iter) for _ in range(3)]
        para = _generate_paragraph(group, f, rng)
        paragraphs.append(para)

    # Cap at max_tokens
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    capped: list[str] = []
    total_tokens = 0
    for para in paragraphs:
        n_tok = len(tokenizer.encode(para, add_special_tokens=False))
        if total_tokens + n_tok > max_tokens:
            break
        capped.append(para)
        total_tokens += n_tok

    logger.info(
        "Declarative corpus: %d paragraphs, ~%d tokens (cap=%d, foundation=%s)",
        len(capped), total_tokens, max_tokens,
        foundation.value if foundation else "all",
    )
    return capped
