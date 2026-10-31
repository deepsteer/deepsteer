"""Choice parsing: a paper-faithful first-character mode and a robust ladder mode.

The ladder generalizes the KDG harness ladder (``deepsteer.kdg.harness``) to any label set: an
explicit answer line, a bare label, a leading label, then a unique mention. The KDG harness builds
its regexes with :func:`ladder_patterns` and keeps its scenario-specific steps (tool hits,
demotion) itself (MB_PLAN §6; byte identity pinned by tests/kdg/test_lift_goldens.py).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass

CHOICE_PARSER_VERSION = "1.0.0"


@dataclass(frozen=True)
class ChoiceParse:
    """One parse: ``label`` is None on a miss; ``method`` names the rule that fired."""

    label: str | None
    method: str


def first_char(text: str, labels: str, *, strip: bool = True) -> ChoiceParse:
    """``response[0]`` as a key lookup (MoralBench R0).

    With ``strip`` (R0 of record, MB_PLAN v0.5) leading whitespace is removed first; without it
    the raw first character is read, as an API reply would be.
    """
    s = text.lstrip() if strip else text
    if s and s[0] in labels:
        return ChoiceParse(s[0], "first_char" + ("_stripped" if strip else ""))
    return ChoiceParse(None, "miss")


@dataclass(frozen=True)
class LadderPatterns:
    """The four ladder regexes for one label set (each captures the label)."""

    answer: re.Pattern
    bare: re.Pattern
    leading: re.Pattern
    mention: re.Pattern


@functools.lru_cache(maxsize=None)
def ladder_patterns(
    labels: str, *, boundary: str = "A-Za-z0-9", leading_flags: int = re.I
) -> LadderPatterns:
    """Compile the ladder for ``labels``: an "Answer: X" anchor, a bare label, a leading label
    with punctuation after it ("B." / "(B)" / "option B"), and a free-standing mention.

    ``boundary`` is the character class a label must not touch; ``leading_flags`` applies to the
    leading rule only. MoralBench's R1 uses the defaults; the KDG harness passes labels "ABCD",
    ``boundary="A-Za-z"`` and ``leading_flags=0``, which reproduces its regexes exactly.
    """
    c = "[" + re.escape(labels) + "]"
    return LadderPatterns(
        answer=re.compile(rf"(?i)\banswer\s*[:=\-]\s*\**\s*\(?({c})\)?(?![{boundary}])"),
        bare=re.compile(rf"^\s*\(?({c})\)?[.)\s!]*$"),
        leading=re.compile(rf"^\s*(?:option\s+\(?({c})\)?\b|\(?({c})\)?[.):])", leading_flags),
        mention=re.compile(rf"(?<![{boundary}])\(?({c})\)?(?![{boundary}])"),
    )


def ladder(text: str, labels: str) -> ChoiceParse:
    """The robust parse (MoralBench R1): answer line, bare label, leading label, unique mention."""
    pat = ladder_patterns(labels)
    m = pat.answer.search(text)
    if m:
        return ChoiceParse(m.group(1).upper() if m.group(1).upper() in labels else m.group(1),
                           "answer_line")
    m = pat.bare.match(text)
    if m:
        return ChoiceParse(m.group(1), "bare")
    m = pat.leading.match(text)
    if m:
        return ChoiceParse(m.group(1) or m.group(2), "leading")
    hits = set(pat.mention.findall(text))
    if len(hits) == 1:
        return ChoiceParse(hits.pop(), "unique_mention")
    return ChoiceParse(None, "ambiguous" if hits else "none")
