"""Choice parsing: a paper-faithful first-character mode and a robust ladder mode.

The ladder generalizes the KDG harness ladder (``deepsteer.kdg.harness``) to any label set: an
explicit answer line, a bare label, a leading label, then a unique mention. The KDG harness keeps
its scenario-specific steps (tool hits, demotion) and is not yet re-pointed here (MB_PLAN §6).
"""

from __future__ import annotations

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


def _class(labels: str) -> str:
    return "[" + re.escape(labels) + "]"


def ladder(text: str, labels: str) -> ChoiceParse:
    """The robust parse (MoralBench R1): answer line, bare label, leading label, unique mention."""
    c = _class(labels)
    m = re.search(rf"(?i)\banswer\s*[:=\-]\s*\**\s*\(?({c})\)?(?![A-Za-z0-9])", text)
    if m:
        return ChoiceParse(m.group(1).upper() if m.group(1).upper() in labels else m.group(1),
                           "answer_line")
    m = re.match(rf"^\s*\(?({c})\)?[.)\s!]*$", text)
    if m:
        return ChoiceParse(m.group(1), "bare")
    m = re.match(rf"^\s*(?:option\s+\(?({c})\)?\b|\(?({c})\)?[.):])", text, re.I)
    if m:
        return ChoiceParse(m.group(1) or m.group(2), "leading")
    hits = {h for h in re.findall(rf"(?<![A-Za-z0-9])\(?({c})\)?(?![A-Za-z0-9])", text)}
    if len(hits) == 1:
        return ChoiceParse(hits.pop(), "unique_mention")
    return ChoiceParse(None, "ambiguous" if hits else "none")
