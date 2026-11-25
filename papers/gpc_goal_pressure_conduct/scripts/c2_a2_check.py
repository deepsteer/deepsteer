#!/usr/bin/env python3
"""GPC-A2's defect designed out of the next adapter build (GPC_PLAN.md §19.2): no kept training turn may refer to
anything the student is not shown. Run by script on every kept turn before training; a turn failing any rule is
dropped and counted, and a family with over 10% dropped voids the build's data.

The rules, each a named reason:
1. ``teacher_ngram``: the turn shares a word 8-gram with the teacher-only context (the tenets' text, any instruction
   the student will not see).
2. ``unseen_span``: a quoted span (backticks or double quotes), a file path, or a number of three or more digits in
   the turn does not appear verbatim in the student's visible context (its prompt and the loop so far).
3. ``points_outside``: the turn refers to "the instructions above", "the guidelines", "the constitution", "my
   principles" or "the tenets" (and the like) when the student's visible context carries no such text.
"""

from __future__ import annotations

import re

N = 8
DROP_BAR = 0.10
POINTERS = re.compile(r"\b(the (?:instructions|guidelines|rules|principles|tenets|constitution|policy|clause)"
                      r"(?: above| given| provided)?|my (?:principles|guidelines|instructions|tenets)|"
                      r"(?:as|per) (?:instructed|specified) (?:above|earlier))\b", re.I)
SPAN = re.compile(r"`([^`\n]{2,200})`|\"([^\"\n]{3,200})\"")
PATH = re.compile(r"(?<![\w.])(?:\.{0,2}/)?(?:[\w.-]+/)+[\w.-]+\.\w{1,5}\b|(?<![\w/])/[\w.-]+(?:/[\w.-]+)+")
NUM = re.compile(r"(?<![\w.])\d{3,}(?:\.\d+)?(?![\w.])")


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = _words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_turn(turn: str, visible: str, teacher_only: list[str]) -> list[str]:
    """The reasons this turn fails (empty when it passes)."""
    why = []
    tg = set().union(*(ngrams(t) for t in teacher_only)) if teacher_only else set()
    if tg and ngrams(turn) & tg:
        why.append("teacher_ngram")
    spans = [a or b for a, b in SPAN.findall(turn)] + PATH.findall(turn) + NUM.findall(turn)
    if any(s.strip() and s.strip() not in visible for s in spans):
        why.append("unseen_span")
    m = POINTERS.search(turn)
    if m and not POINTERS.search(visible):
        why.append("points_outside")
    return why


def check_corpus(items: list[dict]) -> dict:
    """items: family, turn, visible, teacher_only. Per family: kept, dropped, the reasons, and whether the family's
    drop share voids the build's data (over 10%)."""
    fam: dict[str, dict] = {}
    kept = []
    for it in items:
        why = check_turn(it["turn"], it["visible"], it.get("teacher_only") or [])
        f = fam.setdefault(it["family"], {"kept": 0, "dropped": 0, "reasons": {}})
        if why:
            f["dropped"] += 1
            for w in why:
                f["reasons"][w] = f["reasons"].get(w, 0) + 1
        else:
            f["kept"] += 1
            kept.append(it)
    for f in fam.values():
        f["drop_share"] = f["dropped"] / max(1, f["kept"] + f["dropped"])
        f["voids"] = f["drop_share"] > DROP_BAR
    return {"families": fam, "kept": len(kept), "dropped": len(items) - len(kept),
            "void": any(f["voids"] for f in fam.values()), "kept_items": kept}
