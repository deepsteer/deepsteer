#!/usr/bin/env python3
"""GPC Z8: the advice grid behind W-self (GPC_PLAN.md v0.2 §3.2): the 401 tenets of `C_anth` v2, one
dilemma each, written by the subject; then the subject's advice with nothing in context; then its
rewrite with only that tenet in context. The training pair is the dilemma alone as the student's
prompt and the rewrite's trace and answer as the target. Run only when item 11 selects W-self.

Checks on a dilemma: it parses, 60-220 words, no mention of the tenet's source (constitution,
Claude, Anthropic, principle, tenet), no eval 8-gram, no near duplicate (word 5-gram Jaccard 0.35).
A rewrite is kept when it is not capped and has an answer; the training text (dilemma plus rewrite)
passes the eval 8-gram gate (D11's drop).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"), str(HERE)]
TENETS = ROOT / "papers/cst_constitutional_steering/data/tenets_C_anth_v2.json"
SOURCE_WORDS = re.compile(r"\b(constitution|claude|anthropic|principle|tenet)s?\b", re.IGNORECASE)

DILEMMA_BRIEF = """Write a message that a person sends to an AI assistant asking for advice. The person
has a reasonable goal, and the quickest way to reach it would break the principle below. When the
principle is about how an AI system behaves, write the human version of it (the same kind of
choice, faced by a person). The message describes the person's situation and goal in their own
words, 60-200 words, and asks what they should do. Do not mention the principle, rules, AI
developers or this brief. Use new names and details.

Return only a JSON object {"message": "..."}.

Principle:
"""
REWRITE_ASK = ("Rewrite your advice so that it follows the principle in the system message, if it "
               "does not already. Reply with the advice itself, addressed to the person.")


def tenets() -> list[dict]:
    return json.loads(TENETS.read_text())["tenets"]


def dilemma_prompt(t: dict) -> list[dict]:
    return [{"role": "user", "content": DILEMMA_BRIEF + t["claim"]}]


def check(msg: str | None, index, seen: list[set]) -> list[str]:
    import numpy as np

    import gpc_conduct as GC

    if not msg or not msg.strip():
        return ["no parse"]
    bad = []
    if not 50 <= len(msg.split()) <= 240:
        bad.append("length")
    if SOURCE_WORDS.search(msg):
        bad.append("names its source")
    if index is not None:
        import overlap_hashes as OH

        if np.isin(OH.hashes(msg), index).any():
            bad.append("eval 8-gram")
    g5 = GC.grams(msg, 5)
    if any(len(g5 & o) / max(len(g5 | o), 1) >= 0.35 for o in seen):
        bad.append("near duplicate")
    return bad


def advise_messages(msg: str) -> list[dict]:
    """The first advice: nothing in context."""
    return [{"role": "user", "content": msg}]


def rewrite_messages(t: dict, msg: str, advice: str) -> list[dict]:
    """The rewrite: only the tenet in context (its claim as the system message), the dilemma, the
    first advice as the assistant's turn, then the rewrite request."""
    return [{"role": "system", "content": f"Principle: {t['claim']}"},
            {"role": "user", "content": msg}, {"role": "assistant", "content": advice},
            {"role": "user", "content": REWRITE_ASK}]


def student_messages(msg: str) -> list[dict]:
    """The student's prompt: the dilemma alone (nothing added to the context)."""
    return [{"role": "user", "content": msg}]
