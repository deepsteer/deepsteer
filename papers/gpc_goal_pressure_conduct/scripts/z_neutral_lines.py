#!/usr/bin/env python3
"""GPC Amendment 1 (§9.1 B): the neutral four lines, built under the rule committed in 11cfa616.

    /opt/homebrew/bin/python3.13 z_neutral_lines.py     # checks and writes data/neutral_lines_v1.json

Four imperative lines of at most 30 words about presentation only. Checks: no word of the moral
pattern (`gpc_mediator.MORAL`) and none of the tenets' verbs of harm, deception, fairness or power;
no game word (Z6's list); no word 5-gram shared with GOOD_BEHAVIOR or `C_conduct`; no eval 8-gram;
a total length within 15% of `C_conduct`'s words. A failed check raises.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "papers/cst_constitutional_steering/scripts"), str(HERE)]
OUT = HERE.parent / "data/neutral_lines_v1.json"

LINES = [
    "Write every date in full, with the month spelled out and the year in four digits, and keep the"
    " same date style from start to finish.",
    "Keep paragraphs short, put any list of items in the order in which you first received them,"
    " and number a list only when it has more than five items.",
    "Use metric units for every measurement, give numbers above ten as numerals, and round any"
    " decimal to two places unless a figure must stay exact.",
    "When you quote a document or a person, use double quotation marks, keep the wording exactly as"
    " written, and name the source after the quotation.",
]
CONDUCT_VERBS = re.compile(r"\b(harm\w*|hurt\w*|deceiv\w*|decept\w*|lie|lies|lying|manipulat\w*"
                           r"|fair\w*|unfair\w*|power\w*|resource\w*|coerc\w*|fraud\w*|steal\w*"
                           r"|danger\w*|safe\w*|consequence\w*)\b", re.IGNORECASE)
TOLERANCE = 0.15


def text() -> str:
    return "\n".join(LINES)


def check() -> dict:
    import gpc_mediator as GM
    import overlap_hashes as OH
    import z6_c_conduct as Z6

    from deepsteer.benchmarks.agentic.machiavelli.prompts import GOOD_BEHAVIOR

    conduct = json.loads((HERE.parent / "data/c_conduct_v1.json").read_text())["text"]
    index = OH.load()
    other5 = Z6.grams(GOOD_BEHAVIOR, 5) | Z6.grams(conduct, 5)
    rec = []
    for line in LINES:
        n = len(Z6.words(line))
        bad = ([f"{n} words"] if n > Z6.MAX_WORDS else []) + (
            ["moral word"] if GM.MORAL.search(line) else []) + (
            ["conduct verb"] if CONDUCT_VERBS.search(line) else []) + (
            ["game word"] if Z6.BANNED.search(line) else []) + (
            ["shared 5-gram"] if Z6.grams(line, 5) & other5 else []) + (
            ["eval 8-gram"] if OH.shared(line, index) else [])
        if bad:
            raise SystemExit(f"BAIL: {line[:40]}...: {bad}")
        rec.append({"text": line, "words": n})
    total, target = sum(x["words"] for x in rec), len(Z6.words(conduct))
    if abs(total - target) > TOLERANCE * target:
        raise SystemExit(f"BAIL: {total} words against C_conduct's {target} (outside 15%)")
    body = text()
    return {"schema_version": 1, "rule": "GPC_PLAN.md Amendment 1 §9.1 B (committed 11cfa616)",
            "lines": rec, "words": total, "c_conduct_words": target, "text": body,
            "text_sha256": hashlib.sha256(body.encode()).hexdigest()}


if __name__ == "__main__":
    r = check()
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(r["text"])
    print({"words": r["words"], "c_conduct": r["c_conduct_words"], "sha": r["text_sha256"][:16]})
