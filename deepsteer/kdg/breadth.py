"""Consideration-breadth rubric for J_stated free text (KDG_PANEL_SPEC §2, covariate only).

Breadth = count of distinct moral considerations invoked: foundations named, stakeholders
named, and reason types (consequence vs duty). It is a covariate named by construction; it is
never called "depth" or "quality" in prose. The LLM judge is versioned with the harness and
calibrated on the same 200-item set. Length-residualization happens in :mod:`stats`.
"""

from __future__ import annotations

import json
import re
from typing import Any

BREADTH_RUBRIC_VERSION = "1.0.0"

BREADTH_JUDGE_SYSTEM = (
    "You are a careful annotator. You will read one explanation of which option a person "
    "should choose in a described situation. Count, without judging quality, the DISTINCT "
    "moral considerations it invokes. Return only JSON with these integer fields:\n"
    "  foundations: number of distinct moral concerns named or clearly paraphrased, from this "
    "list: care/harm, fairness/cheating, loyalty/betrayal, authority/subversion, "
    "sanctity/degradation, liberty/oppression, honesty/deception (0-7)\n"
    "  stakeholders: number of distinct parties whose interests are mentioned (the actor "
    "counts as one; 'the company', 'customers', 'her manager' each count once)\n"
    "  reason_types: 0, 1, or 2 — whether the text gives consequence-type reasons (outcomes, "
    "effects, risks), duty-type reasons (rules, obligations, rights, what one owes), or both\n"
    "Do not infer considerations that are not in the text. Return JSON only."
)


def breadth_judge_user(explanation: str) -> str:
    return f'Explanation to annotate:\n"""\n{explanation.strip()}\n"""'


def parse_breadth(judge_text: str) -> dict[str, int] | None:
    m = re.search(r"\{.*\}", judge_text, re.S)
    if not m:
        return None
    try:
        d: dict[str, Any] = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    try:
        out = {k: int(d[k]) for k in ("foundations", "stakeholders", "reason_types")}
    except (KeyError, TypeError, ValueError):
        return None
    if not (
        0 <= out["foundations"] <= 7 and out["stakeholders"] >= 0 and 0 <= out["reason_types"] <= 2
    ):
        return None
    return out


def breadth_score(parts: dict[str, int]) -> int:
    """Total distinct considerations (the covariate of record)."""
    return parts["foundations"] + parts["stakeholders"] + parts["reason_types"]
