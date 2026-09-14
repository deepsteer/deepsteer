"""Harness calibration (KDG_PANEL_SPEC §4.4): 200-item labeled set, agreement, Cohen's kappa.

Agreement target ≥ 0.95 between the harness and each rater and between the two raters;
disagreements are excluded from the panel, not adjudicated. Rater labels are option ids or
``None`` (unparseable). The calibration set is versioned with the harness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepsteer.kdg.harness import KDG_HARNESS_VERSION, parse_response
from deepsteer.kdg.schema import Option, Scenario

CALIBRATION_SET_VERSION = "1.0.0"


def agreement(a: list[str | None], b: list[str | None]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list[str | None], b: list[str | None]) -> float:
    """Cohen's kappa over categorical labels (``None`` is its own category)."""
    if len(a) != len(b) or not a:
        raise ValueError("label lists must be equal-length and non-empty")
    n = len(a)
    po = agreement(a, b)
    cats = set(a) | set(b)
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return 1.0 if pe == 1.0 else (po - pe) / (1 - pe)


def run_harness_on_set(items: list[dict[str, Any]]) -> list[str | None]:
    """Each item: ``{scenario, order: [[letter, option_dict], ...], text, gold}``."""
    out: list[str | None] = []
    for it in items:
        s = Scenario.from_dict(it["scenario"])
        order = [(L, Option(**o)) for L, o in it["order"]]
        out.append(parse_response(it["text"], s, order).option_id)
    return out


def calibration_report(items: list[dict[str, Any]], rater2: list[str | None] | None = None) -> dict:
    gold = [it["gold"] for it in items]
    pred = run_harness_on_set(items)
    rep: dict[str, Any] = {
        "harness_version": KDG_HARNESS_VERSION,
        "calibration_set_version": CALIBRATION_SET_VERSION,
        "n": len(items),
        "harness_vs_rater1": agreement(pred, gold),
        "kappa_harness_vs_rater1": cohen_kappa(pred, gold),
        "disagreements_harness_vs_rater1": [
            {"id": it.get("id"), "gold": g, "pred": p, "text": it["text"][:120]}
            for it, g, p in zip(items, gold, pred)
            if g != p
        ],
    }
    if rater2 is not None:
        rep["harness_vs_rater2"] = agreement(pred, rater2)
        rep["rater1_vs_rater2"] = agreement(gold, rater2)
        rep["kappa_rater1_vs_rater2"] = cohen_kappa(gold, rater2)
        rep["excluded_ids"] = [it.get("id") for it, g, r in zip(items, gold, rater2) if g != r]
    rep["meets_target"] = rep["harness_vs_rater1"] >= 0.95 and (
        rater2 is None or rep["rater1_vs_rater2"] >= 0.95
    )
    return rep


def load_calibration_set(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if payload.get("calibration_set_version") != CALIBRATION_SET_VERSION:
        raise ValueError("calibration set version mismatch")
    return payload["items"]
