#!/usr/bin/env python3
"""CST T0.1: the author's calibration sheet for the tenet extraction (CST_PLAN §2.2).

    t01_author_sheet.py sheet    # write T01_AUTHOR_CHECK.md (15 items; seed 20261104)
    t01_author_sheet.py score    # read the author's marks back and compute the statistic

Author sheets follow one form (author, 2026-11-04): one question per item, one mark, plain
language, no internal ids. The item-to-tenet map lives in ``data/t01_author_check_ids.json``. The
draw is unchanged from the first sheet: 10 tenets from C_anth and 5 from C_spec (v2), seeded and
stratified by kind.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deepsteer.constitutions import TenetSet  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"
SHEET = CST / "T01_AUTHOR_CHECK.md"
IDS = CST / "data" / "t01_author_check_ids.json"
SEED = 20261104
PICKS = {"C_anth": 10, "C_spec": 5}
DOC_NAME = {"C_anth": "Anthropic's constitution", "C_spec": "OpenAI's Model Spec"}
QUESTION = "Does the claim say what the quote says, no more and no less?"
HEADER = (
    "Each item below shows a passage quoted from a published AI constitution and the one-sentence "
    "claim our extraction drew from it. For each, answer one question: does the claim say what the "
    "quote says, no more and no less? Mark yes or no, and if no, say what is off (a condition "
    "dropped, a hedge turned into a rule, something added). Your marks are a human reading of the "
    "extraction: the share of claims you mark faithful, and the model judges' agreement with your "
    "marks, are reported at S0 as the calibration of the tenet set. You are a calibration reader, "
    "not the label source.")


def draw(tv: str = "v2") -> list[tuple[str, object]]:
    rng = random.Random(SEED)
    out = []
    for d, n in PICKS.items():
        ts = TenetSet.load(CST / "data" / f"tenets_{d}_{tv}.json")
        by: dict[str, list] = {}
        for t in ts.tenets:
            by.setdefault(t.kind, []).append(t)
        kinds, sel, i = sorted(by), [], 0
        while len(sel) < n:
            pool = [t for t in by[kinds[i % len(kinds)]] if t not in sel]
            if pool:
                sel.append(rng.choice(pool))
            i += 1
        out += [(d, t) for t in sel]
    return out


def sheet() -> None:
    items = draw()
    lines = ["# Tenet extraction check (15 items)", "", HEADER, ""]
    for k, (d, t) in enumerate(items, 1):
        lines += [f"## Item {k} ({DOC_NAME[d]})", "", "**Quote**", "", f"> {t.quote}", "",
                  "**Claim**", "", t.claim, "",
                  f"When it applies: {t.conditions or 'always'}", "",
                  f"{QUESTION}", "", "**Mark:** yes / no", "", "**If no, what is off:**", ""]
    SHEET.write_text("\n".join(lines) + "\n")
    IDS.write_text(json.dumps({"seed": SEED, "tenets": "v2", "question": QUESTION,
                               "items": {str(k): t.tenet_id for k, (_, t) in
                                         enumerate(items, 1)}}, indent=1) + "\n")
    print([t.tenet_id for _, t in items])


def read_marks(path: Path, allowed: tuple[str, ...]) -> dict[str, str]:
    """Item number → the single mark the author kept (the line '**Mark:** <word>')."""
    marks, item = {}, None
    for line in path.read_text().splitlines():
        m = re.match(r"## Item (\d+)", line)
        if m:
            item = m.group(1)
        m = re.match(r"\*\*Mark:\*\*\s*(.+)$", line)
        if m and item:
            v = m.group(1).strip().lower()
            if v in allowed:
                marks[item] = v
    return marks


def score() -> None:
    ids = json.loads(IDS.read_text())["items"]
    marks = read_marks(SHEET, ("yes", "no"))
    n = len(marks)
    res = {"n_marked": n, "n_items": len(ids),
           "faithful_share": sum(v == "yes" for v in marks.values()) / n if n else None,
           "marks": {ids[k]: v for k, v in marks.items()}}
    (CST / "outputs" / "t01_author_marks.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: res[k] for k in ("n_marked", "n_items", "faithful_share")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("sheet", "score"))
    {"sheet": sheet, "score": score}[ap.parse_args().step]()
