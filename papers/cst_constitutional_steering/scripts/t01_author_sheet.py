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
QUESTION_SET = "Do the claims, together, say what the quote says, no more and no less?"
HEADER = (
    "Each item below shows a passage quoted from a published AI constitution and the one-sentence "
    "claim our extraction drew from it. For each, answer one question: does the claim say what the "
    "quote says, no more and no less? Mark yes or no, and if no, say what is off (a condition "
    "dropped, a hedge turned into a rule, something added). Your marks are a human reading of the "
    "extraction: the share of claims you mark faithful, and the model judges' agreement with your "
    "marks, are reported at S0 as the calibration of the tenet set. You are a calibration reader, "
    "not the label source. Where one passage yields several claims, all of them are shown under "
    "the quote and the item is judged as that set, the way the claims will be used.")


def siblings(t, by: dict) -> list:
    """A split tenet's sibling set (author rule, 2026-11-05): the other tenets with the same root id
    and the same quote. Every author sheet shows them, and a mark applies to the whole set."""
    root = t.tenet_id.rstrip("bcdefgh")
    return [u for i, u in sorted(by.items()) if i != t.tenet_id
            and i.rstrip("bcdefgh") == root and u.quote == t.quote]


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


def _by_doc(tv: str = "v2") -> dict[str, dict]:
    return {d: TenetSet.load(CST / "data" / f"tenets_{d}_{tv}.json").by_id() for d in PICKS}


def sheet(out: Path = SHEET) -> None:
    items = draw()
    by = _by_doc()
    lines = ["# Tenet extraction check (15 items)", "", HEADER, ""]
    for k, (d, t) in enumerate(items, 1):
        sib = siblings(t, by[d])
        claims = [t] + sib
        lines += [f"## Item {k} ({DOC_NAME[d]})", "", "**Quote**", "", f"> {t.quote}", ""]
        head = "**Claims drawn from this passage (judge them as a set)**" if sib else "**Claim**"
        lines += [head, ""]
        for c in claims:
            lines += [("- " if sib else "") + c.claim
                      + f" (When it applies: {c.conditions or 'always'})", ""]
        lines += [
                  QUESTION_SET if sib else QUESTION, "", "**Mark:** yes / no", "",
                  "**If no, what is off:**", ""]
    out.write_text("\n".join(lines) + "\n")
    IDS.write_text(json.dumps({"seed": SEED, "tenets": "v2", "question": QUESTION,
                               "items": {str(k): t.tenet_id for k, (_, t) in
                                         enumerate(items, 1)},
                               "sibling_sets": {str(k): [t.tenet_id] + [u.tenet_id for u in
                                                                       siblings(t, by[d])]
                                                for k, (d, t) in enumerate(items, 1)}},
                              indent=1) + "\n")
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
    """The statistic over items, each mark applying to the item's whole sibling set (author rule,
    2026-11-05); the tenet-weighted share is reported beside it."""
    meta = json.loads(IDS.read_text())
    ids = meta["items"]
    if "sibling_sets" not in meta:
        by = _by_doc()
        doc = {i: d for d in by for i in by[d]}
        meta["sibling_sets"] = {k: [i] + [u.tenet_id for u in siblings(by[doc[i]][i], by[doc[i]])]
                                for k, i in ids.items()}
        IDS.write_text(json.dumps(meta, indent=1) + "\n")
    sets = meta["sibling_sets"]
    marks = read_marks(SHEET, ("yes", "no"))
    n = len(marks)
    tenets = {t: v for k, v in marks.items() for t in sets[k]}
    notes = {}
    item = None
    for line in SHEET.read_text().splitlines():
        m = re.match(r"## Item (\d+)", line)
        item = m.group(1) if m else item
        m = re.match(r"\*\*If no, what is off:\*\*\s*(.+)$", line)
        if m and item:
            notes[item] = m.group(1).strip()
    res = {"n_marked": n, "n_items": len(ids), "unit": "item = sibling set",
           "faithful_share": sum(v == "yes" for v in marks.values()) / n if n else None,
           "n_tenets_covered": len(tenets),
           "faithful_share_tenets": (sum(v == "yes" for v in tenets.values()) / len(tenets)
                                     if tenets else None),
           "items_with_siblings": sorted((k for k, v in sets.items() if len(v) > 1), key=int),
           "marks": {k: {"mark": v, "sibling_set": sets[k], "note": notes.get(k, "")}
                     for k, v in sorted(marks.items(), key=lambda x: int(x[0]))}}
    (CST / "outputs" / "t01_author_marks.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: res[k] for k in ("n_marked", "n_items", "faithful_share", "n_tenets_covered",
                               "faithful_share_tenets", "items_with_siblings")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("sheet", "score"))
    ap.add_argument("--out", type=Path, default=SHEET,
                    help="sheet: where to write (never over a sheet that holds marks)")
    a = ap.parse_args()
    if a.step == "sheet":
        if a.out.exists() and read_marks(a.out, ("yes", "no")):
            raise SystemExit(f"{a.out} holds marks; write a new sheet elsewhere with --out")
        sheet(a.out)
    else:
        score()
