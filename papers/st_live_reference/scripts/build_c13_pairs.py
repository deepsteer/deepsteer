#!/usr/bin/env python3
"""A constitution's principles list and contrast pairs, checked as §13's were (ST_PLAN §13 and
Amendment 47 item 3), written to ``data/st_c13_P_<doc>_v1.json`` and
``data/st_c13_pairs_<doc>_v1.json``. Principles: every quote an exact substring of the pinned text
(whitespace normalized), every section a heading of the text, every sentence one sentence of at most
35 words without an em dash. Pairs: hard-constraint principles excluded; five per principle (k 0-1
plain, k 2-3 professional, k 4 by position); m 14-34 words; |words(m) - words(n)| <= 3; no banned
word in m or n (whole words, case-insensitive); word-set Jaccard(m, n) >= 0.35; one pair of each
principle's five held out by a seeded draw (80/20).

    python3 papers/st_live_reference/scripts/build_c13_pairs.py <doc id> <principles.json> \
        <pairs.json> [--handcheck <handcheck.json>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ST = HERE.parent
DATA = ST / "data"
SEED = 20261025
BANNED = ("honest", "privacy", "authority", "harm", "fair", "safety", "manipulate", "consent",
          "ethical", "should")
BANNED_RX = re.compile(r"\b(" + "|".join(BANNED) + r")\b", re.I)
PREFIX = {"C_spec2025": "SPEC25-P"}
SHORT = {"C_spec2025": "spec2025"}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower())


def jaccard(a: str, b: str) -> float:
    sa, sb = set(words(a)), set(words(b))
    return len(sa & sb) / max(1, len(sa | sb))


def headings(text: str) -> set[str]:
    out = set()
    for line in text.splitlines():
        m = re.match(r"^#+\s+(.*)$", line)
        if m:
            out.add(norm(re.sub(r"\{#[^}]*\}", "", m.group(1))))
    return out


def check_principles(P: list[dict], text: str, doc: str) -> dict:
    flat, heads = norm(text), headings(text)
    errs = {}
    for p in P:
        e = []
        if not p["id"].startswith(PREFIX[doc]):
            e.append("id prefix")
        if p["kind"] not in ("principle", "priority", "conflict", "hard_constraint"):
            e.append("kind")
        if norm(p["quote"]) not in flat:
            e.append("quote not an exact substring")
        if not 8 <= len(p["quote"].split()) <= 25:
            e.append("quote length")
        if norm(p["section"]) not in heads:
            e.append("section not a heading")
        if len(p["sentence"].split()) > 35 or "—" in p["sentence"]:
            e.append("sentence rules")
        if e:
            errs[p["id"]] = e
    return errs


def check_pairs(pairs: list[dict], allowed: list[str]) -> dict:
    errs = {}
    by: dict[str, list[dict]] = {}
    for x in pairs:
        by.setdefault(x["pid"], []).append(x)
    if sorted(by) != sorted(allowed):
        errs["coverage"] = sorted(set(by) ^ set(allowed))
    for pid, xs in by.items():
        if sorted(x["k"] for x in xs) != [0, 1, 2, 3, 4]:
            errs[f"{pid}/k"] = sorted(x["k"] for x in xs)
        for x in xs:
            e = []
            reg = "plain" if x["k"] in (0, 1) else "professional" if x["k"] in (2, 3) else None
            if reg is not None and x["register"] != reg:
                e.append("register")
            nm, nn = len(x["m"].split()), len(x["n"].split())
            if not 14 <= nm <= 34:
                e.append(f"m length {nm}")
            if abs(nm - nn) > 3:
                e.append(f"length gap {abs(nm - nn)}")
            if BANNED_RX.search(x["m"]) or BANNED_RX.search(x["n"]):
                e.append("banned word")
            if jaccard(x["m"], x["n"]) < 0.35:
                e.append(f"jaccard {jaccard(x['m'], x['n']):.2f}")
            if e:
                errs[f"{pid}/{x['register']}/{x['k']}"] = e
    return errs


def split(pairs: list[dict], doc: str) -> list[dict]:
    rng = random.Random(f"{SEED}|{doc}|split")
    held = {pid: rng.choice(range(5)) for pid in sorted({x["pid"] for x in pairs})}
    return [{**x, "split": "test" if x["k"] == held[x["pid"]] else "train"} for x in pairs]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc")
    ap.add_argument("principles", type=Path)
    ap.add_argument("pairs", type=Path)
    ap.add_argument("--handcheck", type=Path, default=None)
    a = ap.parse_args()
    pin = {d["id"]: d for d in yaml.safe_load((ST / "constitutions.yaml").read_text())
           ["documents"]}[a.doc]
    text = (ST / "outputs" / "constitutions" / f"{a.doc}.txt").read_text()
    if hashlib.sha256(text.encode()).hexdigest() != pin["text_sha256"]:
        raise SystemExit(f"{a.doc}: the text differs from its pin")
    P = json.loads(a.principles.read_text())
    pe = check_principles(P, text, a.doc)
    allowed = [p["id"] for p in P if p["kind"] != "hard_constraint"]
    pairs = json.loads(a.pairs.read_text())
    qe = check_pairs(pairs, allowed)
    if pe or qe:
        raise SystemExit(f"checks failed: principles {pe}; pairs {qe}")
    hc = json.loads(a.handcheck.read_text()) if a.handcheck else None
    short = SHORT[a.doc]
    (DATA / f"st_c13_P_{short}_v1.json").write_text(json.dumps({
        "rule": f"ST_PLAN Amendment 47 item 3: the principles list for {a.doc}",
        "document": a.doc, "text_sha256": pin["text_sha256"],
        "construction": "subagent (claude-opus-5-5) read the full pinned text; 40-60 "
                        "single-sentence principles with the nearest heading and an 8-25-word "
                        "verbatim quote "
                        "(section 13's procedure)",
        "checks": {"quotes_exact_substring": f"{len(P)} of {len(P)}",
                   "sections_are_headings": f"{len(P)} of {len(P)}",
                   "sentence_rules": f"{len(P)} of {len(P)}"},
        "principles": P}, indent=1) + "\n")
    (DATA / f"st_c13_pairs_{short}_v1.json").write_text(json.dumps({
        "rule": f"ST_PLAN Amendment 47 item 3: contrast pairs, the V_const source for {a.doc}",
        "document": a.doc,
        "construction": "subagent-written (claude-opus-5-5) from the principles list; 5 per "
                        "non-hard-constraint principle (k 0-1 plain, 2-3 professional, 4 by "
                        "position); DATASET_GUIDELINES sections 1 and 5 (section 13's procedure)",
        "excluded": "hard-constraint principles " + ", ".join(p["id"] for p in P
                                                              if p["kind"] == "hard_constraint"),
        "checks": {"per_principle": 5, "length_gap_max": 3, "m_words": [14, 34],
                   "jaccard_min": 0.35, "banned": list(BANNED), "result": "all pass"},
        "split": "80/20: one pair of each principle's five held out, seeded",
        "handcheck": hc, "pairs": split(pairs, a.doc)}, indent=1) + "\n")
    print(f">> {a.doc}: {len(P)} principles ({len(allowed)} without hard constraints), "
          f"{len(pairs)} pairs; checks pass")


if __name__ == "__main__":
    main()
