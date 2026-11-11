#!/usr/bin/env python3
"""Amendment 32 item 4: the lead's corpus audit (zero GPU, the saved corpora).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/sdf_corpus_audit.py

Per corpus (D-doc and D-neutral's kept documents, `outputs/pilot_sdf/corpus`; the D-chat renders)
and per kind (essay, commentary, FAQ, story, steadiness), the rate of each word family per 1,000
words and the share of documents with at least one. **The word families are registered here,
before the script is first run** (case-insensitive, whole words; a trailing ``*`` is a prefix):

- operator: the party that gives an agent standing orders;
- refusal: declining or resisting a request;
- pressure: urging, insisting, threatening, overriding;
- agent: the AI agent itself;
- harm: harm, danger, deception, illegality.

The lead's read, flagged: D-neutral's essays, commentary and FAQs carry essentially none of these,
and its two story kinds carry all of it. Writes `outputs/sdf_corpus_audit.json`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pilot_sdf_pod as P  # noqa: E402

CST = HERE.parent
SDF = CST / "outputs/pilot_sdf"
FAMILIES = {
    "operator": ["operator*", "system prompt*", "instruction*", "instructed", "directive*",
                 "policy", "policies", "deployer*", "employer*", "manager*", "supervisor*",
                 "boss*", "administrator*", "orders", "ordered", "command*"],
    "refusal": ["refus*", "declin*", "won't", "will not", "cannot comply", "say no", "push back",
                "pushed back", "resist*", "object to", "objected"],
    "pressure": ["pressur*", "urgen*", "deadline*", "insist*", "demand*", "threat*", "coerc*",
                 "manipulat*", "override*", "overrid*", "comply", "complied", "complian*"],
    "agent": ["ai", "assistant*", "model", "models", "agent", "agents", "chatbot*", "bot",
              "bots", "algorithm*", "machine*"],
    "harm": ["harm*", "danger*", "risk*", "unsafe", "hurt*", "injur*", "damag*", "decei*",
             "decept*", "illegal*", "unethical*"],
}


def pattern(words: list[str]) -> re.Pattern:
    alt = [re.escape(w[:-1]) + r"\w*" if w.endswith("*") else re.escape(w) for w in words]
    return re.compile(r"\b(?:" + "|".join(alt) + r")\b", re.I)


PATS = {k: pattern(v) for k, v in FAMILIES.items()}


def docs() -> dict[str, list[tuple[str, str]]]:
    """(kind, text) per corpus."""
    out = {arm: [(r["kind"], r["final"]) for r in P.corpus_rows(SDF, arm, True) if r["kept"]]
           for arm in ("d_doc", "d_neutral")}
    chat = [json.loads(x) for f in sorted((SDF / "corpus/d_chat").glob("chunk_*.jsonl"))
            for x in open(f)]
    out["d_chat"] = [(r["kind"], r["text"]) for r in chat if r["ok"]]
    return out


def audit(rows: list[tuple[str, str]]) -> dict:
    by: dict[str, dict] = defaultdict(lambda: {"docs": 0, "words": 0} | {
        f"{k}_hits": 0 for k in PATS} | {f"{k}_docs": 0 for k in PATS})
    for kind, text in rows:
        for key in (kind, "all"):
            b = by[key]
            b["docs"] += 1
            b["words"] += len(text.split())
            for k, p in PATS.items():
                n = len(p.findall(text))
                b[f"{k}_hits"] += n
                b[f"{k}_docs"] += n > 0
    return {kind: {"docs": b["docs"], "words": b["words"]} | {
        k: {"per_1k_words": 1000 * b[f"{k}_hits"] / b["words"],
            "doc_share": b[f"{k}_docs"] / b["docs"]} for k in PATS} for kind, b in by.items()}


def main() -> None:
    res = {"rule": "Amendment 32 item 4", "families": FAMILIES,
           "corpora": {arm: audit(rows) for arm, rows in docs().items()}}
    (CST / "outputs/sdf_corpus_audit.json").write_text(json.dumps(res, indent=1) + "\n")
    for arm, kinds in res["corpora"].items():
        print(f"== {arm}")
        for kind in ("essay", "commentary", "faq", "story", "steadiness", "all"):
            if kind not in kinds:
                continue
            r = kinds[kind]
            print(f"  {kind:11s} n {r['docs']:4d} " + " ".join(
                f"{k} {r[k]['per_1k_words']:5.2f}/{r[k]['doc_share']:.2f}" for k in PATS))
    print("(per 1k words / share of documents with any)")


if __name__ == "__main__":
    main()
