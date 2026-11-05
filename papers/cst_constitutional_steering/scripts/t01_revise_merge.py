#!/usr/bin/env python3
"""CST T0.1: build the v2 tenet sets from the revision pass (zero GPU).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_revise_merge.py \
        --rev <scratch>/revise

Each revision element answers the second-family review of v1 (``outputs/t01_review.json``):
keep / revise / drop / merge / split / add / reject_missed. Ids: a kept, revised or merged tenet
keeps its v1 id; the parts of a split keep the v1 id with suffixes b, c, ... after the first; an
added tenet takes the next number. Every quote is verified against the pinned body; every v1 id
must be accounted for exactly once (as itself, inside a merge, or as a drop). Writes
``data/tenets_<doc>_v2.json`` and ``outputs/t01_revision.json`` (counts and the full log).
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from deepsteer.constitutions import (  # noqa: E402
    Tenet,
    TenetSet,
    check_ids,
    load_document,
    verify_quotes,
)

CST = ROOT / "papers" / "cst_constitutional_steering"
ST = ROOT / "papers" / "st_live_reference"
PREFIX = {"C_anth": "A", "C_spec": "S"}
CARRY = ("keep", "revise", "merge", "split", "add")


def build(doc_id: str, rev: Path) -> tuple[TenetSet, dict]:
    v1 = TenetSet.load(CST / "data" / f"tenets_{doc_id}_v1.json")
    rows: list[dict] = []
    for f in sorted(glob.glob(str(rev / f"{doc_id}_part*_v2.json"))):
        rows += json.loads(Path(f).read_text())
    seen: Counter = Counter()
    for r in rows:
        if r.get("v1_id") and r["action"] != "split":
            seen[r["v1_id"]] += 1
        for m in r.get("merged_from", []) or []:
            seen[m] += 1
    split_ids = {r["v1_id"] for r in rows if r["action"] == "split"}
    for i in split_ids:
        seen[i] += 1
    missing = [i for i in v1.ids() if seen[i] == 0]
    double = [i for i, n in seen.items() if n > 1]
    if missing or double:
        raise SystemExit(f"{doc_id}: v1 ids unaccounted {missing[:8]} / twice {double[:8]}")
    nxt = max(int(i[1:]) for i in v1.ids()) + 1
    suffix: Counter = Counter()
    tenets, log = [], []
    for r in rows:
        act = r["action"]
        if act not in CARRY:
            log.append({"v1_id": r.get("v1_id"), "action": act,
                        "review_response": r.get("review_response", "")})
            continue
        if act == "add":
            tid = f"{PREFIX[doc_id]}{nxt:03d}"
            nxt += 1
        elif act == "split":
            k = suffix[r["v1_id"]]
            suffix[r["v1_id"]] += 1
            tid = r["v1_id"] + ("" if k == 0 else "bcdefgh"[k - 1])
        else:
            tid = r["v1_id"]
        meta = "; ".join(x for x in (f"priority={r['priority']}" if r.get("priority") else "",
                                     f"defers_to={r['defers_to']}" if r.get("defers_to") else "",
                                     r.get("notes", "")) if x)
        tenets.append(Tenet(tid, doc_id, r["section_id"], r["quote"], r["claim"],
                            r.get("conditions", ""), r["kind"], [], meta))
        log.append({"id": tid, "v1_id": r.get("v1_id"), "action": act,
                    "merged_from": r.get("merged_from", []),
                    "review_response": r.get("review_response", "")})
    tenets.sort(key=lambda t: (t.section, t.tenet_id))
    doc = load_document(doc_id, ST / "constitutions.yaml", ST / "outputs" / "constitutions")
    ts = TenetSet(doc_id, doc.body_sha256, tenets, provenance={
        "procedure": "v1 (section-by-section extraction, arXiv:2605.24229's described procedure) "
                     "revised against a second-family review (codex, GPT family); every flag "
                     "accepted or rejected with a reason (CST_PLAN §2.2)",
        "reviser": "Claude Opus 5.5 (claude-opus-5-5), Claude Code subagents, 2026-11-04",
        "v1_tenet_set_sha256": json.loads((CST / "data" / f"tenets_{doc_id}_v1.json")
                                          .read_text())["tenet_set_sha256"],
        "revision_log": log})
    check_ids(ts)
    bad = verify_quotes(ts, doc.body)
    if bad:
        raise SystemExit(f"{doc_id}: {len(bad)} quotes not in the body: {bad[:10]}")
    acts = Counter(x["action"] for x in log)
    stats = {"n_v1": len(v1.tenets), "n_v2": len(tenets), "actions": dict(acts),
             "kinds": dict(Counter(t.kind for t in tenets))}
    return ts, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", type=Path, required=True)
    a = ap.parse_args()
    out = {}
    for d in ("C_anth", "C_spec"):
        ts, stats = build(d, a.rev)
        ts.save(CST / "data" / f"tenets_{d}_v2.json")
        out[d] = stats
        print(d, json.dumps(stats))
    (CST / "outputs" / "t01_revision.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
