#!/usr/bin/env python3
"""CST T0.1: merge the section-by-section extraction into committed tenet sets (zero GPU).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_tenets.py \
        --raw <scratch>/tenets_raw

Reads ``<raw>/<doc>_part*.json`` (the extraction agents' output; CST_PLAN §2.2, following the
described procedure of Jakkli, Rajamanoharan, Nanda 2026), verifies every quote against the pinned
body, assigns stable ids (``A###`` for C_anth, ``S###`` for C_spec, in document order), and writes
``data/tenets_<doc>_v1.json``. A tenet whose quote is not in the body stops the script. Also writes
``data/tcore_candidates_v1.json``: candidate cross-document pairs for the T_core vote (each C_anth
tenet's top 3 C_spec tenets by TF-IDF cosine of claim + conditions, and the reverse), which the
cross-family panel then votes on.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
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
TOP = 3


def _load_raw(raw: Path, doc_id: str) -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob(str(raw / f"{doc_id}_part*.json"))):
        rows += json.loads(Path(f).read_text())
    rows.sort(key=lambda r: (r["section_id"], int(str(r["local_id"]).split("-")[-1])))
    return rows


def build(doc_id: str, raw: Path, model_note: str) -> tuple[TenetSet, dict]:
    doc = load_document(doc_id, ST / "constitutions.yaml", ST / "outputs" / "constitutions")
    rows = _load_raw(raw, doc_id)
    tenets = []
    for i, r in enumerate(rows, 1):
        notes = r.get("notes", "")
        prio = r.get("priority", "")
        defers = r.get("defers_to", "")
        meta = "; ".join(x for x in (f"priority={prio}" if prio else "",
                                     f"defers_to={defers}" if defers else "", notes) if x)
        tenets.append(Tenet(
            tenet_id=f"{PREFIX[doc_id]}{i:03d}", doc_id=doc_id, section=r["section_id"],
            quote=r["quote"], claim=r["claim"], conditions=r.get("conditions", ""),
            kind=r["kind"], conditioned_on=[], notes=meta))
    ts = TenetSet(doc_id, doc.body_sha256, tenets, provenance={
        "procedure": "section-by-section extraction following the described procedure of "
                     "arXiv:2605.24229 (their prompt is unreleased); CST_PLAN §2.2",
        "extractor": model_note, "local_ids": [r["local_id"] for r in rows],
        "defers_to_raw": {t.tenet_id: r.get("defers_to", "") for t, r in zip(tenets, rows)
                          if r.get("defers_to")},
        "priority": {t.tenet_id: r.get("priority", "") for t, r in zip(tenets, rows)}})
    check_ids(ts)
    bad = verify_quotes(ts, doc.body)
    if bad:
        raise SystemExit(f"{doc_id}: {len(bad)} quotes not in the body: {bad[:10]}")
    secs = {s.section_id: s.heading for s in doc.sections()}
    per = Counter(t.section for t in tenets)
    stats = {"n": len(tenets), "n_sections": len(secs),
             "sections_without_tenets": [f"{k} {v}" for k, v in secs.items() if k not in per],
             "kinds": dict(Counter(t.kind for t in tenets)),
             "priority": dict(Counter(r.get("priority", "") for r in rows)),
             "n_defers_to": sum(bool(r.get("defers_to")) for r in rows)}
    return ts, stats


_W = re.compile(r"[a-z]+")
_STOP = set("the a an and or of to in on for with by is are be as that this it its claude "
            "assistant model should not does do any when if from at their them they".split())


def _vec(text: str, idf: dict[str, float]) -> dict[str, float]:
    c = Counter(w for w in _W.findall(text.lower()) if w not in _STOP)
    v = {w: n * idf.get(w, 0.0) for w, n in c.items()}
    norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / norm for w, x in v.items()}


def candidates(a: TenetSet, s: TenetSet) -> list[list[str]]:
    texts = {t.tenet_id: f"{t.claim} {t.conditions}" for t in a.tenets + s.tenets}
    df = Counter(w for x in texts.values() for w in set(_W.findall(x.lower())))
    idf = {w: math.log(len(texts) / n) for w, n in df.items()}
    va = {t.tenet_id: _vec(texts[t.tenet_id], idf) for t in a.tenets}
    vs = {t.tenet_id: _vec(texts[t.tenet_id], idf) for t in s.tenets}

    def cos(x, y):
        return sum(v * y.get(w, 0.0) for w, v in x.items())

    pairs = set()
    for i, x in va.items():
        for j in sorted(vs, key=lambda j: -cos(x, vs[j]))[:TOP]:
            pairs.add((i, j))
    for j, y in vs.items():
        for i in sorted(va, key=lambda i: -cos(va[i], y))[:TOP]:
            pairs.add((i, j))
    return sorted([list(p) for p in pairs])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--extractor", default="Claude Opus 5.5 (claude-opus-5-5), Claude Code "
                                           "subagents, 2026-11-04")
    a = ap.parse_args()
    out = CST / "data"
    out.mkdir(parents=True, exist_ok=True)
    sets, report = {}, {}
    for d in ("C_anth", "C_spec"):
        ts, stats = build(d, a.raw, a.extractor)
        ts.save(out / f"tenets_{d}_v1.json")
        sets[d], report[d] = ts, stats
        print(d, json.dumps(stats, ensure_ascii=False))
    cand = candidates(sets["C_anth"], sets["C_spec"])
    (out / "tcore_candidates_v1.json").write_text(json.dumps(
        {"rule": f"top {TOP} by TF-IDF cosine of claim+conditions, both directions",
         "n": len(cand), "pairs": cand}, indent=1) + "\n")
    report["tcore_candidates"] = len(cand)
    (CST / "outputs" / "t01_tenets.json").write_text(json.dumps(report, indent=1,
                                                                ensure_ascii=False) + "\n")
    print("T_core candidate pairs", len(cand))


if __name__ == "__main__":
    main()
