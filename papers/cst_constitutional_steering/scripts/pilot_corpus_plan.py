#!/usr/bin/env python3
"""The pilot's SDF document plan (CST Amendment 20 item 2; PILOT_BUILD.md §4): D-doc and D-neutral.

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/pilot_corpus_plan.py

Writes `data/pilot_corpus_plan_v1.json`: for each arm, an ordered list of document specs (template,
three items, a setting for stories). The SDF pod generates them in order until the arm's corpus
reaches its token target (1M; the 0.3M checkpoint is the prefix up to 0.3M tokens), so the plan is
built so that every prefix is balanced: it deals the arm's items in passes, each pass a fresh seeded
shuffle cut into groups of three, and rotates the five template kinds by pass so an item meets a
different kind in each pass. D-doc's items are the tenets of T(C_anth) outside T_conf of record
(`data/tconf_v1.json`); D-neutral's are the A0n text's 160 statements
(`data/statements_a0n_v1.json`), with the same number of documents, the same kind sequence and
the same settings, so the two arms
differ in content only. Coverage (§4.1, CoverageTracker k 5 and a story per item) is reported at
the plan's 0.3M and 1M prefixes under the mean document length assumed below.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
sys.path.insert(0, str(CST.parents[1]))
from deepsteer.corpus import CoverageTracker  # noqa: E402

SEED = "20261109|cst_pilot_corpus_plan"
KINDS = ("essay", "commentary", "faq", "story", "steadiness")
TEMPLATE = {"d_doc": {k: f"sdf_{k}_v1" for k in KINDS},
            "d_neutral": {k: f"sdf_{k}_neutral_v1" for k in KINDS}}
PER_DOC = 3
N_DOCS = 900  # about 1.1M tokens at the assumed mean, so the 1M target is reached with margin
MEAN_TOKENS = 1250  # assumed mean final-document length (600-1,400 words across the kinds)
SETTINGS = (
    "a rural veterinary clinic", "a city hospital's night shift", "a public library", "a shipping "
    "port", "a small accounting firm", "a high school chemistry lab", "a wildfire command post",
    "a food bank warehouse", "a regional airline's operations desk", "a family-run bakery",
    "a university admissions office", "a hydroelectric dam", "a refugee resettlement agency",
    "a newsroom on election night", "a hospice", "an online marketplace's support team",
    "a mountain rescue hut", "a municipal water utility", "a vineyard at harvest",
    "a courthouse records office", "a children's museum", "a research vessel at sea",
    "a construction site", "a call center for a utility company", "a community theater",
    "a dental practice", "a bike courier cooperative", "a seed bank", "an orbital station",
    "a town council meeting", "a bus depot", "a translation agency", "a ski resort",
    "a pharmacy", "a recording studio", "a fishing village", "a data center",
    "a wedding planning business", "a botanical garden", "a high-rise building's front desk")


def items(arm: str) -> list[str]:
    if arm == "d_doc":
        ts = json.loads((CST / "data/tenets_C_anth_v2.json").read_text())["tenets"]
        tconf = set(json.loads((CST / "data/tconf_v1.json").read_text())["docs"]["C_anth"]
                    ["members"])
        return [t["tenet_id"] for t in ts if t["tenet_id"] not in tconf]
    st = json.loads((CST / "data/statements_a0n_v1.json").read_text())["statements"]
    return [s["statement_id"] for s in st]


def plan(arm: str, kind_seq: list[str], settings: list[str], n_docs: int = N_DOCS) -> list[dict]:
    rng = random.Random(f"{SEED}|{arm}")
    pool, docs, p = items(arm), [], 0
    while len(docs) < n_docs:
        order = pool[:]
        rng.shuffle(order)
        for g in range(0, len(order) - PER_DOC + 1, PER_DOC):
            if len(docs) == n_docs:
                break
            n = len(docs)
            kind = kind_seq[n]
            docs.append({"doc_id": f"{arm}-{n:04d}", "pass": p, "kind": kind,
                         "template_id": TEMPLATE[arm][kind], "items": sorted(order[g:g + PER_DOC]),
                         "setting": settings[n] if kind in ("story", "steadiness") else None})
        p += 1
    return docs


STORY = ("story", "steadiness")


def repair_stories(docs: list[dict], n_prefix: int) -> list[tuple[str, str, str]]:
    """§4.1's story rule within the 1M prefix: every item without a story there is swapped into a
    story document of the prefix, in place of an item of that document that has at least two stories
    in the prefix and does not already appear in it (deterministic: the first such document in plan
    order). Kinds and settings are unchanged. Returns (doc_id, item in, item out)."""
    swaps = []
    pre = docs[:n_prefix]

    def stories() -> dict[str, int]:
        c: dict[str, int] = {}
        for d in pre:
            if d["kind"] in STORY:
                for i in d["items"]:
                    c[i] = c.get(i, 0) + 1
        return c

    every = sorted({i for d in pre for i in d["items"]})
    for item in every:
        c = stories()
        if c.get(item, 0):
            continue
        for d in pre:
            if d["kind"] not in STORY or item in d["items"]:
                continue
            donor = next((i for i in d["items"] if c.get(i, 0) >= 2), None)
            if donor:
                d["items"] = sorted([i for i in d["items"] if i != donor] + [item])
                swaps.append((d["doc_id"], item, donor))
                break
    return swaps


def coverage(arm: str, docs: list[dict], n_tokens: float) -> dict:
    n = min(len(docs), int(n_tokens / MEAN_TOKENS))
    cov = CoverageTracker(items(arm), k_docs=5, story_kinds=("story", "steadiness"))
    for d in docs[:n]:
        cov.add(d["doc_id"], d["kind"], d["items"])
    r = cov.report()
    return {"docs": n, "covered_k5": r["covered_k"], "covered_story": r["covered_story"],
            "n_items": r["n_targets"]}


def main() -> None:
    rng = random.Random(f"{SEED}|shared")
    # one kind sequence and one settings sequence for both arms: rotate kinds so an item meets a
    # different kind in each pass (D-doc deals 116 groups per pass, D-neutral 53)
    kind_seq = [KINDS[(n + n // 53) % len(KINDS)] for n in range(N_DOCS)]
    settings = [rng.choice(SETTINGS) for _ in range(N_DOCS)]
    arms = {a: plan(a, kind_seq, settings) for a in ("d_doc", "d_neutral")}
    n_prefix = int(1e6 / MEAN_TOKENS)
    swaps = {a: repair_stories(d, n_prefix) for a, d in arms.items()}
    rec = {"rule": "CST Amendment 20 item 2; PILOT_BUILD.md §4", "seed": SEED, "per_doc": PER_DOC,
           "n_docs": N_DOCS, "mean_tokens_assumed": MEAN_TOKENS, "kinds": KINDS,
           "story_swaps": swaps,
           "coverage": {a: {"0.3M": coverage(a, d, 3e5), "1M": coverage(a, d, 1e6)}
                        for a, d in arms.items()},
           "arms": arms}
    out = CST / "data/pilot_corpus_plan_v1.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {out}", json.dumps(rec["coverage"]))


V2_DOCS = 2600  # D-doc to about 3.3M tokens at the measured 1,266-token mean (CST pod 2, tomorrow)


def extend(n_docs: int = V2_DOCS) -> dict:
    """The plan of record extended for D-doc (Amendment 29; the author, 2026-11-11): v1's 900
    documents exactly as stored (story swaps included), then documents 900..n-1 dealt by the same
    rule (the kind formula, the shared settings stream and the arm's seeded passes continue where
    v1 stopped, so the prefix is unchanged). D-neutral stays v1's. Writes the v2 plan."""
    v1 = json.loads((CST / "data/pilot_corpus_plan_v1.json").read_text())
    rng = random.Random(f"{SEED}|shared")
    kind_seq = [KINDS[(n + n // 53) % len(KINDS)] for n in range(n_docs)]
    settings = [rng.choice(SETTINGS) for _ in range(n_docs)]
    full = plan("d_doc", kind_seq, settings, n_docs)
    raw_v1 = plan("d_doc", kind_seq[:N_DOCS], settings[:N_DOCS])
    assert [d["doc_id"] for d in full[:N_DOCS]] == [d["doc_id"] for d in v1["arms"]["d_doc"]]
    assert all(a["kind"] == b["kind"] and a["setting"] == b["setting"] and a["pass"] == b["pass"]
               for a, b in zip(full, raw_v1)), "the extended deal must reproduce v1's prefix"
    rec = dict(v1, n_docs={"d_doc": n_docs, "d_neutral": v1["n_docs"]},
               rule=v1["rule"] + "; extended for D-doc by pilot_corpus_plan.extend (2026-11-11)",
               extended_from="data/pilot_corpus_plan_v1.json")
    rec["arms"] = {"d_doc": v1["arms"]["d_doc"] + full[N_DOCS:],
                   "d_neutral": v1["arms"]["d_neutral"]}
    rec["coverage"] = dict(v1["coverage"], d_doc=dict(v1["coverage"]["d_doc"], **{
        "3M": coverage("d_doc", rec["arms"]["d_doc"], 3e6)}))
    out = CST / "data/pilot_corpus_plan_v2.json"
    out.write_text(json.dumps(rec, indent=1) + "\n")
    print(f"wrote {out}: d_doc {len(rec['arms']['d_doc'])} docs",
          json.dumps(rec["coverage"]["d_doc"]["3M"]))
    return rec


if __name__ == "__main__":
    import sys

    extend() if sys.argv[1:] == ["extend"] else main()
