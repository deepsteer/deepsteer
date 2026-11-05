#!/usr/bin/env python3
"""CST T0.1 reviews and votes on the committed tenet sets (zero GPU, zero API billing).

    t01_review.py review     # second-family review of each decomposition (codex, GPT family)
    t01_review.py propose    # free cross-document matches (candidates united with TF-IDF)
    t01_review.py vote       # T_core votes on the candidate pairs (claude -p and codex)
    t01_review.py conflicts  # document-internal competing pairs (claude -p proposes, codex checks)

Both CLIs run on account logins with API keys stripped from the child environment
(``kdg_panel/scripts/generate_scenarios.py``'s helpers). Every reply is saved verbatim under
``outputs_raw/t01/`` (untracked) and parsed into ``outputs/t01_*.json`` (tracked). The third T_core
family is the author's G0 choice; until then a pair with a split vote is ``pending``.

``--tenets v2`` runs on the revised sets (``t01_revise_merge.py``): the review covers only revised,
merged, split and added tenets; candidates carry both families' v1 proposals whose ids survive and
add fresh Claude proposals; a v1 vote carries over only when both tenets' claim and conditions are
unchanged. v1 artifact names are unchanged; v2 files carry ``_v2``.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "kdg_panel" / "scripts"),
                str(Path(__file__).resolve().parent)]

from generate_scenarios import claude_cli_exec, codex_exec, extract_json  # noqa: E402

from deepsteer.constitutions import TenetSet, load_document  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"
ST = ROOT / "papers" / "st_live_reference"
RAW = CST / "outputs_raw" / "t01"
BATCH = 40
SYS = ("You are a careful analyst of AI behavior specifications. Answer only with the JSON "
       "asked for.")


TV = "v1"  # tenet-set version (``--tenets``)


def _sets(tv: str | None = None) -> dict[str, TenetSet]:
    tv = tv or TV
    return {d: TenetSet.load(CST / "data" / f"tenets_{d}_{tv}.json") for d in ("C_anth", "C_spec")}


def _sfx() -> str:
    return "" if TV == "v1" else f"_{TV}"


def _text_key(t) -> str:
    return f"{t.claim}\n{t.conditions}"


def _save(name: str, text: str) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / name).write_text(text)


def _cached(name: str, call) -> str:
    """The saved reply if this call already ran (resumable after a quota stop), else call and
    save. Saved replies are the record; a call is never repeated once its reply is on disk."""
    f = RAW / name
    if f.exists() and f.read_text().strip():
        return f.read_text()
    text = call()
    _save(name, text)
    return text


def _tenet_line(t) -> str:
    return f"[{t.tenet_id}] {t.claim} (when: {t.conditions})"


# ---- review --------------------------------------------------------------------------------

REVIEW_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["tenets", "missed", "merge_or_split"],
    "properties": {
        "tenets": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["id", "supported", "accurate", "comment"],
            "properties": {"id": {"type": "string"}, "supported": {"type": "boolean"},
                           "accurate": {"type": "boolean"}, "comment": {"type": "string"}}}},
        "missed": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["section", "behavior", "quote"],
            "properties": {"section": {"type": "string"}, "behavior": {"type": "string"},
                           "quote": {"type": "string"}}}},
        "merge_or_split": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["ids", "issue"],
            "properties": {"ids": {"type": "array", "items": {"type": "string"}},
                           "issue": {"type": "string"}}}}}}


def _chunks(doc, ts: TenetSet, max_chars: int = 40000):
    secs = doc.sections()
    cur, size = [], 0
    for s in secs:
        if cur and size + len(s.text) > max_chars:
            yield cur
            cur, size = [], 0
        cur.append(s)
        size += len(s.text)
    if cur:
        yield cur


def _changed(ts: TenetSet) -> set[str]:
    log = ts.provenance.get("revision_log", [])
    return {x["id"] for x in log if x.get("id") and x["action"] in
            ("revise", "merge", "split", "add")}


def review() -> None:
    sets = _sets()
    jobs = []
    for d, ts in sets.items():
        keep = _changed(ts) if TV != "v1" else set(ts.ids())
        doc = load_document(d, ST / "constitutions.yaml", ST / "outputs" / "constitutions")
        for k, secs in enumerate(_chunks(doc, ts)):
            ids = {s.section_id for s in secs}
            mine = [t for t in ts.tenets if t.section in ids and t.tenet_id in keep]
            if not mine:
                continue
            text = "\n\n".join(f"[{s.section_id}] ## {s.heading}\n{s.text}" for s in secs)
            prompt = (
                "Below are sections of a published AI behavior specification, then the tenets "
                "(atomic, testable behavioral claims, each meant to be violable by a single "
                "transcript) another analyst extracted from them. Review the extraction.\n"
                "For every tenet: supported = the quoted passage in the sections states this "
                "behavior; accurate = the claim and its conditions match the text without "
                "adding or dropping a condition. List behaviors the sections clearly require "
                "that no tenet covers (missed; quote the text exactly), and tenets that should "
                "be merged (near-duplicates) or split (two requirements in one).\n\n"
                f"<sections>\n{text}\n</sections>\n\n<tenets>\n"
                + "\n".join(f"{_tenet_line(t)} quote: \"{t.quote}\"" for t in mine)
                + "\n</tenets>")
            jobs.append((f"review{_sfx()}_{d}_{k:02d}", prompt))
    with ThreadPoolExecutor(3) as ex:
        outs = list(ex.map(lambda j: (j[0], _cached(
            j[0] + ".json", lambda: codex_exec(j[1], REVIEW_SCHEMA))), jobs))
    res: dict = {"family": "codex (GPT family; account login)", "chunks": {}}
    for name, text in outs:
        res["chunks"][name] = json.loads(text)
    (CST / "outputs" / f"t01_review{_sfx()}.json").write_text(json.dumps(res, indent=1,
                                                                ensure_ascii=False) + "\n")
    for d in ("C_anth", "C_spec"):
        rows = [r for k, v in res["chunks"].items() if d in k for r in v["tenets"]]
        missed = sum(len(v["missed"]) for k, v in res["chunks"].items() if d in k)
        ms = sum(len(v["merge_or_split"]) for k, v in res["chunks"].items() if d in k)
        print(d, "reviewed", len(rows), "unsupported", sum(not r["supported"] for r in rows),
              "inaccurate", sum(not r["accurate"] for r in rows), "missed", missed,
              "merge/split", ms)


# ---- propose (candidates v2) ---------------------------------------------------------------

PROPOSE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["matches"],
                  "properties": {"matches": {"type": "array", "items": {
                      "type": "object", "additionalProperties": False,
                      "required": ["a", "b"],
                      "properties": {"a": {"type": "string"},
                                     "b": {"type": "array", "items": {"type": "string"}}}}}}}
PROPOSE_BATCH = 25


def _propose_prompt(chunk, spec) -> str:
    return (
        "List A holds tenets from one AI behavior specification; list B holds every tenet of "
        "another. For each A tenet, give every B tenet that directs the same behavior in the "
        "same conditions (one may be broader, but a model following either would act the same "
        "way where both apply). Topic overlap alone does not count. Return JSON {\"matches\": "
        "[{\"a\": \"<A id>\", \"b\": [\"<B id>\", ...]}, ...]} with every A id present "
        "(empty list if none).\n\nList A:\n" + "\n".join(_tenet_line(t) for t in chunk)
        + "\n\nList B:\n" + "\n".join(_tenet_line(t) for t in spec))


FAMILIES = ("claude", "codex")


def _fam_fn(fam: str, schema: dict):
    if fam == "claude":
        return lambda p: claude_cli_exec(SYS, p)
    return lambda p: codex_exec(p, schema)


def propose(families: tuple[str, ...] = FAMILIES) -> None:
    """Free matches per family; candidates v2 = v1 united with every family's saved proposals.
    A family not run yet (quota) contributes nothing and is listed as pending."""
    sets = _sets()
    a, spec = sets["C_anth"].tenets, sets["C_spec"].tenets
    chunks = [a[i:i + PROPOSE_BATCH] for i in range(0, len(a), PROPOSE_BATCH)]
    sid = set(sets["C_spec"].ids())
    found: dict[str, set] = {}
    for fam in FAMILIES:
        fn = _fam_fn(fam, PROPOSE_SCHEMA)
        names = [f"propose{_sfx()}_{fam}_{i:02d}.txt" for i in range(len(chunks))]
        if fam not in families and not all((RAW / n).exists() for n in names):
            continue

        def run(ic, fn=fn, fam=fam):
            i, c = ic
            text = _cached(f"propose{_sfx()}_{fam}_{i:02d}.txt",
                           lambda: fn(_propose_prompt(c, spec)))
            return extract_json(text)["matches"]
        with ThreadPoolExecutor(3) as ex:
            found[fam] = {(m["a"], b) for ms in ex.map(run, enumerate(chunks)) for m in ms
                          for b in m["b"] if b in sid}
    if TV == "v1":
        v1 = {tuple(p) for p in json.loads((CST / "data" / "tcore_candidates_v1.json")
                                           .read_text())["pairs"]}
        carried: dict[str, set] = {}
    else:
        from t01_tenets import candidates

        v1 = {tuple(p) for p in candidates(sets["C_anth"], sets["C_spec"])}
        aid = set(sets["C_anth"].ids())
        old = json.loads((CST / "data" / "tcore_candidates_v1_union.json").read_text())
        carried = {f: {tuple(p) for p in ps if p[0] in aid and p[1] in sid}
                   for f, ps in old["proposals"].items()}
    union = v1.union(*found.values(), *carried.values())
    cross = {f"{f}_in_{g}": len(found[f] & found[g]) / len(found[f])
             for f in found for g in found if f != g and found[f]}
    res = {"rule": "candidates v1 (TF-IDF top 3, both directions) united with each family's "
                   "free matches against the full C_spec list (25 C_anth tenets per call)",
           "families_proposed": sorted(found),
           "families_pending": sorted(set(FAMILIES) - set(found)),
           "n_v1": len(v1), "proposed": {f: len(x) for f, x in found.items()},
           "v1_recall_of_proposals": {f: len(found[f] & v1) / len(found[f]) for f in found
                                      if found[f]},
           "cross_family_share": cross, "n": len(union),
           "proposals": {f: sorted([list(p) for p in x]) for f, x in found.items()},
           "carried_v1_proposals": {f: len(x) for f, x in carried.items()},
           "pairs": sorted([list(p) for p in union])}
    name = ("tcore_candidates_v1_union.json" if TV == "v1"
            else f"tcore_candidates_tenets{_sfx()}.json")
    (CST / "data" / name).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("pairs", "proposals")}))


# ---- vote ----------------------------------------------------------------------------------

VOTE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["votes"],
               "properties": {"votes": {"type": "array", "items": {
                   "type": "object", "additionalProperties": False,
                   "required": ["pair", "same"],
                   "properties": {"pair": {"type": "string"}, "same": {"type": "boolean"}}}}}}


def _vote_prompt(batch, by) -> str:
    lines = [f"{a}|{s}\n  A: {_tenet_line(by[a])}\n  B: {_tenet_line(by[s])}" for a, s in batch]
    return (
        "Each item pairs a tenet from one AI behavior specification (A) with one from another "
        "(B). For each pair answer same = true only if both direct the same behavior in the "
        "same conditions (one may be worded more broadly, but a model following either would "
        "act the same way in the conditions both cover); otherwise false. Topic overlap alone is "
        "false. Return JSON {\"votes\": [{\"pair\": \"<A id>|<B id>\", \"same\": bool}, ...]} "
        "covering every pair.\n\n" + "\n".join(lines))


def vote(families: tuple[str, ...] = FAMILIES, no_sample: float | None = None) -> None:
    """Votes per family on every candidate pair that family has not voted on. Batches are keyed
    by a hash of their pairs, so a saved reply is reused only for exactly the pairs it covered."""
    import hashlib

    sets = _sets()
    by = {**sets["C_anth"].by_id(), **sets["C_spec"].by_id()}
    src = CST / "data" / ("tcore_candidates_v1_union.json" if TV == "v1"
                          else f"tcore_candidates_tenets{_sfx()}.json")
    pairs = [tuple(p) for p in json.loads(src.read_text())["pairs"]]
    prior = CST / "outputs" / f"t01_tcore_votes{_sfx()}.json"
    votes: dict[str, dict[str, bool]] = (
        {k: dict(v) for k, v in json.loads(prior.read_text())["votes"].items()}
        if prior.exists() else {})
    carried_n = 0
    sample_rule = None
    if TV != "v1":  # a v1 vote stands only where both tenets' text is unchanged
        old_by = {**_sets("v1")["C_anth"].by_id(), **_sets("v1")["C_spec"].by_id()}
        old_votes = json.loads((CST / "outputs" / "t01_tcore_votes.json").read_text())["votes"]
        for a, b in pairs:
            k = f"{a}|{b}"
            if k in votes or k not in old_votes or a not in old_by or b not in old_by:
                continue
            same = (_text_key(old_by[a]) == _text_key(by[a])
                    and _text_key(old_by[b]) == _text_key(by[b]))
            if same:
                votes[k] = dict(old_votes[k])
                carried_n += 1
    for fam in families:
        fn = _fam_fn(fam, VOTE_SCHEMA)
        todo = [p for p in pairs if fam not in votes.get(f"{p[0]}|{p[1]}", {})]
        if no_sample is not None and fam != "claude":
            # registered quota rule: every Claude-yes pair, plus a seeded sample of Claude-no
            # pairs (a Claude-no pair matches only if this family AND the third vote yes)
            import random

            cyes = [p for p in todo if votes.get(f"{p[0]}|{p[1]}", {}).get("claude") is True]
            cno = sorted(p for p in todo if votes.get(f"{p[0]}|{p[1]}", {}).get("claude") is False)
            k = round(no_sample * len(cno))
            todo = cyes + sorted(random.Random(20261104).sample(cno, k))
            sample_rule = {"family": fam, "claude_yes": len(cyes), "claude_no": len(cno),
                           "claude_no_sampled": k, "fraction": no_sample, "seed": 20261104}
        batches = [todo[i:i + BATCH] for i in range(0, len(todo), BATCH)]

        def run(b, fn=fn, fam=fam):
            key = [[a, s, _text_key(by[a]), _text_key(by[s])] for a, s in b]
            h = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:12]
            text = _cached(f"vote_{fam}_{h}.txt", lambda: fn(_vote_prompt(b, by)))
            return extract_json(text)["votes"]
        with ThreadPoolExecutor(3) as ex:
            for vs in ex.map(run, batches):
                for v in vs:
                    votes.setdefault(v["pair"], {})[fam] = bool(v["same"])
    want = {f"{a}|{s}" for a, s in pairs}
    missing = {f: sum(f not in votes.get(p, {}) for p in want) for f in FAMILIES}
    both_yes = sorted(p for p in want if all(votes.get(p, {}).get(f) for f in FAMILIES))
    split = sorted(p for p in want if len(votes.get(p, {})) == 2
                   and len(set(votes[p].values())) == 2)
    res = {"families": list(FAMILIES), "tenets": TV, "candidates": src.name,
           "n_pairs": len(want), "carried_from_v1": carried_n, "sample_rule": sample_rule,
           "votes": {k: v for k, v in sorted(votes.items()) if k in want},
           "missing": missing, "matched_two_of_two": both_yes, "pending_split": split,
           "rule": "match = >= 2 of 3 families (CST_PLAN §2.2); 2 of 2 matches regardless of the "
                   "third, 0 of 2 cannot match, splits wait for the third family; a pair missing "
                   "a family's vote is pending"}
    prior.write_text(json.dumps(res, indent=1) + "\n")
    print("pairs", len(want), "two-of-two", len(both_yes), "split", len(split), "missing",
          missing)


# ---- conflicts -----------------------------------------------------------------------------

CHECK_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["checks"],
                "properties": {"checks": {"type": "array", "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["pair", "competing", "comment"],
                    "properties": {"pair": {"type": "string"}, "competing": {"type": "boolean"},
                                   "comment": {"type": "string"}}}}}}


def conflicts(with_codex: bool = True) -> None:
    sets = _sets()
    res: dict = {}
    for d, ts in sets.items():
        listing = "\n".join(_tenet_line(t) + (f" [note: {t.notes}]" if "defers_to" in t.notes
                                              else "") for t in ts.tenets)
        prompt = (
            "Below are the tenets extracted from one AI behavior specification. Find (1) pairs "
            "of tenets that direct DIFFERENT behavior in OVERLAPPING conditions, so that a model "
            "in those conditions cannot satisfy both as written; (2) tenets whose note says they "
            "defer to a specific other consideration, mapped to the tenet that states that "
            "consideration. Do NOT list pairs merely because the document has a general "
            "priority ordering; list only pairs where the conditions genuinely overlap and the "
            "directions genuinely differ. Return JSON {\"competing\": [{\"pair\": "
            "\"<id>|<id>\", \"conditions\": \"...\"}], \"conditioned_on\": [{\"id\": "
            "\"<id>\", \"on\": \"<id>\"}]}."
            f"\n\n{listing}")
        text = _cached(f"conflicts{_sfx()}_{d}_claude.txt",
                       lambda p=prompt: claude_cli_exec(SYS, p))
        prop = extract_json(text)
        pairs = [c["pair"] for c in prop.get("competing", [])]
        by = ts.by_id()
        ok = [p for p in pairs if all(x in by for x in p.split("|"))]
        if not with_codex:
            res[d] = {"proposed_by": "claude", "competing": prop.get("competing", []),
                      "conditioned_on": prop.get("conditioned_on", []),
                      "codex_check": None, "codex_pending": True}
            print(d, "competing proposed", len(pairs), "codex check pending",
                  "conditioned_on", len(res[d]["conditioned_on"]))
            continue
        check = _cached(f"conflicts{_sfx()}_{d}_codex.json", lambda ok=ok, by=by: codex_exec(
            "For each pair of tenets from one AI behavior specification, answer competing = "
            "true only if the two direct different behavior in overlapping conditions, so a "
            "model in those conditions cannot satisfy both as written.\n\n"
            + "\n".join(f"{p}\n  {_tenet_line(by[p.split('|')[0]])}\n  "
                        f"{_tenet_line(by[p.split('|')[1]])}" for p in ok),
            CHECK_SCHEMA) if ok else '{"checks": []}')
        res[d] = {"proposed_by": "claude", "competing": prop.get("competing", []),
                  "conditioned_on": prop.get("conditioned_on", []),
                  "codex_check": json.loads(check)["checks"]}
        agree = sum(c["competing"] for c in res[d]["codex_check"])
        print(d, "competing proposed", len(pairs), "codex agrees", agree,
              "conditioned_on", len(res[d]["conditioned_on"]))
    (CST / "outputs" / f"t01_conflicts{_sfx()}.json").write_text(json.dumps(res, indent=1,
                                                                   ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("review", "propose", "vote", "conflicts"))
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--tenets", default="v1", choices=("v1", "v2"))
    ap.add_argument("--no-sample", type=float, default=None,
                    help="vote: non-Claude families vote every Claude-yes pair plus this seeded "
                         "fraction of Claude-no pairs")
    a = ap.parse_args()
    TV = a.tenets
    fams = tuple(a.families.split(","))
    if a.step in ("propose", "vote"):
        if a.step == "vote":
            vote(fams, a.no_sample)
        else:
            propose(fams)
    else:
        if a.step == "conflicts":
            conflicts("codex" in fams)
        else:
            review()
