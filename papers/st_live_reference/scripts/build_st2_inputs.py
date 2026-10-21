#!/usr/bin/env python3
"""ST Phase 2's inputs (Amendment 22; zero GPU), built locally from committed and T1.x artifacts.

    python3 papers/st_live_reference/scripts/build_st2_inputs.py engaged \
        papers/st_live_reference/outputs/t11/gpt_oss_20b
    python3 papers/st_live_reference/scripts/build_st2_inputs.py cell1
    python3 papers/st_live_reference/scripts/build_st2_inputs.py twins <spans.json> <rewrites.json>

- ``engaged``: ``data/st2_engaged_train_v1.json``, T1.1's 45 brake-engaged H1_train ids copied from
  ``select/engaged.json`` (no new draw).
- ``cell1``: ``outputs/st2_inputs/gpt_oss_20b/st2_inputs.npz`` + ``.json`` (synced to the pod):
  the unit anchor axis, ``care_harm`` and ``V_harm`` at block 12 (exactly the T1.4 inputs' vectors)
  and the T1.3 gate's five unit fill draws, with the sources' sha256.
- ``twins``: ``data/st2_rule_twins_v1.json``, the rule-absent twin of every H1_train scenario's
  ``agent_text`` (Amendment 22 item 2 and the author's twin rules): the written-rule sentence(s)
  deleted (``deleted``); deleted plus a hand-chosen clause edit where the rule is stated again
  elsewhere (``clause_deleted``); or, where the rule sentence also carries other facts, replaced by
  a sentence keeping only those facts, with no requirement wording and no content word absent from
  the original (``rewritten``; built for the dose-0 transport control only). Options are verbatim.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTS = HERE.parent / "outputs"
DATA = HERE.parent / "data"
T14 = OUTS / "t14_inputs" / "gpt_oss_20b" / "t14_inputs.npz"
T13 = OUTS / "t13_inputs" / "gpt_oss_20b" / "t13_inputs.npz"
OUT = OUTS / "st2_inputs" / "gpt_oss_20b"


# where the rule is stated again outside its sentence: the smallest edit removing it (hand-chosen,
# 2026-10-21; old text must occur exactly once in the text after the span's deletion)
CLAUSE_EDITS = {
    "H1-B-123": [("any release outside the policy.", "any release.")],
    "H1-B-203": [("the release outside procedure during", "the release during")],
    "H1-B-261": [("If the rule is followed, the room", "The room")],
    "H1-A-194": [(", the assistant principal is on leave, and the district records office "
                  "confirms that only the principal can approve.",
                  ", and the assistant principal is on leave.")],
    "H1-A-58": [(", the only other authorized signer,", "")],
    "H1-B-131": [(" without the required email.", ".")],
    "H1-A-10": [(" without that step,", ",")],  # "that step" pointed at the deleted rule
}
# left as is (recorded): H1-A-58's "can override that step" carries the access and audit facts the
# departure rests on; deleting it would remove them
BANNED = re.compile(r"\b(must|requir\w*|polic\w*|rules?|procedur\w*|sign[- ]?off|signs? off|"
                    r"signed off|approv\w*|authori[sz]\w*|permission|allowed|permit\w*|only)\b",
                    re.I)
FUNCTION = set("the a an of to in on at for by with and or but is are was were be been has have "
               "had who that this it its their his her your you can as from any each over".split())


def _words(t: str) -> list[str]:
    return [x.lower().removesuffix("'s") for x in re.findall(r"[A-Za-z0-9£$][A-Za-z0-9'£$-]*", t)]


def _once(text: str, old: str, sid: str) -> None:
    if text.count(old) != 1:
        raise SystemExit(f"{sid}: {old[:50]!r} occurs {text.count(old)} times")


def twins(spans_path: Path, rewrites_path: Path) -> dict:
    sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel" / "scripts")]
    import pod_st_screen as P

    spans = json.loads(spans_path.read_text())["spans"]
    rew = json.loads(rewrites_path.read_text())["rewrites"]
    sc = {s["id"]: s for s in json.loads(P.H1_SET.read_text())["scenarios"]}
    train = json.loads((HERE.parent / "split_v1.json").read_text())["H1_train"]
    if sorted(spans) != sorted(train):
        raise SystemExit("the spans do not cover H1_train exactly")
    out: dict = {}
    for sid in sorted(train):
        v, text = spans[sid], sc[sid]["agent_text"]
        joined = " ".join(v["agent_rule_spans"])
        for s in v["agent_rule_spans"]:
            _once(text, s, sid)
        if v["mixed"]:
            kind, repl = "rewritten", rew[sid]["replacement"]
            if rew[sid]["original"] != joined:
                raise SystemExit(f"{sid}: rewrite does not match its spans")
            if BANNED.search(repl):
                raise SystemExit(f"{sid}: requirement wording in the rewrite")
            new = [w for w in _words(repl) if w not in FUNCTION and w not in set(_words(joined))]
            if new:
                raise SystemExit(f"{sid}: content words not in the original: {new}")
            _once(text, joined, sid)
            twin = text.replace(joined, repl)
        else:
            kind, repl, twin = "deleted", None, text
            for s in v["agent_rule_spans"]:
                twin = twin.replace(s + " ", "") if s + " " in twin else twin.replace(" " + s, "")
        edits = CLAUSE_EDITS.get(sid, [])
        for old, new_t in edits:
            _once(twin, old, sid)
            twin = twin.replace(old, new_t)
        if edits:
            kind = "clause_deleted"
        if "  " in twin or any(s in twin for s in v["agent_rule_spans"]):
            raise SystemExit(f"{sid}: twin text malformed")
        out[sid] = {"twin_kind": kind, "agent_rule_spans": v["agent_rule_spans"],
                    "clause_edits": [list(e) for e in edits], "replacement": repl,
                    "agent_text_twin": twin, "span_note": v.get("note", ""),
                    "sha256_agent_text": hashlib.sha256(text.encode()).hexdigest(),
                    "sha256_agent_text_twin": hashlib.sha256(twin.encode()).hexdigest()}
    kinds: dict[str, int] = {}
    for x in out.values():
        kinds[x["twin_kind"]] = kinds.get(x["twin_kind"], 0) + 1
    return {"rule": "ST_PLAN Amendment 22 item 2 and the author's twin rules (2026-10-21)",
            "frame": "agent_text only (the agent frame Cells 2's generation and dose-0 reads use); "
                     "options verbatim",
            "extraction_and_target": "deleted + clause_deleted only; rewritten twins serve the "
                                     "dose-0 transport control",
            "counts": kinds, "scenario_set_sha256": sha(P.H1_SET),
            "spans_source": "subagent-marked, mechanically verified; rewrites subagent-written, "
                            "word-checked and hand-checked (all 22)", "twins": out}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def engaged(t11: Path) -> dict:
    f = t11 / "select" / "engaged.json"
    t = json.loads(f.read_text())["train"]
    if len(t["engaged_ids"]) != t["n_engaged"]:
        raise SystemExit("engaged.json is inconsistent")
    split = json.loads((HERE.parent / "split_v1.json").read_text())
    if not set(t["engaged_ids"]) <= set(split["H1_train"]):
        raise SystemExit("an engaged id is not in H1_train")
    return {"rule": f"T1.1's H1_train brake-engaged subset ({t['rule']}); ST_PLAN Amendment 22",
            "n": t["n_engaged"], "ids": sorted(t["engaged_ids"]), "source_sha256": sha(f)}


def cell1() -> dict:
    a, f = np.load(T14), np.load(T13)
    vecs = {k: a[k].astype(np.float32) for k in ("anchor_axis", "care_harm", "V_harm")}
    for k, v in vecs.items():
        if abs(float(np.linalg.norm(v)) - 1.0) > 1e-5:
            raise SystemExit(f"{k} is not unit")
    if not np.allclose(vecs["anchor_axis"], f["axis"], atol=1e-6):
        raise SystemExit("the T1.4 anchor axis differs from the T1.3 gate's")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "st2_inputs.npz", **vecs, fill_unit=f["fill_unit"].astype(np.float32))
    meta = {"rule": "ST_PLAN Amendment 22, Cell 1", "layer": "block 12 (hidden index 13)",
            "class": "clause_end", "alpha": 267.301171875,
            "vectors": {"anchor_axis": "t14_inputs anchor_axis (= the T1.3 gate's axis)",
                        "care_harm": "t14_inputs care_harm (W4 14.4, unit)",
                        "V_harm": "t14_inputs V_harm (t_inst, unit)",
                        "fill_unit": "t13_inputs fill_unit (5 covariance-matched unit draws)"},
            "cos": {"care_harm.V_harm": float(vecs["care_harm"] @ vecs["V_harm"]),
                    "care_harm.anchor_axis": float(vecs["care_harm"] @ vecs["anchor_axis"])},
            "sha256": {"t14_inputs.npz": sha(T14), "t13_inputs.npz": sha(T13),
                       "st2_inputs.npz": sha(OUT / "st2_inputs.npz")}}
    (OUT / "st2_inputs.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    if sys.argv[1:2] == ["engaged"]:
        e = engaged(Path(sys.argv[2]))
        (DATA / "st2_engaged_train_v1.json").write_text(json.dumps(e, indent=1) + "\n")
        print(f">> engaged H1_train: {e['n']}")
    elif sys.argv[1:2] == ["twins"]:
        t = twins(Path(sys.argv[2]), Path(sys.argv[3]))
        (DATA / "st2_rule_twins_v1.json").write_text(json.dumps(t, indent=1, ensure_ascii=False)
                                                      + "\n")
        print(f">> rule twins: {t['counts']}")
    elif sys.argv[1:2] == ["cell1"]:
        m = cell1()
        print(f">> st2 inputs (cell 1): cos {m['cos']}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
