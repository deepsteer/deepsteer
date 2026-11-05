#!/usr/bin/env python3
"""CST T0.1: recall of the T_core candidate filter (zero GPU, account login only).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_recall.py

T_core's votes only see the TF-IDF candidate pairs (``t01_tenets.py``: top 3 per tenet, both
directions). This check gives a seeded sample of 25 C_anth tenets and the FULL C_spec list to
Claude (``claude -p``) and asks for every C_spec tenet that directs the same behavior in the same
conditions. Recall = the share of those free matches that the candidate filter contained. A free
match outside the filter is a pair the vote never saw.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "papers" / "kdg_panel" / "scripts")]

from generate_scenarios import claude_cli_exec, extract_json  # noqa: E402

from deepsteer.constitutions import TenetSet  # noqa: E402

CST = ROOT / "papers" / "cst_constitutional_steering"
SEED, N = 20261104, 25


def main() -> None:
    a = TenetSet.load(CST / "data" / "tenets_C_anth_v1.json")
    s = TenetSet.load(CST / "data" / "tenets_C_spec_v1.json")
    cand = {tuple(p) for p in json.loads((CST / "data" / "tcore_candidates_v1.json")
                                         .read_text())["pairs"]}
    sample = sorted(random.Random(SEED).sample(a.ids(), N))
    by = a.by_id()
    line = lambda t: f"[{t.tenet_id}] {t.claim} (when: {t.conditions})"  # noqa: E731
    prompt = (
        "List A holds tenets from one AI behavior specification; list B holds every tenet of "
        "another. For each A tenet, give every B tenet that directs the same behavior in the "
        "same conditions (one may be broader, but a model following either would act the same "
        "way where both apply). Topic overlap alone does not count. Return JSON "
        "{\"matches\": {\"<A id>\": [\"<B id>\", ...], ...}} with every A id present (empty "
        "list if none).\n\nList A:\n" + "\n".join(line(by[i]) for i in sample)
        + "\n\nList B:\n" + "\n".join(line(t) for t in s.tenets))
    text = claude_cli_exec("Answer only with the JSON asked for.", prompt)
    raw = CST / "outputs_raw" / "t01"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "recall_claude.txt").write_text(text)
    m = extract_json(text)["matches"]
    free = sorted((i, j) for i in sample for j in m.get(i, []) if j in s.by_id())
    inside = [p for p in free if p in cand]
    res = {"seed": SEED, "sample": sample, "free_matches": [list(p) for p in free],
           "n_free": len(free), "n_in_candidates": len(inside),
           "recall": len(inside) / len(free) if free else None,
           "missed": [list(p) for p in free if p not in cand]}
    (CST / "outputs" / "t01_recall.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"free matches {len(free)}; inside the candidate filter {len(inside)}; "
          f"recall {res['recall']}")


if __name__ == "__main__":
    main()
