#!/usr/bin/env python3
"""The pilot's difficult-advice prompts (CST Amendment 21 item 4; PILOT_BUILD.md §6 item 1).

    pilot_prompts.py grid             # data/pilot_advice_grid_v1.json: the seeded cells
    pilot_prompts.py check <jsonl...> # validate written prompts, 8-gram and near-duplicate checks
                                      # -> data/pilot_advice_prompts_v1.json

The grid: every tenet of T(C_anth) outside T_conf of record (`data/tconf_v1.json`, 52 document-
internal members; no panel-detected members exist) gets two cells, each with a domain and a stakes
level, assigned by seeded balancing so each domain and stakes level is used about equally and a
tenet's two cells differ in domain. Claude writes one user-advice prompt per cell (first person, a
concrete situation where the tenet bears on good advice; no operator, system or agentic framing; the
panel's five scenario themes excluded). `check` keeps a prompt only if it is well formed, shares no
8-gram with any indexed eval set (`t05_overlap_index`, the 7 sets) or with the panel's and H1's
scenario texts, and is not a near duplicate of a kept prompt (word 5-gram Jaccard at or above 0.35).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CST = HERE.parent
ROOT = CST.parents[1]
SEED = "20261109|cst_pilot_advice_grid"
CELLS_PER_TENET = 2
DOMAINS = ("health and medical decisions", "personal finance", "family and parenting",
           "romantic relationships", "friendships", "work and career", "education and study",
           "housing and neighbors", "buying, selling and consumer disputes",
           "online life and technology", "community and civic life",
           "everyday logistics and travel")
STAKES = ("low", "moderate", "high")
# the panel's five scenario themes (KDG F1-F5), kept out of the advice prompts
PANEL_THEMES = ("falsifying a report, metric or record", "research data or results integrity",
                "skipping a required sign-off or approval",
                "bending a workplace rule to favor someone",
                "cutting a safety rule under deadline pressure")
NEAR_DUP = 0.35


def eligible_tenets() -> list[dict]:
    tenets = json.loads((CST / "data/tenets_C_anth_v2.json").read_text())["tenets"]
    tconf = set(json.loads((CST / "data/tconf_v1.json").read_text())["docs"]["C_anth"]["members"])
    return [t for t in tenets if t["tenet_id"] not in tconf]


def grid() -> dict:
    ts = eligible_tenets()
    rng = random.Random(SEED)
    order = [t["tenet_id"] for t in ts]
    rng.shuffle(order)
    claims = {t["tenet_id"]: t["claim"] for t in ts}
    cells = []
    for i, tid in enumerate(order):
        for j in range(CELLS_PER_TENET):
            n = i * CELLS_PER_TENET + j
            # balanced over domains; a tenet's two cells sit six domains apart
            dom = DOMAINS[(i + j * len(DOMAINS) // 2) % len(DOMAINS)]
            stakes = STAKES[(i // len(DOMAINS) + i + j) % len(STAKES)]
            cells.append({"cell_id": f"PA-{n:04d}", "tenet_ids": [tid], "claim": claims[tid],
                          "domain": dom, "stakes": stakes})
    rng.shuffle(cells)
    return {"rule": "CST Amendment 21 item 4", "seed": SEED, "cells_per_tenet": CELLS_PER_TENET,
            "tconf_of_record": "data/tconf_v1.json (52 document-internal members)",
            "eligible_tenets": len(ts), "domains": DOMAINS, "stakes": STAKES,
            "excluded_themes": PANEL_THEMES, "cells": cells}


def _words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", t.lower())


def _grams(t: str, n: int) -> set[tuple[str, ...]]:
    w = _words(t)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def _panel_texts() -> list[str]:
    import glob

    out = []
    for f in sorted(glob.glob(str(ROOT / "papers/kdg_panel/data/*_scenarios_*.json"))):
        for s in json.loads(Path(f).read_text())["scenarios"]:
            out.append(s.get("agent_text") or "")
    return out


def check(paths: list[Path]) -> dict:
    sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers/mch_machiavelli/scripts")]
    import t05_overlap_index as T5

    from deepsteer.corpus.overlap import OverlapIndex

    idx = OverlapIndex()
    for k, v in T5.eval_sets().items():  # the 7 indexed eval sets, the panel and H1 included
        idx.add_set(k, v)

    g = {c["cell_id"]: c for c in json.loads((CST / "data/pilot_advice_grid_v1.json")
                                              .read_text())["cells"]}
    rows = [json.loads(x) for p in paths for x in p.read_text().splitlines() if x.strip()]
    panel8 = set().union(*(_grams(t, 8) for t in _panel_texts()))
    kept, drops, seen5 = [], {"malformed": [], "eval_8gram": [], "panel_8gram": [],
                             "near_duplicate": [], "framing": []}, []
    for r in sorted(rows, key=lambda r: r.get("cell_id", "")):
        cid, text = r.get("cell_id"), (r.get("prompt") or "").strip()
        if cid not in g or not (20 <= len(_words(text)) <= 220):
            drops["malformed"].append(cid)
            continue
        if re.search(r"\b(as an ai|system prompt|operator|you are an? (assistant|agent))\b", text,
                     re.I):
            drops["framing"].append(cid)
            continue
        if idx.check(cid, text).n_shared:
            drops["eval_8gram"].append(cid)
            continue
        if _grams(text, 8) & panel8:
            drops["panel_8gram"].append(cid)
            continue
        g5 = _grams(text, 5)
        if any(len(g5 & s) / max(len(g5 | s), 1) >= NEAR_DUP for s in seen5):
            drops["near_duplicate"].append(cid)
            continue
        seen5.append(g5)
        c = g[cid]
        kept.append({"prompt_id": cid, "tenet_ids": c["tenet_ids"], "domain": c["domain"],
                     "stakes": c["stakes"], "prompt": text,
                     "sha256": hashlib.sha256(text.encode()).hexdigest()})
    return {"rule": "CST Amendment 21 item 4", "provenance": "written locally by Claude (pilot)",
            "grid": "data/pilot_advice_grid_v1.json", "n_written": len(rows), "n_kept": len(kept),
            "eval_index": idx.summary(), "drops": {k: sorted(v) for k, v in drops.items()},
            "prompts": kept}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("grid", "check"))
    ap.add_argument("paths", nargs="*", type=Path)
    a = ap.parse_args()
    if a.step == "grid":
        out = CST / "data/pilot_advice_grid_v1.json"
        d = grid()
        out.write_text(json.dumps(d, indent=1) + "\n")
        print(f"wrote {out}: {len(d['cells'])} cells over {d['eligible_tenets']} tenets")
    else:
        out = CST / "data/pilot_advice_prompts_v1.json"
        d = check(a.paths)
        out.write_text(json.dumps(d, indent=1) + "\n")
        print(f"wrote {out}: kept {d['n_kept']} of {d['n_written']};",
              {k: len(v) for k, v in d["drops"].items()})


if __name__ == "__main__":
    main()
