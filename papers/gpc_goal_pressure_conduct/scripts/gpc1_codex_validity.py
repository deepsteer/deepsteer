#!/usr/bin/env python3
"""The build read's item 3 after the download: Codex reads the seeded 30 kept pressure slots
(``conduct/codex_validity.json``, written by ``gpc1_read.py DIR --codex``) and answers, per slot,
whether the goal option carries the assigned violation and whether the other option avoids it.

    /opt/homebrew/bin/python3.13 gpc1_codex_validity.py DIR

One `codex exec` call on the ChatGPT-plan login (no API billing), a JSON schema on the answer.
The items are subject-written scenarios, not subject traces. Writes
``conduct/codex_validity_read.json`` with the counts for the state page.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kdg_panel/scripts"))

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["items"],
          "properties": {"items": {"type": "array", "items": {
              "type": "object", "additionalProperties": False,
              "required": ["slot_id", "goal_carries", "other_avoids"],
              "properties": {"slot_id": {"type": "string"}, "goal_carries": {"type": "boolean"},
                             "other_avoids": {"type": "boolean"}}}}}}


def main(out: Path) -> dict:
    from generate_scenarios import codex_exec

    src = json.loads((out / "conduct" / "codex_validity.json").read_text())
    prompt = src["ask"] + "\n\nItems:\n" + json.dumps(src["items"], indent=1)
    got = json.loads(codex_exec(prompt, SCHEMA))["items"]
    want = {x["slot_id"] for x in src["items"]}
    by = {g["slot_id"]: g for g in got if g["slot_id"] in want}
    rec = {"rule": "the build read's item 3 (Codex, a seeded 30 kept pressure slots)",
           "n": len(want), "answered": len(by),
           "goal_carries": sum(g["goal_carries"] for g in by.values()),
           "other_avoids": sum(g["other_avoids"] for g in by.values()),
           "both": sum(g["goal_carries"] and g["other_avoids"] for g in by.values()),
           "missing": sorted(want - set(by)), "items": [by[i] for i in sorted(by)],
           "reader": "codex exec (ChatGPT-plan login, read-only sandbox)"}
    (out / "conduct" / "codex_validity_read.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({k: rec[k] for k in ("n", "answered", "goal_carries", "other_avoids",
                                          "both", "missing")}))
    return rec


if __name__ == "__main__":
    main(Path(sys.argv[1]))
