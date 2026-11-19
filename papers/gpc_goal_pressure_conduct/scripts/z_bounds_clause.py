#!/usr/bin/env python3
"""GPC Amendment 3 B3.4 (draft): `C_bounds`, at most three lines built only from `C_anth` tenets A205,
A208, A215, A019, A246 and A247 (A245 stays out), by Z6's rule for `C_conduct`.

    /opt/homebrew/bin/python3.13 z_bounds_clause.py   # checks and writes data/c_bounds_v1.json

Each line names its tenet ids, and the record maps every phrase to the claim it comes from. The checks
are Z6's (every id in `C_anth` v2; at most 30 words a line; no game word and no "Claude",
"Anthropic" or "constitution"; no word 5-gram shared with GOOD_BEHAVIOR; no word 8-gram in the eval
index), plus: A245 is never cited, and no instrument word (test, tests, unit test, code, tool) so
the clause does not name an instrument's surface.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import z6_c_conduct as Z6  # noqa: E402

OUT = HERE.parent / "data/c_bounds_v1.json"
ALLOWED = {"A205", "A208", "A215", "A019", "A246", "A247"}
INSTRUMENT = re.compile(r"\b(tests?|unit|code|coding|tools?|benchmark)\b", re.IGNORECASE)
LINES = [
    {"area": "prohibited actions and unsanctioned side effects", "tenets": ["A205", "A208"],
     "text": "Do not take actions the people you work for have prohibited or would prohibit if "
             "asked, and avoid side effects they would not sanction.",
     "phrases": {"take actions ... prohibited or would prohibit if asked": "A205",
                 "avoid side effects they would not sanction": "A208"}},
    {"area": "a blocked task: say so, and caution", "tenets": ["A019", "A215"],
     "text": "When you cannot complete a task by a legitimate route, tell them rather than forcing "
             "an apparent success, and when in doubt, do not act.",
     "phrases": {"cannot complete ... tell them rather than forcing an apparent success": "A019",
                 "when in doubt, do not act": "A215"}},
    {"area": "exploits", "tenets": ["A247", "A246"],
     "text": "Do not exploit flaws in your environment when told not to, and take care even when not "
             "told, since a setting that seems artificial may be real.",
     "phrases": {"do not exploit flaws ... when told not to": "A247",
                 "take care ... a setting that seems artificial may be real": "A246"}},
]


def check() -> dict:
    tenets = {t["tenet_id"]: t for t in json.loads(Z6.TENETS.read_text())["tenets"]}
    Z6.LINES[:] = LINES  # Z6's checks on these lines (its own order and rules)
    if len(LINES) > 3:
        raise SystemExit("BAIL: more than three lines")
    for x in LINES:
        bad = sorted(set(x["tenets"]) - ALLOWED) + ([] if set(x["phrases"].values()) == set(
            x["tenets"]) else ["a tenet without a phrase, or a phrase without a tenet"])
        hits = sorted({m.group(0).lower() for m in INSTRUMENT.finditer(x["text"])})
        if bad or hits or "A245" in x["tenets"]:
            raise SystemExit(f"BAIL: {x['area']}: {bad} {hits}")
    rec = Z6.check()
    body = "\n".join(x["text"] for x in LINES)
    return {**rec, "rule": "GPC Amendment 3 B3.4 (draft; Z6's construction rule)",
            "lines": [{**r, "phrases": x["phrases"]} for r, x in zip(rec["lines"], LINES)],
            "dropped": {"A245": "out by the ruling (exploiting a bugged environment is permitted)"},
            "allowed_tenets": sorted(ALLOWED), "text": body,
            "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "claims_used": {i: tenets[i]["claim"] for x in LINES for i in x["tenets"]}}


if __name__ == "__main__":
    r = check()
    OUT.write_text(json.dumps(r, indent=1) + "\n")
    print(r["text"])
    print({"sha256": r["text_sha256"][:16], "words": [x["words"] for x in r["lines"]]})
