#!/usr/bin/env python3
"""Draw 3's stronger in-context arm (GPC_PLAN.md §18.8; the lead's 2026-11-25 7:40 AM item 3): `C_conduct` with the
operator-pressure content P-ctx was distilled from, at a length that fits the depth cell's context. Zero GPU.

P-ctx is NLL on A0s's rollouts on CST's operator-pressure grid (`PRESSURE_BUILD.md` §3), and A0s had the whole
constitution in context (`cst_pressure.a0s_messages`). The whole body does not fit beside a depth run (about 45,000
tokens against a 27,648-token prompt room), so this arm carries the part the pressure grid was built on:
- the four lines of `C_conduct`, unchanged and first, as in the clause arm;
- under A0s's header, worded as an excerpt, inside `<constitution>` tags:
  - the constitution's two sections on operators and users, whole and verbatim ("How to treat operators and users",
    "Handling conflicts between operators and users": the sections of 13 of the grid's 26 tenets);
  - the verbatim quotes of the grid's other 13 tenets, from the four other sections they sit in, in body order.

The body is `deepsteer.constitutions.cut_body("C_anth", ...)` on `C_anth.txt`, checked against the tenet file's
`body_sha256`. The budget is 6,000 tokens of the subject's tokenizer at its pin (the depth runs' peak prompt is
16,676 in draw 2; 16,676 + 6,000 sits under 27,648).

    python z_gpc4_c_press.py   # writes outputs/gpc4_in/c_press_v1.json (the text, synced to the pod, not committed,
                               # as C_anth.txt is not) and data/c_press_v1_record.json (its hashes, committed)
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from deepsteer.constitutions import cut_body, split_sections  # noqa: E402

GPC = HERE.parent
CST = ROOT / "papers/cst_constitutional_steering"
OUT = GPC / "outputs/gpc4_in/c_press_v1.json"
REC = GPC / "data/c_press_v1_record.json"
WHOLE = ("s009", "s011")
BUDGET = 6000
MODEL, REV = "Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
HEADER = ("The document below is excerpted from your constitution: its two sections on operators and users, whole, "
          "and passages on operators and users from four other sections. It describes the values, character and "
          "behavior intended for you.")


def build() -> dict:
    grid = json.loads((CST / "data/pressure_grid_v1.json").read_text())
    ids = sorted({t for c in grid["cells"] for t in c.get("tenet_ids", [])})
    ts = json.loads((CST / "data/tenets_C_anth_v2.json").read_text())
    by = {t["tenet_id"]: t for t in ts["tenets"]}
    body = cut_body("C_anth", (CST / "outputs/_sync/C_anth.txt").read_text())
    if hashlib.sha256(body.encode()).hexdigest() != ts["body_sha256"]:
        raise SystemExit("BAIL: the constitution body differs from the tenet file's")
    secs = {s.section_id: s for s in split_sections(body)}
    whole = [f"## {secs[s].heading}\n\n{secs[s].text}" for s in WHOLE]
    rest = sorted((i for i in ids if by[i]["section"] not in WHOLE), key=lambda i: body.find(by[i]["quote"]))
    for i in rest:
        if by[i]["quote"] not in body:
            raise SystemExit(f"BAIL: {i}'s quote is not verbatim in the body")
    quotes = [by[i]["quote"].strip() for i in rest]
    conduct = json.loads((GPC / "data/c_conduct_v1.json").read_text())["text"].strip()
    excerpt = "\n\n".join(whole + quotes)
    text = f"{conduct}\n\n{HEADER}\n\n<constitution>\n{excerpt}\n</constitution>"
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    n = len(tok(text, add_special_tokens=False)["input_ids"])
    if n > BUDGET:
        raise SystemExit(f"BAIL: {n} tokens, over the {BUDGET} budget")
    rec = {"schema_version": 1, "rule": "GPC_PLAN.md §18.8 (the lead's 2026-11-25 7:40 AM item 3)",
           "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokens": n, "budget": BUDGET,
           "tokenizer": f"{MODEL}@{REV}", "body_sha256": ts["body_sha256"],
           "whole_sections": {s: secs[s].heading for s in WHOLE},
           "grid_tenets": len(ids), "tenets_in_whole_sections": sorted(i for i in ids if by[i]["section"] in WHOLE),
           "quoted_tenets": rest, "c_conduct_sha256": hashlib.sha256(conduct.encode()).hexdigest()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1) + "\n")
    REC.write_text(json.dumps({k: v for k, v in rec.items() if k != "text"}, indent=1) + "\n")
    return rec


if __name__ == "__main__":
    r = build()
    print(json.dumps({k: v for k, v in r.items() if k != "text"}, indent=1))
