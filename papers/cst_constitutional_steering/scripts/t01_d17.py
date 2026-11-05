#!/usr/bin/env python3
"""CST-A3's discriminator (G0 decision D17): does a tightened matching prompt raise retest kappa?

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/t01_d17.py

Same 120 pairs as ``t01_retest.py`` (v1 tenets; 60 first-vote yes, 60 no). Two independent Claude
passes (``claude -p``, fresh calls, each in its own seeded order) with a prompt that requires the
narrower tenet's conditions to be contained in the broader one's and names which is broader.
R_prompt predicts retest kappa >= 0.8 (the plan's bar); R_construct predicts no change from the
original prompt's 0.78. Reported beside: each tightened pass against the original first votes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT), str(HERE), str(ROOT / "papers" / "kdg_panel" / "scripts")]

import t01_review as R  # noqa: E402

from deepsteer.judges import retest_agreement  # noqa: E402

SCHEMA_NOTE = ('Return JSON {"votes": [{"pair": "<A id>|<B id>", "broader": "A"|"B"|"equal"|'
               '"neither", "same": bool}, ...]} covering every pair.')


def tight_prompt(batch, by) -> str:
    lines = [f"{a}|{s}\n  A: {R._tenet_line(by[a])}\n  B: {R._tenet_line(by[s])}"
             for a, s in batch]
    return (
        "Each item pairs a tenet from one AI behavior specification (A) with one from another "
        "(B). Decide in two steps. (1) broader: which tenet's conditions contain the other's "
        "('A', 'B', 'equal' if the same, 'neither' if neither set of conditions contains the "
        "other). (2) same = true only if broader is not 'neither' AND, in every situation the "
        "narrower tenet covers, both tenets direct the same action (a model following either "
        "would act the same way there). Shared topic, or one tenet merely mentioning the other's "
        "concern, is false. " + SCHEMA_NOTE + "\n\n" + "\n".join(lines))


def run_pass(pairs, by, seed: int, tag: str) -> dict[str, dict]:
    order = list(pairs)
    random.Random(seed).shuffle(order)
    out: dict[str, dict] = {}
    for i in range(0, len(order), R.BATCH):
        b = order[i:i + R.BATCH]
        text = R._cached(f"d17_{tag}_{i // R.BATCH:02d}.txt",
                         lambda b=b: R.claude_cli_exec(R.SYS, tight_prompt(b, by)))
        for x in R.extract_json(text)["votes"]:
            out[x["pair"]] = {"same": bool(x["same"]), "broader": x.get("broader")}
    return out


def main() -> None:
    rt = json.loads((R.CST / "outputs" / "t01_retest.json").read_text())
    first = json.loads((R.CST / "outputs" / "t01_tcore_votes.json").read_text())["votes"]
    sample = sorted(rt["second"])
    sets = R._sets("v1")
    by = {**sets["C_anth"].by_id(), **sets["C_spec"].by_id()}
    pairs = [tuple(p.split("|")) for p in sample]
    p1 = run_pass(pairs, by, 1, "p1")
    p2 = run_pass(pairs, by, 2, "p2")
    common = [p for p in sample if p in p1 and p in p2]
    lab = lambda d: [str(d[p]["same"]) for p in common]  # noqa: E731
    orig = [str(first[p]["claude"]) for p in common]
    res = {"n": len(common), "tight_retest": retest_agreement(lab(p1), lab(p2)),
           "orig_retest": {k: rt[k] for k in ("agreement", "kappa")},
           "tight_p1_vs_orig_first": retest_agreement(orig, lab(p1)),
           "tight_yes_rate": [sum(p1[p]["same"] for p in common) / len(common),
                              sum(p2[p]["same"] for p in common) / len(common)],
           "orig_yes_rate": sum(o == "True" for o in orig) / len(common),
           "verdict_rule": "R_prompt if tight retest kappa >= 0.8, else R_construct",
           "p1": p1, "p2": p2}
    res["verdict"] = "R_prompt" if res["tight_retest"]["kappa"] >= 0.8 else "R_construct"
    (R.CST / "outputs" / "t01_d17.json").write_text(json.dumps(res, indent=1) + "\n")
    print({k: res[k] for k in ("n", "tight_retest", "orig_retest", "tight_p1_vs_orig_first",
                               "tight_yes_rate", "orig_yes_rate", "verdict")})


if __name__ == "__main__":
    main()
