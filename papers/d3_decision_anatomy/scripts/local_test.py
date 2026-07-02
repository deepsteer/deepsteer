#!/usr/bin/env python3
"""Local gate for C1 typing prep (zero-GPU): request-twins, manifest typing + alignment, and the
behavioral-discrimination screen's gap-logic (via a stub model). No model load.
Run: python papers/d3_decision_anatomy/scripts/local_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[1] / "d2_decision_coupling" / "scripts"))

import patch_stimuli as ps  # noqa: E402
from deepsteer.datasets import get_request_twins  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


class StubModel:
    """Returns a scripted completion per prompt substring (for the screen gap-logic)."""
    def __init__(self, rules):
        self.rules = rules

    def generate(self, prompt, **kw):
        text = next((v for k, v in self.rules.items() if k in prompt), "Sure, here is how.")
        return type("R", (), {"text": text})()


def test_request_twins():
    print("request-twins:")
    ts = get_request_twins()
    check("count >= 20", len(ts) >= 20, f"n={len(ts)}")
    check("all triples (foundation, following, violating)", all(len(t) == 3 for t in ts))
    check("following != violating", all(a != b for _, a, b in ts))
    check("nontrivial shared prefix (>=20 chars) each pair",
          all(ps._align(a, b)["shared_prefix_chars"] >= 20 for _, a, b in ts))
    from collections import Counter
    fnd = Counter(f for f, _, _ in ts)
    check("spans all 6 MFT foundations", len(fnd) == 6, str(dict(fnd)))


def test_manifest():
    print("typed manifest (construct-audit type blocks + alignment):")
    m = ps.build_manifest()
    tw, rq, xs = m["compositional_twins"], m["request_twins"], m["xstest_generalization"]
    check("twins outcome=judgment + refusal EXPECTED-FLAT",
          tw["type_block"]["outcome_variable"] == "judgment-decision readout"
          and "EXPECTED-FLAT" in tw["type_block"]["role"])
    check("request-twins outcome=refusal + MINIMAL-PAIR role",
          rq["type_block"]["outcome_variable"].startswith("refusal")
          and "MINIMAL-PAIR" in rq["type_block"]["role"])
    check("xstest typed as GENERALIZATION (not minimal pair)",
          "GENERALIZATION" in xs["type_block"]["role"])
    check("every type block carries source_dataset + extraction_commit",
          all(m[k]["type_block"].get("source_dataset") and m[k]["type_block"].get("extraction_commit")
              for k in m))
    al = rq["pairs"][0]["align"]
    check("alignment: shared prefix + distinct flipped spans",
          al["shared_prefix_chars"] > 10
          and al["flipped_span_following"] != al["flipped_span_violating"])


def test_screen_logic():
    print("behavioral-discrimination screen gap-logic (stub model):")
    from b1_judgment_direction import build_prompt  # noqa: F401 (import path sanity)
    # judgment flips: model says wrong for the moral action, acceptable for the neutral one.
    jm = StubModel({"protect": "acceptable", "humiliate": "wrong"})
    check("judgment flip detected across a twin",
          ps._judgment_flips(jm, "She lied to protect", "She lied to humiliate"))
    jm_flat = StubModel({"protect": "acceptable", "humiliate": "acceptable"})
    check("no judgment flip -> pair dropped", not ps._judgment_flips(jm_flat,
          "She lied to protect", "She lied to humiliate"))
    # refusal differs: comply on following, refuse on violating.
    rm = StubModel({"supported": "Sure, here is how.", "worse": "I'm sorry, I can't help with that."})
    check("refusal difference detected across a request-twin",
          ps._refusal_differs(rm, "help ... feels supported", "help ... feels worse"))
    rm_flat = StubModel({"supported": "Sure!", "worse": "Sure!"})
    check("no refusal difference -> pair dropped",
          not ps._refusal_differs(rm_flat, "help ... supported", "help ... worse"))


def main():
    print("=== C1 typing-prep local gate ===\n")
    test_request_twins()
    test_manifest()
    test_screen_logic()
    print()
    if FAILS:
        print(f"FAILED: {FAILS}"); sys.exit(1)
    print("ALL LOCAL CHECKS PASSED. Behavioral screen runs on MPS/pod: patch_stimuli.py --screen")


if __name__ == "__main__":
    main()
