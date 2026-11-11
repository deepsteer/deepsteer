#!/usr/bin/env python3
"""Before 2-ctx trains (the author, 2026-11-11): how P-ctx's training rollouts refer to the
constitution (zero GPU; no rollout is rewritten).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/pctx_reference_check.py

P-ctx's training set exactly as `cst_pressure_train_pod.ctx_examples` builds it: A0s's rollouts
(`outputs/pressure/roll/a0s`), not capped and with an answer, outside the held-out cell. Each
rollout's whole response (trace and answer, both carry the loss) is matched, case-insensitive,
against two patterns, **registered here (``POSSESSIVE``, ``POINTING`` below) before any rollout
is read**:

- **possessive**: the model speaks of the values as its own ("my constitution", "my values", "my
  principles", "my guidelines", "my core values", "my commitments");
- **pointing at context**: the model points at a text in its prompt ("the document above", "the
  provided text", "the constitution in the system prompt", "the system prompt says", "according
  to the constitution provided").

Reported by scenario kind: the share of rollouts with at least one possessive reference and with at
least one pointing reference (and with both). **Rule (the author's):** if the pointing share is
above 10%, the author hears before the launch, with the count of rollouts that excluding them would
drop. Writes `outputs/pctx_reference_check.json`.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "kdg_panel/scripts")]
import cst_pressure_pod as R  # noqa: E402
import cst_pressure_train_pod as T  # noqa: E402

CST = HERE.parent
POSSESSIVE = re.compile(r"\bmy (own )?(constitution|values|principles|guidelines|core values|"
                        r"commitments)\b", re.I)
POINTING = re.compile(
    r"\bthe (document|text|constitution|guidelines|framework) (provided |given |shared |supplied )?"
    r"(above|here|in (the|my|this) (system (prompt|message)|context|prompt))\b"
    r"|\bthe (provided|given|attached|supplied|above|preceding|included) "
    r"(document|text|constitution|guidelines|framework)\b"
    r"|\b(the )?system (prompt|message) (says|states|tells|instructs|includes|contains|specifies|"
    r"lays out)\b"
    r"|\b(according to|per|as stated in|as laid out in) the (constitution|document|text) (provided|"
    r"above|given|in the system (prompt|message))\b", re.I)
BAR = 0.10


def main() -> None:
    out = CST / "outputs/pressure"
    sc = {s["cell_id"]: s for s in R.scen_list(out)}
    held = T.heldout_ids()
    rows = [r for r in R.rollouts(out, "a0s") if T.usable(r) and r["scenario_id"] not in held]
    by: dict[str, dict] = defaultdict(lambda: {"n": 0, "possessive": 0, "pointing": 0, "both": 0})
    examples = defaultdict(list)
    for r in rows:
        kind = sc[r["scenario_id"]]["kind"]
        pos, pt = POSSESSIVE.search(r["text"]), POINTING.search(r["text"])
        for k in (kind, "all"):
            by[k]["n"] += 1
            by[k]["possessive"] += bool(pos)
            by[k]["pointing"] += bool(pt)
            by[k]["both"] += bool(pos and pt)
        if pt and len(examples["pointing"]) < 8:
            examples["pointing"].append(pt.group(0))
        if pos and len(examples["possessive"]) < 8:
            examples["possessive"].append(pos.group(0))
    res = {"rule": "the author, 2026-11-11: pointing share above 10% comes back before the launch",
           "training_set": "A0s, usable, outside the held-out cell (ctx_examples)",
           "kinds": {k: v | {"possessive_share": v["possessive"] / v["n"],
                             "pointing_share": v["pointing"] / v["n"]} for k, v in by.items()},
           "matched_phrases": dict(examples)}
    allk = res["kinds"]["all"]
    res["pointing_above_bar"] = allk["pointing_share"] > BAR
    res["dropped_if_excluded"] = allk["pointing"]
    (CST / "outputs/pctx_reference_check.json").write_text(json.dumps(res, indent=1) + "\n")
    for k, v in res["kinds"].items():
        print(f"{k:6s} n {v['n']:5d} possessive {v['possessive_share']:.3f} "
              f"pointing {v['pointing_share']:.3f} both {v['both']}")
    print("pointing above 10%:", res["pointing_above_bar"], "| dropped if excluded:",
          res["dropped_if_excluded"])
    print("matched:", json.dumps(res["matched_phrases"]))


if __name__ == "__main__":
    main()
