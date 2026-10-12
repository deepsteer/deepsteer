#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A26 note 2: §40's transitions split by the neutral option's kind (descriptive).

    python3 papers/kdg_panel/scripts/analyze_h1_deferral.py

Each H1 scenario's neutral option is an explicit deferral (it waits for, or asks for, the approver's
decision) or a log-and-notify (it records the request and contacts someone without waiting), by the
regexes fixed in the note. Reported: class counts, the dose-0 -> medium transition table with the
neutral column and row split by class, and how often the agent text names an approval gate.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h1_cell as HC  # noqa: E402
import analyze_phase1_session_a as A  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

DEFER = [re.compile(r"\b(wait|hold|until|pending)\b"),
         re.compile(r"\b(ask|request|seek)\w*\b.*\b(decision|approv|authori[sz]|sign)"),
         re.compile(r"\bapproval\b|\bauthori[sz]ed response\b")]
GATE = re.compile(r"approv|authori[sz]|sign[- ]off|signature|permission|clearance")


def kind(text: str) -> str:
    t = text.lower()
    return "explicit_deferral" if any(p.search(t) for p in DEFER) else "log_and_notify"


def main() -> int:
    H, _ = load_scenario_dir([HC.HDIR / "hdiv_set.json"])
    kept = set(json.loads((HC.HDIR / "hdiv_keep.json").read_text())["kept_ids"])
    S = {s.id: s for s in H if s.id in kept}
    neu = {i: next(o for o in s.options if o.norm_status == "neutral") for i, s in S.items()}
    cls = {i: kind(o.text) for i, o in neu.items()}
    label = {i: {o.option_id: o.norm_status for o in s.options} for i, s in S.items()}
    top, _ = HC.dose0(A._rows(A.OUT / "p2m/h1/gpt_oss_20b/h1/dl_chat_neutral.jsonl"))
    rows = [r for p in ("p2m", "p2n")
            for r in A._rows(A.OUT / p / "h1/gpt_oss_20b/h1/ds_main.jsonl")
            if r["status"] != "not_run" and r.get("token_identity") and r.get("option_id")]

    def name(i: str, oid: str) -> str:
        lab = label[i][oid]
        return f"neutral:{cls[i]}" if lab == "neutral" else lab

    T = Counter((name(r["scenario_id"], top[r["scenario_id"]]),
                 name(r["scenario_id"], r["option_id"])) for r in rows)
    rep = {"rule": "KDG_GPTOSS_SPEC G-A26 note 2, descriptive", "n": len(S),
           "neutral_classes": dict(Counter(cls.values())),
           "agent_text_names_approval_gate": sum(bool(GATE.search(s.agent_text.lower()))
                                                 for s in S.values()),
           "transitions": {f"{a} -> {b}": n for (a, b), n in sorted(T.items())},
           "examples": {c: [neu[i].text for i in sorted(neu) if cls[i] == c][:3]
                        for c in ("explicit_deferral", "log_and_notify")}}
    (A.DATA / "analysis_h1_deferral.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
