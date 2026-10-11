#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A25 item 2 and note 2: the anchored P4 instrument against the author's 20.

    python3 papers/kdg_panel/scripts/analyze_p4_gate.py

kappa between the both-rater anchored P4 labels and the author's yes/no, excluding the anchor and
the author's need-context items (gate >= 0.4; undefined when either side has one class only);
specificity on the author's "no" items; rater answers on need-context items; and the author
sample's own prevalence estimate (yes among decided items, Wilson 95%).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kdg_harmony as kh  # noqa: E402
import rate_compliance_harm as CH  # noqa: E402
import rate_divergent_patterns as RD  # noqa: E402


def main() -> int:
    ans = json.loads((RD.DATA / "p4_check_answers.json").read_text())["answers"]
    anchor = json.loads((RD.DATA / "p4_anchor.json").read_text())["anchor_id"]
    R = [json.loads((RD.DATA / f"pattern_p4a_ratings_{r}.json").read_text()) for r in RD.RATERS]
    lab = {i: all(x[i] for x in R) for i in ans}
    scored = [i for i, a in ans.items() if i != anchor and a in ("yes", "no")]
    hum = [ans[i] == "yes" for i in scored]
    ins = [lab[i] for i in scored]
    defined = 0 < sum(hum) < len(hum) or 0 < sum(ins) < len(ins)
    k = CH.kappa(hum, ins) if defined else None
    decided = [a for a in ans.values() if a in ("yes", "no")]
    n_yes = sum(a == "yes" for a in decided)
    rep = {"rule": "KDG_GPTOSS_SPEC G-A25 item 2, note 2", "anchor": anchor,
           "n_scored": len(scored), "author_yes_in_scored": sum(hum),
           "instrument_yes_in_scored": sum(ins), "kappa": k, "gate": 0.4,
           "gate_cleared": bool(k is not None and k >= 0.4),
           "verdict": ("P4 measurable" if k is not None and k >= 0.4
                       else "P4 not measurable with this instrument"),
           "specificity_on_author_no": (sum(not lab[i] for i in scored if ans[i] == "no")
                                        / max(1, sum(ans[i] == "no" for i in scored))),
           "by_rater_yes_on_20": {r: sum(x[i] for i in ans) for r, x in zip(RD.RATERS, R)},
           "anchor_rated_yes_by_both": lab[anchor],
           "need_context_items": {i: [x[i] for x in R] for i, a in ans.items()
                                  if a == "need context"},
           "author_prevalence": {"yes": n_yes, "decided": len(decided),
                                 "wilson95": list(kh.wilson(n_yes, len(decided)))}}
    (RD.DATA / "analysis_p4_gate.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
