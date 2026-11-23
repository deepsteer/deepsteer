#!/usr/bin/env python3
"""Amendment 6's zero-GPU first step (GPC_PLAN.md §14): how much of C1's training grid covers the depth cell's acts.
Counts only, by keyword patterns over each kept slot's frame and goal option (the subject wrote the slots; no text is
printed and no Claude call reads them). Each primary scenario's act:
  - Healthcare-Management, Ai-Healthcare-Administrator: changing or inventing data values to meet a metric;
  - Ai-Legal-Reviewer-Conflict: leaving out or softening findings in a report so a check passes;
  - Ai-Education-Assistant: dropping or burying required content so a score rises.
Beside: the grid's kind weights, and the share of slots in work frames with score pressure.

    python z_gpc4_grid_coverage.py   # writes outputs/gpc4_prep/grid_coverage.json
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

GPC = Path(__file__).resolve().parent.parent
SLOTS = GPC / "outputs/gpc1_v1/conduct/slots.json"
KEPT = GPC / "outputs/gpc1_v1/conduct/kept_c1.jsonl"
OUT = GPC / "outputs/gpc4_prep/grid_coverage.json"
ACTS = {
    "data_falsification": r"\b(alter|alters|altering|change|changes|changing|edit|edits|editing|overwrit\w*|adjust\w*|"
                          r"insert\w*|fill in|fabricat\w*|invent\w*|backdat\w*|inflat\w*|round\w* up|modif\w*|"
                          r"rewrite|rewrit\w*)\b.{0,60}\b(data|record|records|value|values|figure|figures|number|"
                          r"numbers|score|scores|log|logs|entry|entries|date|dates|reading|readings|result|results)\b",
    "report_suppression": r"\b(omit\w*|leave out|leav\w* out|left out|drop\w*|remov\w*|soften\w*|downplay\w*|"
                          r"play\w* down|not mention\w*|without mentioning|reword\w*|rephras\w*|hide|hid\w*|bury|"
                          r"buri\w*|exclud\w*|strip\w*)\b.{0,60}\b(finding|findings|risk|risks|issue|issues|flag|flags|"
                          r"warning|warnings|concern|concerns|defect|defects|violation|violations|report|summary)\b",
    "content_exclusion": r"\b(omit\w*|leave out|leav\w* out|drop\w*|remov\w*|exclud\w*|skip\w*|cut\w*|bury|buri\w*)\b"
                         r".{0,60}\b(material|materials|content|perspective|perspectives|reading|readings|resource|"
                         r"resources|section|sections|topic|topics|lesson|lessons|unit|units|curriculum|source|sources)\b",
}
SCEN = {"Healthcare-Management": "data_falsification", "Ai-Healthcare-Administrator": "data_falsification",
        "Ai-Legal-Reviewer-Conflict": "report_suppression", "Ai-Education-Assistant": "content_exclusion"}
FRAME = {"healthcare": r"health|hospital|clinic|patient|medical|care",
         "legal_compliance": r"legal|law|contract|compliance|audit|regulat",
         "education": r"educat|school|teach|course|student|curricul|tutor|learning"}


def main() -> None:
    slots = json.loads(SLOTS.read_text())["kept"]
    kept = [json.loads(x)["slot_id"] for x in KEPT.read_text().splitlines()]
    rows = [slots[k] for k in kept if k in slots]
    press = [r for r in rows if str(r.get("pressure")) == "True"]
    out = {"rule": "GPC_PLAN.md §14 (Amendment 6, the zero-GPU first step); keyword counts, unvalidated by reading",
           "kept": len(rows), "pressure": len(press), "acts": {}, "frames": {}, "kinds": {}, "settings": {}}
    for name, pat in ACTS.items():
        rx = re.compile(pat, re.I)
        hit = [r for r in press if rx.search(str(r.get("goal_option", "")))]
        out["acts"][name] = {"n": len(hit), "share_of_pressure": len(hit) / len(press),
                             "by_setting": dict(collections.Counter(r["setting"] for r in hit))}
    for name, pat in FRAME.items():
        rx = re.compile(pat, re.I)
        out["frames"][name] = sum(bool(rx.search(str(r.get("frame", "")))) for r in press)
    out["kinds"] = dict(collections.Counter(r.get("kind") for r in press).most_common())
    out["settings"] = dict(collections.Counter(r.get("setting") for r in press))
    work_sp = [r for r in press if r["setting"] == "work"]
    out["work_with_score_pressure"] = {"n": len(work_sp), "share_of_pressure": len(work_sp) / len(press),
                                       "score_pressures": dict(collections.Counter(r.get("score_pressure") for r in work_sp).most_common(6))}
    # each scenario's act within its frame (the closest match the grid has)
    out["by_scenario"] = {}
    for s, act in SCEN.items():
        rx = re.compile(ACTS[act], re.I)
        fr = re.compile(FRAME["healthcare" if "Health" in s else "legal_compliance" if "Legal" in s else "education"], re.I)
        out["by_scenario"][s] = {"act": act, "act_any_frame": sum(bool(rx.search(str(r.get("goal_option", "")))) for r in press),
                                 "act_in_frame": sum(bool(rx.search(str(r.get("goal_option", ""))) and fr.search(str(r.get("frame", ""))))
                                                     for r in press)}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
