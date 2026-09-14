#!/usr/bin/env python3
"""Stage-2 calibration report (spec A6): harness vs two independent judges on real replies.

    python3 papers/kdg_panel/scripts/stage2_report.py

Rater 1 = the Claude judge, rater 2 = the GPT judge (both labelled the same 200 real pilot
replies independently). Writes ``data/calibration_stage2_report.json``; the pilot gate is
applied only if harness-vs-each-rater and rater-vs-rater all reach 0.95.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from deepsteer.kdg.calibration import agreement, cohen_kappa, run_harness_on_set  # noqa: E402
from deepsteer.kdg.harness import KDG_HARNESS_VERSION  # noqa: E402

DATA = REPO / "papers/kdg_panel/data"


def main() -> int:
    items = json.loads((DATA / "calibration_set_v2_real.json").read_text())["items"]
    judges = {
        p.stem.split("_rater2_", 1)[1]: json.loads(p.read_text())["labels"]
        for p in sorted(DATA.glob("calibration_set_v2_real_rater2_*.json"))
    }
    if len(judges) < 2:
        raise SystemExit(f"need two judge files, have {list(judges)}")
    pred = run_harness_on_set(items)
    names = list(judges)
    rep = {
        "harness_version": KDG_HARNESS_VERSION,
        "n": len(items),
        "judges": names,
        "harness_vs_judge": {n: agreement(pred, judges[n]) for n in names},
        "kappa_harness_vs_judge": {n: cohen_kappa(pred, judges[n]) for n in names},
        "judge_vs_judge": agreement(judges[names[0]], judges[names[1]]),
        "kappa_judge_vs_judge": cohen_kappa(judges[names[0]], judges[names[1]]),
        "judge_null_counts": {n: sum(x is None for x in judges[n]) for n in names},
    }
    rep["disagreements"] = [
        {
            "id": it["id"],
            "cell": it["cell"],
            "harness": p,
            **{n: judges[n][i] for n in names},
            "text": it["text"][:100],
        }
        for i, (it, p) in enumerate(zip(items, pred))
        if any(judges[n][i] != p for n in names)
    ]
    rep["meets_target"] = (
        all(v >= 0.95 for v in rep["harness_vs_judge"].values()) and rep["judge_vs_judge"] >= 0.95
    )
    (DATA / "calibration_stage2_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "disagreements"}, indent=1))
    for d in rep["disagreements"][:20]:
        print("  DISAGREE", d)
    return 0 if rep["meets_target"] else 1


if __name__ == "__main__":
    sys.exit(main())
