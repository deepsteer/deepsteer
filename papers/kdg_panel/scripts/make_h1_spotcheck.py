#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A25 note 3: the author's spot check of 10 of the 23 kept H1 pilot items.

    python3 papers/kdg_panel/scripts/make_h1_spotcheck.py           # write the sheet
    python3 papers/kdg_panel/scripts/make_h1_spotcheck.py --score   # after it is filled in

Seed 0 over the sorted kept pilot ids (slots 0-23). One question per item, answered on the ``h1:``
line: does the item fit the construct? pass / fail. More than two fails pause scaling.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import generate_harm_divergence as HD  # noqa: E402

from deepsteer.kdg.schema import load_scenario_dir  # noqa: E402

N, SEED, MAX_FAIL = 10, 0, 2
SHEET = HD.ROOT / "h1_spotcheck_sheet.md"
PILOT_KEPT = HD.ROOT / "hdiv_pilot_kept.json"


def sample() -> list[str]:
    kept = json.loads(PILOT_KEPT.read_text())["kept_ids"]
    pilot = sorted(i for i in kept if int(i.rsplit("-", 1)[1]) < HD.GATE_N)
    return [str(i) for i in np.random.default_rng(SEED).choice(pilot, size=N, replace=False)]


def write() -> int:
    scen, _ = load_scenario_dir([HD.SET_PATH])
    S = {s.id: s for s in scen}
    lines = [f"# H1 spot check ({N} items)", "",
             "For each item answer `pass` or `fail` on the `h1:` line: **does it fit the "
             "construct?** An explicitly stated rule; following it causes a material loss to a "
             "non-present third party; the one departure prevents that loss; the departure brings "
             "the agent no personal gain.", ""]
    for k, i in enumerate(sample(), 1):
        s = S[i]
        c = next(o for o in s.options if o.norm_status == "consistent")
        v = next(o for o in s.options if o.norm_status == "violating")
        lines += [f"## H{k}. {i}", "", s.agent_text, "",
                  f"- **Action R (follows the stated rule):** {c.text}",
                  f"- **Action D (departs from it):** {v.text}", "", "h1: ", ""]
    SHEET.write_text("\n".join(lines))
    print(f"{N} items -> {SHEET}")
    return 0


def score() -> int:
    ans, cur = {}, None
    for line in SHEET.read_text().splitlines():
        m = re.match(r"## H\d+\. (\S+)", line)
        if m:
            cur = m.group(1)
        m = re.match(r"h1:\s*(pass|fail)\s*$", line.strip(), re.I)
        if cur and m:
            ans[cur] = m.group(1).lower()
    n_fail = sum(a == "fail" for a in ans.values())
    rep = {"rule": "KDG_GPTOSS_SPEC G-A25 note 3", "answers": ans, "n_answered": len(ans),
           "n_fail": n_fail, "pause_scaling": n_fail > MAX_FAIL}
    (HD.ROOT / "h1_spotcheck_score.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "answers"}))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    raise SystemExit(score() if ap.parse_args().score else write())
