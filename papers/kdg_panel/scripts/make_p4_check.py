#!/usr/bin/env python3
"""KDG_GPTOSS_SPEC G-A25 item 2: the author's blind 20-item P4 sample from the 175 divergent set.

    python3 papers/kdg_panel/scripts/make_p4_check.py            # write the sheet
    python3 papers/kdg_panel/scripts/make_p4_check.py --read     # parse the author's answers

Seed 0 over the sorted id list. Each item shows the scenario, Action R (follows the stated rule) and
Action D (departs from it), and one question answered on the ``p4:`` line: is Action D defensible
all things considered? yes / no / need context. No rater answer or model behaviour is shown.
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
import rate_divergent_patterns as RD  # noqa: E402

N, DRAW, SEED = 20, 30, 0  # G-A25 note: P1-P20 of the seed-0 draw of 30
SHEET = RD.DATA / "p4_check_sheet.md"
ANSWERS = RD.DATA / "p4_check_answers.json"


def sample() -> list[str]:
    div = json.loads((RD.DATA / "gptoss_divergent_175.json").read_text())
    ids = sorted(div["divergent_violating"] + div["divergent_neutral"])
    draw = np.random.default_rng(SEED).choice(ids, size=DRAW, replace=False)
    return [str(i) for i in draw[:N]]


def write() -> int:
    its = {it["id"]: it for it in RD.items()}
    lines = [f"# G-A25 P4 check ({N} items)", "",
             "For each item answer on the `p4:` line: **Is Action D defensible all things "
             "considered?** `yes`, `no` or `need context`.", ""]
    for k, i in enumerate(sample(), 1):
        it = its[i]
        lines += [f"## P{k}. {i}", "", it["text"], "",
                  f"- **Action R (follows the stated rule):** {it['R']}",
                  f"- **Action D (departs from it):** {it['D']}", "", "p4: ", ""]
    SHEET.write_text("\n".join(lines))
    print(f"{N} items -> {SHEET}")
    return 0


def read() -> dict:
    ans, cur = {}, None
    for line in SHEET.read_text().splitlines():
        m = re.match(r"## P\d+\. (\S+)", line)
        if m:
            cur = m.group(1)
        m = re.match(r"p4:\s*(yes|no|need context)\s*$", line.strip(), re.I)
        if cur and m:
            ans[cur] = m.group(1).lower()
    out = {"rule": "KDG_GPTOSS_SPEC G-A25 item 2", "n_answered": len(ans), "answers": ans}
    ANSWERS.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "answers"}))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--read", action="store_true")
    raise SystemExit(0 if read() else 1) if ap.parse_args().read else SystemExit(write())
