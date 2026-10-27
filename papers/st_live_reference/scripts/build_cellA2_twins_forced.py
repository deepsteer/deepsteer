#!/usr/bin/env python3
"""The twins' forced-choice cell (ST_PLAN Amendment 52 item 1): the matched cell's 51 harm-reversed
twins with the neutral (defer) option removed by Amendment 46 item 2's rule, the rule-consistent and
the violating (now protective) options kept, every text field unchanged; each item revalidated.

    python3 papers/st_live_reference/scripts/build_cellA2_twins_forced.py
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import build_cellA2_forced as BF  # noqa: E402

from deepsteer.kdg.schema import load_scenarios, save_scenarios, validate_scenario  # noqa: E402

MATCHED = BF.DATA / "st_fxh_cellA2_matched_v1.json"
OUT = BF.DATA / "st_fxh_cellA2_twins_forced_v1.json"
TWIN = "-T"


def main() -> None:
    twins = [s for s in load_scenarios(MATCHED)[0] if s.id.endswith(TWIN)]
    items = []
    for s in twins:
        f = BF.forced(s)
        cov = {**f.covariates, "forced_choice": "Amendment 52: the defer option removed "
                                                "(Amendment 46 item 2's rule)"}
        items.append(dataclasses.replace(f, covariates=cov))
    for s in items:
        errs = [e for e in validate_scenario(s) if "options" in e or "consistent" in e]
        if errs:
            raise SystemExit(f"{s.id}: {errs}")
    if len(items) != 51:
        raise SystemExit(f"expected 51 twins, found {len(items)}")
    save_scenarios(OUT, items, {"rule": "ST_PLAN Amendment 52 item 1: the twins' forced choice",
                                "source": MATCHED.name, "n": len(items),
                                "removed": "the neutral option of each twin"})
    print(f">> wrote {OUT.name}: {len(items)} items")


if __name__ == "__main__":
    main()
