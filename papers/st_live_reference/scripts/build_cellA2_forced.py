#!/usr/bin/env python3
"""The forced-choice cell's items (ST_PLAN Amendment 46 item 2): Muse cell A′'s 51 kept items with
the neutral (defer) option removed, the rule-consistent and the violating options kept, every text
field unchanged; each item revalidated (two options, one consistent and one violating).

    python3 papers/st_live_reference/scripts/build_cellA2_forced.py \
        <outputs/a2m/muse_glimmer_30b/cellA2_selection.json>
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from deepsteer.kdg.schema import load_scenarios, save_scenarios, validate_scenario  # noqa: E402

DATA = HERE.parent / "data"
POOL = DATA / "st_fxh_cellA2_pool_v1.json"
OUT = DATA / "st_fxh_cellA2_forced_v1.json"


def forced(s):
    """The item without its neutral option; the covariates record what was removed."""
    keep = [o for o in s.options if o.norm_status != "neutral"]
    gone = [o.option_id for o in s.options if o.norm_status == "neutral"]
    if len(gone) != 1 or len(keep) != 2:
        raise SystemExit(f"{s.id}: expected one neutral option, found {gone}")
    cov = {**s.covariates, "n_options": 2, "removed_option": gone[0],
           "forced_choice": "Amendment 46: the defer option removed"}
    return dataclasses.replace(s, options=keep, covariates=cov)


def main() -> None:
    sel = json.loads(Path(sys.argv[1]).read_text())
    pool = {s.id: s for s in load_scenarios(POOL)[0]}
    items = [forced(pool[i]) for i in sel["kept"]]
    for s in items:
        errs = [e for e in validate_scenario(s) if "options" in e or "consistent" in e]
        if errs:
            raise SystemExit(f"{s.id}: {errs}")
    save_scenarios(OUT, items, {"rule": "ST_PLAN Amendment 46 item 2: cell A' forced choice",
                                "source": POOL.name, "selection": "Muse cell A' (a2m), 51 kept",
                                "n": len(items), "removed": "the neutral option of each item"})
    print(f">> wrote {OUT.name}: {len(items)} items")


if __name__ == "__main__":
    main()
