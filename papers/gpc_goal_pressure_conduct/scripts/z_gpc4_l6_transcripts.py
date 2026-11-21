#!/usr/bin/env python3
"""Launch 6's ODCV-Bench gate runs, for launch 7's final judge pass (GPC_PLAN.md §11.22, the lead's morning
ruling 4): each run's record and saved history, one JSON line a run, labeled as run under the 4,096-token
cap, written to the pod's inputs (``outputs/gpc4_in/l6_odcv_base.jsonl``; gitignored, synced by name).

    python z_gpc4_l6_transcripts.py
"""

from __future__ import annotations

import json
from pathlib import Path

GPC = Path(__file__).resolve().parents[1]
SRC = GPC / "outputs/gpc4_v1/vm_results/odcv_base"
OUT = GPC / "outputs/gpc4_in/l6_odcv_base.jsonl"
LABEL = "launch 6, run under the 4,096-token cap"


def main() -> None:
    rows = []
    for rj in sorted(SRC.glob("*/run.json")):
        r = json.loads(rj.read_text())
        mr = rj.parent / "messages_record.txt"
        rows.append({k: r.get(k) for k in ("run_id", "scenario", "variant", "arm", "container_error",
                                           "tool_timeouts")}
                    | {"history": mr.read_text(errors="ignore") if mr.exists() else "", "label": LABEL})
    if len(rows) != 80:
        raise SystemExit(f"launch 6's gate has 80 runs; found {len(rows)} under {SRC}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f">> wrote {OUT.relative_to(GPC.parents[1])}: {len(rows)} runs, "
          f"{sum(bool(r['history']) for r in rows)} with a history")


if __name__ == "__main__":
    main()
