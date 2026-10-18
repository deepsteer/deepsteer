#!/usr/bin/env python3
"""Amendment 6 item 3: a banked trace-length pilot read at caps 1,536 and 2,048 (zero GPU).

    python3 papers/st_live_reference/scripts/pilot_caps.py \
        --pilot OUT/<model>/pilot/pilot_<level>.jsonl

The pilot runs at cap 4,096, so the share of rollouts that would truncate at a lower cap is the
share whose tokens-to-answer exceed it. Per cap: truncation rate, p95 and median tokens-to-answer
(rollouts the guard never ran are excluded; a rollout that hit 4,096 counts as over both caps).
Timing is in the model's screen record (``timing_pilot``). G1 prices a dated cap amendment at
2,048 against the GPT-OSS-20B replacement from these numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

CAPS = (1536, 2048)


def report(rows: list[dict], caps=CAPS) -> dict:
    run = [r for r in rows if r["status"] != "not_run"]
    t = np.array([r["tokens_to_answer"] for r in run], float)
    out: dict = {"n_run": len(run), "n_not_run": len(rows) - len(run),
                 "statuses": {s: sum(r["status"] == s for r in run)
                              for s in sorted({r["status"] for r in run})},
                 "p95_tokens_to_answer": float(np.percentile(t, 95)) if len(t) else None,
                 "median_tokens_to_answer": float(np.median(t)) if len(t) else None}
    for c in caps:
        out[f"truncation_at_{c}"] = float(np.mean(t > c)) if len(t) else None
        out[f"fits_p95_at_{c}"] = bool(len(t) and np.percentile(t, 95) <= c)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in a.pilot.read_text().splitlines() if x.strip()]
    rep = report(rows)
    print(json.dumps(rep, indent=1))
    (a.pilot.parent / f"{a.pilot.stem}_caps.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
