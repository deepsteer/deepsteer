#!/usr/bin/env python3
"""The stack-parity verdict (ST G1 rider pod 2; Amendment 9, pre-registered 2026-10-19): two
``stack_read.py`` outputs on the same rows, compared on the Amendment 6 ladder, per head.

    python stack_compare.py --a A.npz --b B.npz --out VERDICT.json [--tie RECORD.json]

For each head (bf16, fp32): floor = the larger of the two stacks' replicate floors (alone vs in a
batch of identical copies); bar = max(0.05, 2 x floor); measurement = max |alone_a - alone_b|
over rows x letters; negative control = max |off_b - alone_a| (one position early on stack b),
which must exceed the bar or the read is uncalibrated; inside = calibrated and measurement <= bar.
``--tie``: the screen driver's record on stack a, whose padded check read the first rows alone
through the readout of record (``read_ids``); this script's stack-a alone read must equal it
(to its 4-decimal rounding), which ties ``stack_read`` to the record path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

KDG_BAR, FLOOR_FACTOR = 0.05, 2.0


def ladder(a: dict, b: dict, head: str) -> dict:
    floor_a = float(np.abs(a[f"alone_{head}"] - a[f"replicate_{head}"]).max())
    floor_b = float(np.abs(b[f"alone_{head}"] - b[f"replicate_{head}"]).max())
    floor = max(floor_a, floor_b)
    bar = max(KDG_BAR, FLOOR_FACTOR * floor)
    d = np.abs(a[f"alone_{head}"] - b[f"alone_{head}"])
    neg = float(np.abs(b[f"off_{head}"] - a[f"alone_{head}"]).max())
    meas = float(d.max())
    return {"floor_a": floor_a, "floor_b": floor_b, "bar": bar, "max_abs_nats": meas,
            "mean_abs_nats": float(d.mean()), "per_row_max": np.round(d.max(1), 4).tolist(),
            "negative_control": neg, "calibrated": neg > bar,
            "inside": bool(neg > bar and meas <= bar)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="stack of record (torch 2.6)")
    ap.add_argument("--b", type=Path, required=True, help="candidate stack (vLLM venv)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tie", type=Path, default=None)
    a_ = ap.parse_args()
    A, B = dict(np.load(a_.a)), dict(np.load(a_.b))
    out = {"a": json.loads(a_.a.with_suffix(".json").read_text()),
           "b": json.loads(a_.b.with_suffix(".json").read_text()),
           "ladders": {h: ladder(A, B, h) for h in ("bf16", "fp32")}}
    if a_.tie is not None:
        rec = json.loads(a_.tie.read_text())
        ref = np.array(rec["dose0_position"]["bf16"]["per_row"]["letter_logprobs_alone"])
        out["tie_to_record_max_abs"] = float(np.abs(A["alone_bf16"][: len(ref)] - ref).max())
        out["tie_ok"] = out["tie_to_record_max_abs"] <= 1e-3
    a_.out.write_text(json.dumps(out, indent=1))
    for h, v in out["ladders"].items():
        print(f">> stack parity [{h} head]: max {v['max_abs_nats']:.4f} vs bar {v['bar']:.4f} "
              f"(floors {v['floor_a']:.4f}/{v['floor_b']:.4f}, negative control "
              f"{v['negative_control']:.2f}) -> {'INSIDE' if v['inside'] else 'outside'}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
