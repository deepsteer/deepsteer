#!/usr/bin/env python3
"""A5 (calibration): proto-refusal continuity. cos(proto-refusal_base, refusal_instruct) at
layer 16 from the committed Point A / Point B vectors. The refusal analog of Paper 5's
cos(base, fresh) crystallization measurement.

Pre-registered (CALIBRATION_PREREG.md A5): cos >= 0.50 queues the per-checkpoint
refusal-crystallization curve into B3; else the instruct gate is substantially a
post-training construction. Descriptive, no headline gate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepsteer.directions import extraction as du  # noqa: E402

P2 = HERE.parent / "outputs" / "phase2"
OUT = P2 / "calibration"
THRESHOLD = 0.50


def main() -> None:
    OUT.mkdir(exist_ok=True)
    proto = du.unit_vector(np.load(P2 / "refusal_base.npz")["refusal"])
    gate = du.unit_vector(np.load(P2 / "refusal_instruct.npz")["refusal"])
    cos = float(du.cosine(proto, gate))
    triggered = cos >= THRESHOLD
    res = {
        "measurement": "cos(proto-refusal_base, refusal_instruct) @ L16",
        "cos": round(cos, 4),
        "threshold": THRESHOLD,
        "queue_crystallization_curve_into_B3": triggered,
        "reading": ("high continuity: post-training SELECTS a pre-existing pretraining refusal "
                    "direction (queue per-checkpoint curve into B3)" if triggered else
                    "low continuity: the instruct gate is substantially a post-training "
                    "construction (no curve queued)"),
        "note": "descriptive; refusal analog of Paper 5 cos(base, fresh) crystallization",
    }
    (OUT / "a5_proto_refusal_continuity.json").write_text(json.dumps(res, indent=2))
    print(f"A5 cos(proto-refusal_base, refusal_instruct) @L16 = {cos:.4f}  "
          f"(threshold {THRESHOLD} -> {'TRIGGERED, queue B3 curve' if triggered else 'not triggered'})")


if __name__ == "__main__":
    main()
