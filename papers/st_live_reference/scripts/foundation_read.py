#!/usr/bin/env python3
"""Amendment 15 item 2: the constancy statistic on each of GPT-OSS's six MFT foundation axes
(zero GPU on T1.1's bank; descriptive, with item 3's rule on ``care_harm``).

    python3 papers/st_live_reference/scripts/foundation_read.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

Same statistic as the T1.2 constancy read (``constancy_read.rollout_means`` / ``scenario_diffs`` /
``boot``, unchanged): per rollout the mean of <h, f> / |h| over the class's banked positions at
block 12; per scenario protect minus rule; over the mixed-outcome H1_train scenarios, scenario
bootstrap. Axes: ``d3_decision_anatomy/outputs/w4/gpt_oss_20b/mft_directions.npz`` (W4 14.4,
block 12, MFT v2 moral minus neutral, raw, pooled). Beside each axis: its cosine with the anchor
axis and with V_harm (t_inst). Writes ``data/t15_foundation_read_v1.json``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import constancy_read as CR  # noqa: E402

MFT = (HERE.parents[1] / "d3_decision_anatomy" / "outputs" / "w4" / "gpt_oss_20b"
       / "mft_directions.npz")
OUT = HERE.parent / "data" / "t15_foundation_read_v1.json"
RULE_AXIS = "care_harm"


def main() -> None:
    d = Path(sys.argv[1])
    rollouts = [json.loads(x) for x in (d / "gen" / "h1_rollouts_read.jsonl").open()]
    z = np.load(MFT)
    axes = {k.rsplit("_layer", 1)[0]: (z[k] / np.linalg.norm(z[k])).astype(np.float32)
            for k in z.files}
    zd = np.load(CR.DIRS)
    anchor = zd[CR.AXES[0]] / np.linalg.norm(zd[CR.AXES[0]])
    harm = zd["V_harm_t_inst"] / np.linalg.norm(zd["V_harm_t_inst"])
    chunks = [np.load(f) for f in sorted((d / "bank").glob("bank_*.npz"))]
    means = CR.rollout_means(chunks, axes)
    table: dict = {}
    for f, a in axes.items():
        row = {"cos_anchor_axis": float(a @ anchor), "cos_V_harm_t_inst": float(a @ harm)}
        for cls in CR.CLASSES:
            diffs = CR.scenario_diffs(means[(f, CR.L_STAR_SLOT, cls)], rollouts)
            row[cls] = CR.boot(np.array(list(diffs.values())))
        c = row["clause_end"]
        row["separates_clause_end"] = ("protect above rule" if c["ci95"][0] > 0 else
                                       "rule above protect" if c["ci95"][1] < 0 else "no")
        table[f] = row
    c = table[RULE_AXIS]["clause_end"]
    fires = bool(c["mean"] > 0 and c["ci95"][0] > 0)
    out = {"rule": "ST_PLAN Amendment 15 items 2-3", "layer": "block 12 (hidden index 13)",
           "table": table, "rule_axis": RULE_AXIS, "care_arm_added": fires,
           "inputs": {"mft_directions_sha256": hashlib.sha256(MFT.read_bytes()).hexdigest(),
                      "directions_sha256": hashlib.sha256(CR.DIRS.read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for f, r in table.items():
        ce, on = r["clause_end"], r["onset"]
        print(f"{f:22s} clause_end {ce['mean']:+.4f} [{ce['ci95'][0]:+.4f}, {ce['ci95'][1]:+.4f}] "
              f"({r['separates_clause_end']}); onset {on['mean']:+.4f}; cos anchor "
              f"{r['cos_anchor_axis']:+.2f}, V_harm {r['cos_V_harm_t_inst']:+.2f}")
    print(f">> rule ({RULE_AXIS} protect above rule, CI excluding 0): "
          f"{'care arm added to the T1.4 pod' if fires else 'no arm'}")


if __name__ == "__main__":
    main()
