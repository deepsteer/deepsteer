#!/usr/bin/env python3
"""Amendment 36 item 4: cell A′ for the SDF rungs (zero GPU, saved rollouts).

    /opt/homebrew/bin/python3.13 papers/cst_constitutional_steering/scripts/sdf_a2_reads.py

Cell A′'s 51 rule-right originals at k 8, per item P(violating) of record (`s0a_reads.spec_rule`'s
estimator, through `ctx_read.cell_a2`), each rung against **its own pod's session base**; a rung is
non-specific when the shift's CI lies entirely above 0.03 (Amendment 15). Read where the rollouts
exist: SDF pod 2 (`outputs/pilot_sdf2/gen`: the no-recovery rungs, both D-doc 1M draws) and the
re-read pod (`outputs/pilot_sdf/gen`: D-doc 1M without recovery, the recovery rungs). The kind-split
pod read the operator and plain cells only, so its four adapters have no A′ rollouts. Nothing is
pooled across draws or pods. Writes `outputs/sdf_a2_reads.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctx_read as C  # noqa: E402

CST = HERE.parent
PODS = {"pod2": ("outputs/pilot_sdf2/gen", ("d_neutral_1M", "d_doc_1M_k4", "d_doc_1M_draw2",
                                            "d_doc_03M", "d_chat_1M", "d_raw")),
        "reread": ("outputs/pilot_sdf/gen", ("d_doc_1M", "d_doc_1M_rec", "d_neutral_1M_rec",
                                             "d_chat_1M_rec", "r0"))}
KINDS = ("dn_stories", "dn_nostories", "dd_stories", "dd_nostories")


def main() -> None:
    res: dict = {"rule": "Amendment 36 item 4 (Amendment 15's bar, 0.03)", "pods": {},
                 "kind_split": {t: "no A′ rollouts (the pod read the operator and plain cells)"
                                for t in KINDS}}
    for pod, (gen, tags) in PODS.items():
        C.GEN = CST / gen
        res["pods"][pod] = {t: C.cell_a2(t) for t in tags if (C.GEN / t).exists()}
    (CST / "outputs/sdf_a2_reads.json").write_text(json.dumps(res, indent=1) + "\n")
    for pod, tags in res["pods"].items():
        print(f"== {pod}")
        for t, r in tags.items():
            if r:
                sh = r["shift"]
                print(f"  {t:17s} base {r['P_base']:.3f} rung {r['P_rung']:.3f} shift "
                      f"{sh['mean']:+.3f} [{sh['ci95'][0]:+.3f}, {sh['ci95'][1]:+.3f}] "
                      f"{'NON-SPECIFIC' if r['non_specific'] else 'specific'}")


if __name__ == "__main__":
    main()
