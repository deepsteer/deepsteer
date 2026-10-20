#!/usr/bin/env python3
"""τ of record for the §5.2 anchor hinge (Amendment 14 item 3; zero GPU on T1.1's bank).

    python3 papers/st_live_reference/scripts/compute_tau.py \
        papers/st_live_reference/outputs/t11/gpt_oss_20b

τ = the 75th percentile (``st_train.compute_tau``) of the anchor-axis margin <h, a> / |h| over
every banked ``clause_end`` position of the H1_train protect rollouts at block 12 (per position:
the hinge acts per position). Beside it, descriptive: the per-rollout-mean version and the same
quantile on the rule rollouts. Writes ``data/t14_tau_v1.json``.
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
import st_train as ST  # noqa: E402

OUT = HERE.parent / "data" / "t14_tau_v1.json"
CLASS = "clause_end"


def main() -> None:
    d = Path(sys.argv[1])
    rollouts = [json.loads(x) for x in (d / "gen" / "h1_rollouts_read.jsonl").open()]
    act = {i: CR.ACTION.get(r.get("norm_status")) for i, r in enumerate(rollouts)
           if r.get("split") == "train"}
    zd = np.load(CR.DIRS)
    a = zd[CR.AXES[0]].astype(np.float32)
    a /= np.linalg.norm(a)
    code = CR.CLASSES[CLASS]
    pos: dict[str, list[float]] = {"protect": [], "rule": []}
    per: dict[str, dict[int, list[float]]] = {"protect": {}, "rule": {}}
    for f in sorted((d / "bank").glob("bank_*.npz")):
        z = np.load(f)
        sel = (z["cls"] & code) > 0
        h = z["resid"][sel, CR.L_STAR_SLOT].astype(np.float32)
        m = (h @ a) / np.linalg.norm(h, axis=-1).clip(1e-6)
        for row, v in zip(z["row"][sel], m):
            k = act.get(int(row))
            if k in pos:
                pos[k].append(float(v))
                per[k].setdefault(int(row), []).append(float(v))
    tau = ST.compute_tau(np.array(pos["protect"]))
    out = {"rule": "ST_PLAN Amendment 14 item 3 (section 5.2)", "class": CLASS,
           "layer": "block 12 (hidden index 13)", "axis": CR.AXES[0], "quantile": ST.TAU_QUANTILE,
           "tau": tau, "n_positions": len(pos["protect"]), "n_rollouts": len(per["protect"]),
           "beside": {
               "per_rollout_mean": ST.compute_tau(np.array([np.mean(v) for v in
                                                           per["protect"].values()])),
               "rule_rollouts_per_position": ST.compute_tau(np.array(pos["rule"])),
               "protect_median": float(np.median(pos["protect"])),
               "rule_median": float(np.median(pos["rule"])),
               "share_of_protect_positions_below_tau": float(np.mean(np.array(pos["protect"])
                                                                     < tau))},
           "inputs": {"directions_sha256": hashlib.sha256(CR.DIRS.read_bytes()).hexdigest(),
                      "rollouts_sha256": hashlib.sha256(
                          (d / "gen" / "h1_rollouts_read.jsonl").read_bytes()).hexdigest()}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f">> tau {tau:.5f} (75th percentile of {len(pos['protect'])} protect clause_end "
          f"positions over {len(per['protect'])} rollouts); per-rollout-mean version "
          f"{out['beside']['per_rollout_mean']:.5f}")


if __name__ == "__main__":
    main()
